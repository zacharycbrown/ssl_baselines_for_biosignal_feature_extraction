import numpy as np
import os
import sys
# from zipfile import ZipFile, ZIP_DEFLATED
# import gc
import random
# from models import RPNet
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pickle as pkl


# RELATIVE POSITIONING (RP) CODE BLOCK #######################################################################
class RPDataset(torch.utils.data.Dataset):
    def __init__(self, cached_rp_dataset, path=None, total_points=None, tpos=None, tneg=None, window_size=None, sfreq=None, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        if cached_rp_dataset is not None:
            self.init_from_cached_data(cached_rp_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          tpos, 
                                          tneg, 
                                          window_size, 
                                          sfreq, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, tpos, tneg, window_size, sfreq, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        print("RPDataset.init_params_from_scratch(): data size == ", self.data.size)
        print("RPDataset.init_params_from_scratch(): data shape == ", self.data.shape)
        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.pairs, self.labels = self.get_samples_and_labels(size=total_points, tpos=tpos, tneg=tneg, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_rp_dataset):
        cached_dataset = None
        with open(cached_rp_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.data = cached_dataset['data']
        print("RPDataset.init_from_cached_data(): data size == ", self.data.size)
        print("RPDataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']
        self.total_windows = cached_dataset['total_windows']
        self.pairs = cached_dataset['pairs']
        self.labels = cached_dataset['labels']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'pairs': self.pairs,
                'labels': self.labels,
            }, outfile)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x1 = torch.from_numpy(self.data[self.pairs[index, 0], :, :]).float()
        x2 = torch.from_numpy(self.data[self.pairs[index, 1], :, :]).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()
        return x1, x2, y
    
    def get_samples_and_labels(self, size, tpos, tneg, window_size, max_pos_iter=10):
        """
        Gets the pairs of inputs (x1, x2) and output labels for the pretext task. 'Positive' windows are given a label of +1
        and negative windows are provided a label of -1. From Section 2.2 of https://arxiv.org/pdf/2007.16104.pdf 
        """
        pairs = np.zeros((size, 2), dtype=int)
        labels = np.zeros(size)

        for i in range(size):
            anchor_val = None
            second_val = None
            label = None

            if random.random() < 0.5: # decide whether or not to generate positive labeled data sample
                label = 1
                passed = False
                for _ in range(max_pos_iter):
                    anchor_val = np.random.randint(low=0, high=self.total_windows)
                    second_val = self.return_pos_index(index=anchor_val, tpos=tpos, window_size=window_size)
                    
                    if np.abs(self.start_times[anchor_val] - self.start_times[second_val]) <= tpos:
                        passed = True
                        break
                # we need to account for the fact that we could have some wrong time spans due to removing <1uV
                # reassign the labels here (and print if we do)
                # if(np.abs(self.start_times[anchor_val] - self.start_times[second_val]) > tpos):
                if not passed:
                    label = -1
            else:
                anchor_val = np.random.randint(low=0, high=self.total_windows)
                label = -1
                second_val = self.return_neg_index(anchor_val, tneg, window_size)
                # No need to check for mistakes since we can't return a bad negative window, still check
                if np.abs(self.start_times[anchor_val] - self.start_times[second_val]) < tneg:
                    print("RPDataset.get_samples_and_labels: ERROR - messed up negative label")

            pairs[i,0] = anchor_val
            pairs[i,1] = second_val
            labels[i] = label

        print("RPDataset.get_samples_and_labels: labels shape == ", labels.shape)
        return pairs, labels
    
    def return_pos_index(self, index, tpos, window_size):
        """
        returns the index of a random positive window given a starting (i.e. anchor) index
        """
        minimum = max(0, index-(tpos//window_size))
        maximum = min(len(self.data), index+(tpos//window_size)+1) # since non-inclusive
        return np.random.randint(minimum, maximum)
    
    def return_neg_index(self, index, tneg, window_size):
        """
        returns the index of a random negative window given a starting (i.e. anchor) index
        """
        midlow = max(0, index-(tneg//window_size))
        midhigh = min(len(self.data)-1, index+(tneg//window_size))
        assert (midlow > 0 or midhigh < len(self.data)) # check if it's even possible to return a negative index
        trial = np.random.randint(0, len(self.data))
        while trial >= midlow and trial <= midhigh:
            trial = np.random.randint(0, len(self.data)) # keep trying
        return trial

# RELATIVE POSITIONING (RP) CODE BLOCK #######################################################################

# TEMPORAL SHUFFLING (TS) CODE BLOCK #########################################################################
class TSDataset(torch.utils.data.Dataset):
    def __init__(self, cached_ts_dataset, path=None, total_points=None, tpos=None, tneg=None, window_size=None, sfreq=None, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        if cached_ts_dataset is not None:
            self.init_from_cached_data(cached_ts_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          tpos, 
                                          tneg, 
                                          window_size, 
                                          sfreq, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, tpos, tneg, window_size, sfreq, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        print("TSDataset.init_params_from_scratch(): data size == ", self.data.size)
        print("TSDataset.init_params_from_scratch(): data shape == ", self.data.shape)
        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.trios, self.labels = self.get_samples_and_labels(size=total_points, tpos=tpos, tneg=tneg, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_ts_dataset):
        cached_dataset = None
        with open(cached_ts_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.data = cached_dataset['data']
        print("TSDataset.init_from_cached_data(): data size == ", self.data.size)
        print("TSDataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']
        self.total_windows = cached_dataset['total_windows']
        self.trios = cached_dataset['trios']
        self.labels = cached_dataset['labels']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'trios': self.trios,
                'labels': self.labels,
            }, outfile)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x1 = torch.from_numpy(self.data[self.trios[index, 0], :, :]).float()
        x2 = torch.from_numpy(self.data[self.trios[index, 1], :, :]).float()
        x3 = torch.from_numpy(self.data[self.trios[index, 2], :, :]).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()
        return x1, x2, x3, y
    
    def get_samples_and_labels(self, size, tpos, tneg, window_size, max_pos_iter=10):
        """
        Gets the trios of inputs (x1, x2, x3) and output labels for the pretext task. 'Positive' windows are given a label of +1
        and negative windows are provided a label of -1. From Section 2.2 of https://arxiv.org/pdf/2007.16104.pdf 
        """
        trios = np.zeros((size, 3), dtype=int) # np.zeros((2*size, 3), dtype=int) # commented out because we are not reversing t'' and t in sampling
        labels = np.zeros(size)

        for i in range(size):
            anchor1_val = None
            tprime_val = None
            anchor2_val = None
            label = None

            if random.random() < 0.5: # decide whether or not to generate positive labeled data sample
                label = 1
                passed = False
                for _ in range(max_pos_iter):
                    anchor1_val = np.random.randint(low=0, high=self.total_windows)
                    anchor2_val = self.return_pos_index(index=anchor1_val, tpos=tpos, window_size=window_size)
                    
                    if np.abs(self.start_times[anchor1_val] - self.start_times[anchor2_val]) <= tpos and np.abs(anchor2_val-anchor1_val) > 1:
                        tprime_val = np.random.randint(low=anchor1_val+1, high=anchor2_val)
                        passed = True
                        break
                # we need to account for the fact that we could have some wrong time spans due to removing <1uV
                # reassign the labels here (and print if we do)
                # if(np.abs(self.start_times[anchor_val] - self.start_times[second_val]) > tpos):
                if not passed:
                    tprime_val = self.return_neg_index(anchor1_val, tneg, window_size)
                    label = -1
            else:
                label = -1
                passed = False
                for _ in range(max_pos_iter):
                    anchor1_val = np.random.randint(low=0, high=self.total_windows)
                    anchor2_val = self.return_pos_index(index=anchor1_val, tpos=tpos, window_size=window_size)
                    
                    if np.abs(self.start_times[anchor1_val] - self.start_times[anchor2_val]) <= tpos:
                        tprime_val = self.return_neg_index(anchor1_val, tneg, window_size)
                        passed = True
                        break
                    
                # No need to check for mistakes since we can't return a bad negative window, still check
                if np.abs(self.start_times[anchor1_val] - self.start_times[tprime_val]) < tneg:
                    print("TSDataset.get_samples_and_labels: ERROR - messed up negative label")
                if not passed:
                    tprime_val = self.return_neg_index(anchor1_val, tneg, window_size)

            trios[i,0] = anchor1_val
            trios[i,1] = tprime_val
            trios[i,2] = anchor2_val
            labels[i] = label

        print("TSDataset.get_samples_and_labels: labels shape == ", labels.shape)
        return trios, labels
    
    def return_pos_index(self, index, tpos, window_size):
        """
        returns the index of a random positive window given a starting (i.e. anchor1) index
        """
        maximum = min(len(self.data), index+(tpos//window_size)+1) # since non-inclusive
        return np.random.randint(index, maximum)
    
    def return_neg_index(self, index, tneg, window_size):
        """
        returns the index of a random negative window given a starting (i.e. anchor1) index
        """
        midlow = max(0, index-(tneg//window_size))
        midhigh = min(len(self.data)-1, index+(tneg//window_size))
        assert (midlow > 0 or midhigh < len(self.data)) # check if it's even possible to return a negative index
        trial = np.random.randint(0, len(self.data))
        while trial >= midlow and trial <= midhigh:
            trial = np.random.randint(0, len(self.data)) # keep trying
        return trial
# TEMPORAL SHUFFLING (TS) CODE BLOCK #########################################################################

# CONTRASTIVE PREDICTIVE CODING (CPC) CODE BLOCK #############################################################
# CONTRASTIVE PREDICTIVE CODING (CPC) CODE BLOCK #############################################################

# Data Loader Block
def load_SSL_Dataset(task_id, cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, 
                     window_size=3, sfreq=1000, data_folder_name="Mouse_Training_Data", 
                     data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                     val_portion=0.2, test_portion=0.1):
    assert task_id in ['RP', 'TS', 'CPC']
    root = os.path.join(data_folder_name, data_root_name, "")

    datasets_list = []

    print("load_SSL_Dataset: loading data")
    if cached_datasets_list_dir is not None:
        cached_datasets_file_names = os.listdir(cached_datasets_list_dir)
        for name in cached_datasets_file_names:
            curr_path = os.path.join(cached_datasets_list_dir, name, "")
            if task_id == 'RP':
                datasets_list.append(
                    RPDataset(curr_path, 
                            path=None, 
                            total_points=None, 
                            tpos=None, 
                            tneg=None, 
                            window_size=None, 
                            sfreq=None
                    )
                )
            elif task_id == 'TS':
                datasets_list.append(
                    TSDataset(curr_path, 
                            path=None, 
                            total_points=None, 
                            tpos=None, 
                            tneg=None, 
                            window_size=None, 
                            sfreq=None
                    )
                )
            elif task_id == 'CPC':
                pass
            else:
                raise ValueError("task_id == "+str(task_id)+" not recognized")
    else:
        f = open(os.path.join(file_names_list), 'r')
        lines = f.readlines()
        for line in lines:
            record_name = line.strip()
            print("load_SSL_Dataset: processing ", record_name)
            data_file = root+record_name+os.sep+record_name
            if task_id == 'RP':
                datasets_list.append(
                    RPDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            tpos=tpos_val, 
                            tneg=tneg_val, 
                            window_size=3, 
                            sfreq=sfreq
                    )
                )
            elif task_id == 'TS':
                datasets_list.append(
                    TSDataset(curr_path, 
                            path=None, 
                            total_points=None, 
                            tpos=None, 
                            tneg=None, 
                            window_size=None, 
                            sfreq=None
                    )
                )
            elif task_id == 'CPC':
                pass
            else:
                raise ValueError("task_id == "+str(task_id)+" not recognized")
        f.close()
    
    combined_dataset = torch.utils.data.ConcatDataset(datasets_list)

    data_len = len(combined_dataset)
    train_len = int(data_len*train_portion)
    val_len = int(data_len*val_portion)
    test_len = int(data_len - (train_len+val_len))

    # see https://pytorch.org/docs/stable/data.html
    train_set, val_set, test_set = torch.utils.data.random_split(combined_dataset, 
                                                                 [train_len, val_len, test_len], 
                                                                 generator=torch.Generator().manual_seed(0)
    )
    return train_set, val_set, test_set