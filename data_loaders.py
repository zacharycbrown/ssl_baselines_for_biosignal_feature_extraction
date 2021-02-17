import numpy as np
import os
import sys
# from zipfile import ZipFile, ZIP_DEFLATED
# import gc
import random
# from models import RPNet
import torch
from math import floor

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pickle as pkl
from sklearn.model_selection import GroupKFold
from torch.utils.data import TensorDataset
from scipy.signal import iirnotch, lfilter


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
        # print("RPDataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("RPDataset.init_params_from_scratch(): data shape == ", self.data.shape)
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
        # print("RPDataset.init_from_cached_data(): data size == ", self.data.size)
        # print("RPDataset.init_from_cached_data(): data shape == ", self.data.shape)
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

        # print("RPDataset.get_samples_and_labels: labels shape == ", labels.shape)
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
        # print("TSDataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("TSDataset.init_params_from_scratch(): data shape == ", self.data.shape)
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
        # print("TSDataset.init_from_cached_data(): data size == ", self.data.size)
        # print("TSDataset.init_from_cached_data(): data shape == ", self.data.shape)
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

        # print("TSDataset.get_samples_and_labels: labels shape == ", labels.shape)
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
class CPCDataset(torch.utils.data.Dataset):
    def __init__(self, cached_cpc_dataset, path=None, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=0.05, 
                 windowed_data_name="_Windowed_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        """
        see section 2.2 of https://arxiv.org/pdf/2007.16104.pdf for a description of Nc, Np, and Nb
        """
        if cached_cpc_dataset is not None:
            self.init_from_cached_data(cached_cpc_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          Nc, 
                                          Np, 
                                          Nb, 
                                          max_Nb_iters, 
                                          total_points_factor, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, Nc, Np, Nb, max_Nb_iters, total_points_factor, 
                                 windowed_data_name="_Windowed_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        # load data and start_times
        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("CPCDataset.init_params_from_scratch(): data shape == ", self.data.shape)
        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)

        self.total_windows = len(self.data)
        self.total_points = floor(total_points_factor*self.total_windows)

        self.Nc = Nc
        self.Np = Np
        self.Nb = Nb
        self.buffer_needed = self.Nc + self.Np
        self.max_Nb_iters = max_Nb_iters

        self.Xc_starts = self.get_Xc_starts()
        pass
    
    def init_from_cached_data(self, cached_cpc_dataset):
        cached_dataset = None
        with open(cached_cpc_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.data = cached_dataset['data']
        # print("CPCDataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']

        self.total_windows = cached_dataset['total_windows']
        self.total_points = cached_dataset['total_points']

        self.Nc = cached_dataset['Nc']
        self.Np = cached_dataset['Np']
        self.Nb = cached_dataset['Nb']
        self.buffer_needed = self.Nc + self.Np
        self.max_Nb_iters = cached_dataset['max_Nb_iters']

        self.Xc_starts = cached_dataset['Xc_starts']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'total_points': self.total_points,
                'Nc': self.Nc, 
                'Np': self.Np, 
                'Nb': self.Nb, 
                'max_Nb_iters': self.max_Nb_iters, 
                'Xc_starts': self.Xc_starts,
            }, outfile)
    
    def __len__(self):
        return len(self.Xc_starts)
    
    def __getitem__(self, index):
        xc_start_ind = self.Xc_starts[index]
        xp_start_ind = self.Xc_starts[index] + self.Nc
        xp_stop_ind = xp_start_ind + self.Np

        Xc = torch.from_numpy(self.data[xc_start_ind:xp_start_ind,:,:]).float()
        Xp = torch.from_numpy(self.data[xp_start_ind:xp_stop_ind,:,:]).float()
        Xb = [self.generate_negative_sample_list(self.Xc_starts[index]) for _ in range(xp_start_ind, xp_stop_ind)]
        Xb = torch.from_numpy(np.array(Xb)).float()

        return Xc, Xp, Xb
    
    def get_Xc_starts(self):
        """
        returns a random list of Xc starting points. We ensure we have enough buffer so we don't run out of 
        context windows and positive sample windows.
        """
        start_list = [np.random.randint(low=0, high=self.total_windows-self.buffer_needed) for _ in range(self.total_points)]
        return np.array(start_list)
    
    def generate_negative_sample_list(self, xc_start):
        """
        generates a random sample
        """
        return [self.get_random_Nb_sample(xc_start) for _ in range(self.Nb)]
    
    def get_random_Nb_sample(self, xc_start):
        rand_index = xc_start # will update in while loop
        count = 0
        while xc_start <= rand_index <= xc_start+self.buffer_needed:
            rand_index = np.random.randint(low=0, high=self.total_windows)
            count += 1
            if count > self.max_Nb_iters:
                raise Exception("CPCDataset.get_random_Nb_sample: Impossible to find a valid Nb sample, need to debug")
        
        if count == 0:
            raise Exception("CPCDataset.get_random_Nb_sample: never found an Nb sample, need to debug")

        return self.data[rand_index,:,:] # can't be in the range (start+buffer_needed)

# CONTRASTIVE PREDICTIVE CODING (CPC) CODE BLOCK #############################################################

# PHASESWAP (PS) CODE BLOCK ##################################################################################
class PSDataset(torch.utils.data.Dataset):
    def __init__(self, cached_ps_dataset, path=None, total_points=None, window_size=None, sfreq=None, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        if cached_ps_dataset is not None:
            self.init_from_cached_data(cached_ps_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          window_size, 
                                          sfreq, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, window_size, sfreq, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("PSDataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("PSDataset.init_params_from_scratch(): data shape == ", self.data.shape)
        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.pairs, self.labels = self.get_samples_and_labels(size=total_points, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_rp_dataset):
        cached_dataset = None
        with open(cached_rp_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.data = cached_dataset['data']
        # print("RPDataset.init_from_cached_data(): data size == ", self.data.size)
        # print("RPDataset.init_from_cached_data(): data shape == ", self.data.shape)
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
        y = torch.from_numpy(np.array([self.labels[index]])).float()

        x = None
        if self.pairs[index][1] is not np.nan: 
            x1 = torch.from_numpy(self.data[self.pairs[index][0], :, :]).float()
            x2 = torch.from_numpy(self.data[self.pairs[index][1], :, :]).float()
            x = self.phase_swap_operator(x1, x2)
            x = torch.from_numpy(x.real).float() # see https://numpy.org/doc/stable/reference/generated/numpy.real.html
        else:
            x = torch.from_numpy(self.data[self.pairs[index][0], :, :]).float()

        return x, y
    
    def get_samples_and_labels(self, size, window_size):
        """
        Gets the pairs of inputs (x1, x2) and output labels for the pretext task. 'Positive' windows are given a label of +1
        and negative windows are provided a label of -1. From Section 3 of arxiv.org/pdf/2009.07664.pdf 
        """
        pairs = [[None, None] for _ in range(size)] # np.zeros((size, 2), dtype=int)
        labels = np.zeros(size)

        for i in range(size):
            anchor_val = None
            second_val = None
            label = None

            anchor_val = np.random.randint(low=0, high=self.total_windows)
            if random.random() < 0.5: # decide whether or not to generate positive labeled data sample
                label = 1
                second_val = np.random.randint(low=0, high=self.total_windows)
            else:
                label = -1
                second_val = np.nan

            pairs[i][0] = anchor_val
            pairs[i][1] = second_val
            labels[i] = label

        # print("PSDataset.get_samples_and_labels: labels shape == ", labels.shape)
        return pairs, labels
    
    def phase_swap_operator(self, x1, x2):
        """
        Implementation of the PhaseSwap Operator from Section 3 of arxiv.org/pdf/2009.07664.pdf 
        see also 
         - https://stackoverflow.com/questions/47307862/why-do-scipy-and-numpy-fft-plots-look-different for reasoning behind using numpy over scipy
         - https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft for fft
         - https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft for ifft
         - https://stackoverflow.com/questions/43019136/how-to-get-phase-fft-of-a-signal-can-i-get-phase-in-time-domain for magnitude/phase computations
        """
        return np.fft.ifft(np.abs(np.fft.fft(x1))*np.angle(np.fft.fft(x2)))

# PHASESWAP (PS) CODE BLOCK ##################################################################################

# SeqCLR (SQ) CODE BLOCK #####################################################################################
class SQDataset(torch.utils.data.Dataset):
    def __init__(self, cached_ps_dataset, path=None, total_points=None, window_size=None, sfreq=1000, bw=5, 
                 randomized_augmentation=False, num_channels=11, temporal_len=3000, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        if cached_ps_dataset is not None:
            self.init_from_cached_data(cached_ps_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          window_size, 
                                          sfreq, 
                                          bw, 
                                          randomized_augmentation, 
                                          num_channels, 
                                          temporal_len, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, window_size, sfreq, bw, 
                                 randomized_augmentation, num_channels, temporal_len, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        self.available_augmentations = {
            "amplitude_scale": [0.5, 2], 
            "time_shift": [-50, 50], 
            "DC_shift": [-10, 10], 
            "zero-masking": [0, 150], 
            "additive_Gaussian_noise": [0, 0.2], 
            "band-stop_filter": [2.8, 82.5], 
        }
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.NUM_AUGMENTATIONS = 2
        self.NUM_CHANNELS = num_channels
        self.TEMPORAL_LEN = temporal_len
        self.SFREQ = sfreq
        self.BW = bw # band width (?) see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html

        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("SQDataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("SQDataset.init_params_from_scratch(): data shape == ", self.data.shape)

        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.randomized_augmentation = randomized_augmentation
        self.pairs, self.labels = self.get_samples_and_labels(size=total_points, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_rp_dataset):
        cached_dataset = None
        with open(cached_rp_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.available_augmentations = cached_dataset['available_augmentations']
        self.TEMPORAL_DIM = cached_dataset['TEMPORAL_DIM']
        self.CHANNEL_DIM = cached_dataset['CHANNEL_DIM']
        self.NUM_AUGMENTATIONS = cached_dataset['NUM_AUGMENTATIONS']
        self.NUM_CHANNELS = cached_dataset['NUM_CHANNELS']
        self.TEMPORAL_LEN = cached_dataset['TEMPORAL_LEN']
        self.SFREQ = cached_dataset['SFREQ']
        self.BW = cached_dataset['BW']
        self.data = cached_dataset['data']
        # print("SQDataset.init_from_cached_data(): data size == ", self.data.size)
        # print("SQDataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']
        self.total_windows = cached_dataset['total_windows']
        self.randomized_augmentation = cached_dataset['randomized_augmentation']
        self.pairs = cached_dataset['pairs']
        self.labels = cached_dataset['labels']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'available_augmentations': self.available_augmentations, 
                'TEMPORAL_DIM': self.TEMPORAL_DIM, 
                'CHANNEL_DIM': self.CHANNEL_DIM, 
                'NUM_AUGMENTATIONS': self.NUM_AUGMENTATIONS, 
                'NUM_CHANNELS': self.NUM_CHANNELS, 
                'TEMPORAL_LEN': self.TEMPORAL_LEN, 
                'SFREQ': self.SFREQ, 
                'BW': self.BW, 
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'randomized_augmentation': self.randomized_augmentation, 
                'pairs': self.pairs,
                'labels': self.labels,
            }, outfile)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        x = self.data[self.pairs[index][0], :, :]
        x_aug = self.data[self.pairs[index][0], :, :]
        if self.pairs[index][1] is not None: 
            # print("SQDataset.__getitem__: using STORED augmentations set self.pairs[index][1] == ", self.pairs[index][1])
            x_aug = self.apply_augmentations(x_aug, self.pairs[index][1])
        else:
            # print("SQDataset.__getitem__: using NEW augmentations set because self.pairs[index][1] == ", self.pairs[index][1])
            curr_augmentations = self.get_augmentation_set()
            # print("SQDataset.__getitem__: using NEW augmentations set curr_augmentations == ", curr_augmentations)
            x_aug = self.apply_augmentations(x_aug, curr_augmentations)

        x = torch.from_numpy(x).float()
        x_aug = torch.from_numpy(x_aug).float()
        return x, x_aug
    
    def get_samples_and_labels(self, size, window_size):
        """
        Gets the pairs of inputs [x, [t1, t2]] and output labels for the pretext task. 
        see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        pairs = [[None, None] for _ in range(size)] # np.zeros((size, 2), dtype=int)
        labels = None

        for i in range(size):
            anchor_val = np.random.randint(low=0, high=self.total_windows)
            second_val = None
            if not self.randomized_augmentation: # decide whether or not to save augmentation strategy for curr sample
                second_val = self.get_augmentation_set()
            # else:
            #     second_val = None

            pairs[i][0] = anchor_val
            pairs[i][1] = second_val

        # print("PSDataset.get_samples_and_labels: labels shape == ", labels.shape)
        return pairs, labels
    
    def get_augmentation_set(self):
        """
        see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        augmentation_set = [] # [dict()]*self.NUM_CHANNELS
        
        for j in range(self.NUM_CHANNELS):
            augmentation_set.append(dict())
            selected_augmentations = random.sample(list(self.available_augmentations.keys()), self.NUM_AUGMENTATIONS) # see https://pynative.com/python-random-sample/#:~:text=Python's%20random%20module%20provides%20random,it%20random%20sampling%20without%20replacement.
            assert len(selected_augmentations) == 2
            # print("selected_augmentations == ", selected_augmentations)
            # print("SQDataset.get_augmentation_set: len(selected_augmentations) == 2")
            counter = 0
            for _, curr_augmentation in enumerate(selected_augmentations):
                curr_augmentation_val = None

                if curr_augmentation in ['amplitude_scale', 'DC_shift', 'additive_Gaussian_noise', 'band-stop_filter']: # augmentation that requires float val
                    curr_augmentation_val = random.uniform(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range

                elif curr_augmentation in ['time_shift', 'zero-masking']: # augmentation that requires int val
                    curr_augmentation_val = random.randint(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
                    if curr_augmentation == 'zero-masking':
                        curr_augmentation_val = [curr_augmentation_val, random.randint(0, self.TEMPORAL_LEN-1)]

                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for value sampling")

                augmentation_set[j][curr_augmentation] = curr_augmentation_val
                counter += 1
            # print("augmentation_set == ", augmentation_set)
            # print("augmentation_set[j].keys() == ", augmentation_set[j].keys())
            assert len(list(augmentation_set[j].keys())) == 2
            assert counter == 2
        # raise NotImplementedError()
        return augmentation_set
    
    def apply_augmentations(self, x, augmentations):
        """
        see Section 2.2 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        # print("\n#####################\nSQDataset.apply_augmentations: x shape == ", x.shape)
        # print("SQDataset.apply_augmentations: augmentations == ", augmentations)
        assert len(augmentations) == 11
        for j, curr_augmentation_set in enumerate(augmentations):
            assert len(list(curr_augmentation_set.keys())) == 2
            # print("SQDataset.apply_augmentations: len(list(curr_augmentation_set.keys())) == 2")
            # print("\tSQDataset.apply_augmentations: curr_augmentation_set == ", curr_augmentation_set)
            for _, curr_augmentation in enumerate(list(curr_augmentation_set.keys())):
                # print("\t\tSQDataset.apply_augmentations: curr_augmentation == ", curr_augmentation)
                curr_augmentation_val = curr_augmentation_set[curr_augmentation]

                if curr_augmentation == 'amplitude_scale':
                    # print("\n\t\t\tSQDataset.apply_augmentations: NOW APPLYING amplitude_scale AUGMENTATION")
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    x[:,j] = curr_augmentation_val * x[:,j]
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                elif curr_augmentation == 'DC_shift':
                    # print("\n\t\t\tSQDataset.apply_augmentations: NOW APPLYING DC_shift AUGMENTATION")
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    x[:,j] = x[:,j] + curr_augmentation_val
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                elif curr_augmentation == 'additive_Gaussian_noise':
                    # print("\n\t\t\tSQDataset.apply_augmentations: NOW APPLYING additive_Gaussian_noise AUGMENTATION")
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    # print("\t\t\tSQDataset.apply_augmentations: np.random.normal(0, curr_augmentation_val, x[:,j].shape) shape == ", np.random.normal(0, curr_augmentation_val, x[:,j].shape).shape)
                    x[:,j] = x[:,j] + np.random.normal(0, curr_augmentation_val, x[:,j].shape)# see https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python and https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                elif curr_augmentation == 'band-stop_filter':
                    """
                    see:
                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
                        https://www.programcreek.com/python/example/115815/scipy.signal.iirnotch
                        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                    """
                    # print("\n\t\t\tSQDataset.apply_augmentations: NOW APPLYING band-stop_filter AUGMENTATION")
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    # print("\t\t\tSQDataset.apply_augmentations: curr_augmentation_val == ", curr_augmentation_val)
                    # print("\t\t\tSQDataset.apply_augmentations: curr_augmentation_val/self.BW == ", curr_augmentation_val/self.BW)
                    # print("\t\t\tSQDataset.apply_augmentations: self.SFREQ == ", self.SFREQ)
                    b, a = iirnotch(curr_augmentation_val, curr_augmentation_val/self.BW, self.SFREQ)
                    # print("\t\t\tSQDataset.apply_augmentations: b == ", b)
                    # print("\t\t\tSQDataset.apply_augmentations: a == ", a)
                    x[:,j] = lfilter(b, a, x[:,j])
                    # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                elif curr_augmentation == 'time_shift':
                    # print("\n\t\t\tSQDataset.apply_augmentations: NOW APPLYING time_shift AUGMENTATION")
                    if curr_augmentation_val != 0:
                        # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        new_signal = np.zeros(x[:,j].shape)
                        # print("\t\t\tSQDataset.apply_augmentations: new_signal shape == ", new_signal.shape)
                        # print("\t\t\tSQDataset.apply_augmentations: curr_augmentation_val == ", curr_augmentation_val)
                        if curr_augmentation_val < 0:
                            # print("\t\t\tSQDataset.apply_augmentations: new_signal[:curr_augmentation_val] shape == ", new_signal[:curr_augmentation_val].shape)
                            # print("\t\t\tSQDataset.apply_augmentations: x[np.abs(curr_augmentation_val):,j] shape == ", x[np.abs(curr_augmentation_val):,j].shape)
                            new_signal[:curr_augmentation_val] = x[np.abs(curr_augmentation_val):,j]
                            # print("\t\t\tSQDataset.apply_augmentations: new_signal[curr_augmentation_val:] shape == ", new_signal[curr_augmentation_val:].shape)
                            # print("\t\t\tSQDataset.apply_augmentations: x[:np.abs(curr_augmentation_val),j] shape == ", x[:np.abs(curr_augmentation_val),j].shape)
                            new_signal[curr_augmentation_val:] = x[:np.abs(curr_augmentation_val),j]
                        else:
                            # print("\t\t\tSQDataset.apply_augmentations: new_signal[:curr_augmentation_val] shape == ", new_signal[:curr_augmentation_val].shape)
                            # print("\t\t\tSQDataset.apply_augmentations: x[-curr_augmentation_val:,j] shape == ", x[-curr_augmentation_val:,j].shape)
                            new_signal[:curr_augmentation_val] = x[-curr_augmentation_val:,j]
                            # print("\t\t\tSQDataset.apply_augmentations: new_signal[curr_augmentation_val:] shape == ", new_signal[curr_augmentation_val:].shape)
                            # print("\t\t\tSQDataset.apply_augmentations: x[:-curr_augmentation_val,j] shape == ", x[:-curr_augmentation_val,j].shape)
                            new_signal[curr_augmentation_val:] = x[:-curr_augmentation_val,j]
                        # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # print("\t\t\tSQDataset.apply_augmentations: new_signal shape == ", new_signal.shape)
                        x[:,j] = new_signal
                        # print("\t\t\tSQDataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    # else:
                    #     print("\t\t\tSQDataset.apply_augmentations: curr_augmentation_val == 0 -> SKIPPING")
                elif curr_augmentation == 'zero-masking': 
                    # print("\n\t\t\tSQDataset.apply_augmentations: NOW APPLYING zero-masking AUGMENTATION")
                    # print("\t\t\tSQDataset.apply_augmentations: x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] shape == ", x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j].shape)
                    x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] = 0.
                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for application")
        
        # print("SQDataset.apply_augmentations: x shape == ", x.shape)
        return x

# SeqCLR (SQ) CODE BLOCK #####################################################################################
# SACL (SA) CODE BLOCK #######################################################################################
class SADataset(torch.utils.data.Dataset): # NOTE THAT IF perturb_orig_signal==False AND individual_id is None AND num_individuals_in_label is None, THEN THIS DATASET *SHOULD* BE EQUIVALENT TO SQDataset
    def __init__(self, cached_ps_dataset, path=None, total_points=None, window_size=None, sfreq=1000, bw=5, 
                 randomized_augmentation=False, num_channels=11, temporal_len=3000, NUM_AUGMENTATIONS=2, perturb_orig_signal=True, individual_id=None, num_individuals_in_label=None, 
                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", windowed_start_time_name="_Windowed_StartTime.npy"):
        if cached_ps_dataset is not None:
            self.init_from_cached_data(cached_ps_dataset)
        else:
            self.init_params_from_scratch(path, 
                                          total_points, 
                                          window_size, 
                                          sfreq, 
                                          bw, 
                                          randomized_augmentation, 
                                          num_channels, 
                                          temporal_len, 
                                          NUM_AUGMENTATIONS, 
                                          perturb_orig_signal, 
                                          individual_id, 
                                          num_individuals_in_label, 
                                          windowed_data_name, 
                                          windowed_start_time_name
            )
        pass 

    def init_params_from_scratch(self, path, total_points, window_size, sfreq, bw, 
                                 randomized_augmentation, num_channels, temporal_len, 
                                 NUM_AUGMENTATIONS,  perturb_orig_signal, individual_id, num_individuals_in_label, 
                                 windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                 windowed_start_time_name="_Windowed_StartTime.npy"):
        self.available_augmentations = {
            "amplitude_scale": [0.5, 2], 
            "time_shift": [-50, 50], 
            "DC_shift": [-10, 10], 
            "zero-masking": [0, 150], 
            "additive_Gaussian_noise": [0, 0.2], 
            "band-stop_filter": [2.8, 82.5], 
        }
        self.TEMPORAL_DIM = 0
        self.CHANNEL_DIM = 1
        self.NUM_AUGMENTATIONS = NUM_AUGMENTATIONS
        self.perturb_orig_signal = perturb_orig_signal
        self.individual_id = individual_id
        self.num_individuals_in_label = num_individuals_in_label
        self.NUM_CHANNELS = num_channels
        self.TEMPORAL_LEN = temporal_len
        self.SFREQ = sfreq
        self.BW = bw # band width (?) see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html

        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("SADataset.init_params_from_scratch(): data size == ", self.data.size)
        # print("SADataset.init_params_from_scratch(): data shape == ", self.data.shape)

        data_path = path + windowed_start_time_name
        self.start_times = np.load(data_path)
        self.total_windows = len(self.data)
        self.randomized_augmentation = randomized_augmentation
        self.pairs, self.labels = self.get_samples_and_labels(size=total_points, window_size=window_size)
        pass
    
    def init_from_cached_data(self, cached_rp_dataset):
        cached_dataset = None
        with open(cached_rp_dataset, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.available_augmentations = cached_dataset['available_augmentations']
        self.TEMPORAL_DIM = cached_dataset['TEMPORAL_DIM']
        self.CHANNEL_DIM = cached_dataset['CHANNEL_DIM']
        self.NUM_AUGMENTATIONS = cached_dataset['NUM_AUGMENTATIONS']
        self.perturb_orig_signal = cached_dataset['perturb_orig_signal']
        self.individual_id = cached_dataset['individual_id']
        self.num_individuals_in_label = cached_dataset['num_individuals_in_label']
        self.NUM_CHANNELS = cached_dataset['NUM_CHANNELS']
        self.TEMPORAL_LEN = cached_dataset['TEMPORAL_LEN']
        self.SFREQ = cached_dataset['SFREQ']
        self.BW = cached_dataset['BW']
        self.data = cached_dataset['data']
        # print("SADataset.init_from_cached_data(): data size == ", self.data.size)
        # print("SADataset.init_from_cached_data(): data shape == ", self.data.shape)
        self.start_times = cached_dataset['start_times']
        self.total_windows = cached_dataset['total_windows']
        self.randomized_augmentation = cached_dataset['randomized_augmentation']
        self.pairs = cached_dataset['pairs']
        self.labels = cached_dataset['labels']

        del cached_dataset
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'available_augmentations': self.available_augmentations, 
                'TEMPORAL_DIM': self.TEMPORAL_DIM, 
                'CHANNEL_DIM': self.CHANNEL_DIM, 
                'NUM_AUGMENTATIONS': self.NUM_AUGMENTATIONS, 
                'perturb_orig_signal': self.perturb_orig_signal, 
                'individual_id': self.individual_id, 
                'num_individuals_in_label': self.num_individuals_in_label, 
                'NUM_CHANNELS': self.NUM_CHANNELS, 
                'TEMPORAL_LEN': self.TEMPORAL_LEN, 
                'SFREQ': self.SFREQ, 
                'BW': self.BW, 
                'data': self.data,
                'start_times': self.start_times,
                'total_windows': self.total_windows,
                'randomized_augmentation': self.randomized_augmentation, 
                'pairs': self.pairs,
                'labels': self.labels,
            }, outfile)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        x_t1 = self.data[self.pairs[index][0], :, :]
        x_t2 = self.data[self.pairs[index][0], :, :]

        # APPLY RELEVANT TRANSFORMATION T1 TO X_T1
        if self.perturb_orig_signal:
            if self.pairs[index][1][0] is not None: 
                # print("SADataset.__getitem__: using STORED augmentations set self.pairs[index][1][0] == ", self.pairs[index][1][0])
                x_t1 = self.apply_augmentations(x_t1, self.pairs[index][1][0])
            else:
                # print("SADataset.__getitem__: using NEW augmentations set because self.pairs[index][1][0] == ", self.pairs[index][1][0])
                curr_augmentations = self.get_augmentation_set()
                # print("SADataset.__getitem__: using NEW augmentations set curr_augmentations == ", curr_augmentations)
                x_t1 = self.apply_augmentations(x_t1, curr_augmentations)

        # APPLY RELEVANT TRANSFORMATION T2 TO X_T2
        if self.pairs[index][1][1] is not None: 
            # print("SADataset.__getitem__: using STORED augmentations set self.pairs[index][1][1] == ", self.pairs[index][1][1])
            x_t2 = self.apply_augmentations(x_t2, self.pairs[index][1][1])
        else:
            # print("SADataset.__getitem__: using NEW augmentations set because self.pairs[index][1][1] == ", self.pairs[index][1][1])
            curr_augmentations = self.get_augmentation_set()
            # print("SADataset.__getitem__: using NEW augmentations set curr_augmentations == ", curr_augmentations)
            x_t2 = self.apply_augmentations(x_t2, curr_augmentations)

        x_t1 = torch.from_numpy(x_t1).float()
        x_t2 = torch.from_numpy(x_t2).float()

        # y = np.zeros(num_individuals_in_label)
        # y[self.individual_id] = 1.
        y = torch.from_numpy(self.labels).float()

        return x_t1, x_t2, y
    
    def get_samples_and_labels(self, size, window_size):
        """
        Gets the pairs of inputs [x, [t1, t2]] and output labels for the pretext task. 

        Each pair, representing the signal to be transformed by randomly chosen t1 and t2, is given the same label, 
            corresponding to the test subject from which the signals were taken.

        see Sections 3.1 and 3.2 of arxiv.org/pdf/2007.04871.pdf
        """
        pairs = [[None, None] for _ in range(size)] # np.zeros((size, 2), dtype=int)
        labels = np.zeros(self.num_individuals_in_label)
        labels[self.individual_id] = 1.

        for i in range(size):
            anchor_val = np.random.randint(low=0, high=self.total_windows)
            second_val = [None, None]
            if not self.randomized_augmentation: # decide whether or not to save augmentation strategy for curr sample
                if self.perturb_orig_signal:
                    second_val[0] = self.get_augmentation_set()
                second_val[1] = self.get_augmentation_set()
            # else:
            #     second_val = [None, None]

            pairs[i][0] = anchor_val
            pairs[i][1] = second_val

        # print("SADataset.get_samples_and_labels: labels shape == ", labels.shape)
        return pairs, labels
    
    def get_augmentation_set(self):
        """
        see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        augmentation_set = [] # [dict()]*self.NUM_CHANNELS
        
        for j in range(self.NUM_CHANNELS):
            augmentation_set.append(dict())
            selected_augmentations = random.sample(list(self.available_augmentations.keys()), self.NUM_AUGMENTATIONS) # see https://pynative.com/python-random-sample/#:~:text=Python's%20random%20module%20provides%20random,it%20random%20sampling%20without%20replacement.
            assert len(selected_augmentations) == self.NUM_AUGMENTATIONS
            # print("SADataset.get_augmentation_set: selected_augmentations == ", selected_augmentations)
            # print("SADataset.get_augmentation_set: len(selected_augmentations) == 2")
            counter = 0
            for _, curr_augmentation in enumerate(selected_augmentations):
                curr_augmentation_val = None

                if curr_augmentation in ['amplitude_scale', 'DC_shift', 'additive_Gaussian_noise', 'band-stop_filter']: # augmentation that requires float val
                    curr_augmentation_val = random.uniform(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range

                elif curr_augmentation in ['time_shift', 'zero-masking']: # augmentation that requires int val
                    curr_augmentation_val = random.randint(self.available_augmentations[curr_augmentation][0], self.available_augmentations[curr_augmentation][1]) # see https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
                    if curr_augmentation == 'zero-masking':
                        curr_augmentation_val = [curr_augmentation_val, random.randint(0, self.TEMPORAL_LEN-1)]

                else:
                    raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for value sampling")

                augmentation_set[j][curr_augmentation] = curr_augmentation_val
                counter += 1
            # print("SADataset.get_augmentation_set: augmentation_set == ", augmentation_set)
            # print("SADataset.get_augmentation_set: augmentation_set[j].keys() == ", augmentation_set[j].keys())
            assert len(list(augmentation_set[j].keys())) == self.NUM_AUGMENTATIONS
            assert counter == self.NUM_AUGMENTATIONS
        # raise NotImplementedError()
        return augmentation_set
    
    def apply_augmentations(self, x, augmentations):
        """
        see Section 2.2 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
        """
        # print("\n#####################\nSADataset.apply_augmentations: x shape == ", x.shape)
        if augmentations is not None:
            # print("SADataset.apply_augmentations: augmentations == ", augmentations)
            assert len(augmentations) == 11
            for j, curr_augmentation_set in enumerate(augmentations):
                assert len(list(curr_augmentation_set.keys())) == 2
                # print("SADataset.apply_augmentations: len(list(curr_augmentation_set.keys())) == 2")
                # print("\tSADataset.apply_augmentations: curr_augmentation_set == ", curr_augmentation_set)
                for _, curr_augmentation in enumerate(list(curr_augmentation_set.keys())):
                    # print("\t\tSADataset.apply_augmentations: curr_augmentation == ", curr_augmentation)
                    curr_augmentation_val = curr_augmentation_set[curr_augmentation]

                    if curr_augmentation == 'amplitude_scale':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING amplitude_scale AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        x[:,j] = curr_augmentation_val * x[:,j]
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'DC_shift':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING DC_shift AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        x[:,j] = x[:,j] + curr_augmentation_val
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'additive_Gaussian_noise':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING additive_Gaussian_noise AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # print("\t\t\tSADataset.apply_augmentations: np.random.normal(0, curr_augmentation_val, x[:,j].shape) shape == ", np.random.normal(0, curr_augmentation_val, x[:,j].shape).shape)
                        x[:,j] = x[:,j] + np.random.normal(0, curr_augmentation_val, x[:,j].shape)# see https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python and https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'band-stop_filter':
                        """
                        see:
                            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
                            https://www.programcreek.com/python/example/115815/scipy.signal.iirnotch
                            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
                        """
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING band-stop_filter AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val == ", curr_augmentation_val)
                        # print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val/self.BW == ", curr_augmentation_val/self.BW)
                        # print("\t\t\tSADataset.apply_augmentations: self.SFREQ == ", self.SFREQ)
                        b, a = iirnotch(curr_augmentation_val, curr_augmentation_val/self.BW, self.SFREQ)
                        # print("\t\t\tSADataset.apply_augmentations: b == ", b)
                        # print("\t\t\tSADataset.apply_augmentations: a == ", a)
                        x[:,j] = lfilter(b, a, x[:,j])
                        # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                    elif curr_augmentation == 'time_shift':
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING time_shift AUGMENTATION")
                        if curr_augmentation_val != 0:
                            # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                            new_signal = np.zeros(x[:,j].shape)
                            # print("\t\t\tSADataset.apply_augmentations: new_signal shape == ", new_signal.shape)
                            # print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val == ", curr_augmentation_val)
                            if curr_augmentation_val < 0:
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[:curr_augmentation_val] shape == ", new_signal[:curr_augmentation_val].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[np.abs(curr_augmentation_val):,j] shape == ", x[np.abs(curr_augmentation_val):,j].shape)
                                new_signal[:curr_augmentation_val] = x[np.abs(curr_augmentation_val):,j]
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[curr_augmentation_val:] shape == ", new_signal[curr_augmentation_val:].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[:np.abs(curr_augmentation_val),j] shape == ", x[:np.abs(curr_augmentation_val),j].shape)
                                new_signal[curr_augmentation_val:] = x[:np.abs(curr_augmentation_val),j]
                            else:
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[:curr_augmentation_val] shape == ", new_signal[:curr_augmentation_val].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[-curr_augmentation_val:,j] shape == ", x[-curr_augmentation_val:,j].shape)
                                new_signal[:curr_augmentation_val] = x[-curr_augmentation_val:,j]
                                # print("\t\t\tSADataset.apply_augmentations: new_signal[curr_augmentation_val:] shape == ", new_signal[curr_augmentation_val:].shape)
                                # print("\t\t\tSADataset.apply_augmentations: x[:-curr_augmentation_val,j] shape == ", x[:-curr_augmentation_val,j].shape)
                                new_signal[curr_augmentation_val:] = x[:-curr_augmentation_val,j]
                            # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                            # print("\t\t\tSADataset.apply_augmentations: new_signal shape == ", new_signal.shape)
                            x[:,j] = new_signal
                            # print("\t\t\tSADataset.apply_augmentations: x[:,j] shape == ", x[:,j].shape)
                        # else:
                        #     print("\t\t\tSADataset.apply_augmentations: curr_augmentation_val == 0 -> SKIPPING")
                    elif curr_augmentation == 'zero-masking': 
                        # print("\n\t\t\tSADataset.apply_augmentations: NOW APPLYING zero-masking AUGMENTATION")
                        # print("\t\t\tSADataset.apply_augmentations: x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] shape == ", x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j].shape)
                        x[curr_augmentation_val[1]:curr_augmentation_val[1]+curr_augmentation_val[0], j] = 0.
                    else:
                        raise NotImplementedError("curr_augmentation == "+str(curr_augmentation)+" not recognized for application")
        
        # print("SADataset.apply_augmentations: x shape == ", x.shape)
        return x

# SACL (SA) CODE BLOCK #######################################################################################

# UPSTREAM DATA LOADER BLOCK #################################################################################
def load_SSL_Dataset(task_id, cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, 
                     window_size=3, sfreq=1000, Nc=10, Np=16, Nb=4, max_Nb_iters=1000, total_points_factor=0.05, 
                     bw=5, randomized_augmentation=False, num_channels=11, temporal_len=3000, # items relevant to SQ/SA data loading
                     NUM_AUGMENTATIONS=2, perturb_orig_signal=True, # items unique to SA data loading
                     windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                     windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                     data_root_name="Windowed_Data", file_names_list="training_names.txt", 
                     train_portion=0.7, val_portion=0.2, test_portion=0.1):
    assert task_id in ['RP', 'TS', 'CPC', 'PS', 'SQ', 'SA']
    root = os.path.join(data_folder_name, data_root_name, "")

    datasets_list = []

    print("load_SSL_Dataset: loading data")
    if cached_datasets_list_dir is not None:
        cached_datasets_file_names = os.listdir(cached_datasets_list_dir)
        for name in cached_datasets_file_names:
            curr_path = os.path.join(cached_datasets_list_dir, name, "")
            if task_id == 'RP':
                datasets_list.append(RPDataset(curr_path))
            elif task_id == 'TS':
                datasets_list.append(TSDataset(curr_path))
            elif task_id == 'CPC':
                datasets_list.append(CPCDataset(curr_path))
            elif task_id == 'PS':
                datasets_list.append(PSDataset(curr_path))
            elif task_id == 'SQ':
                datasets_list.append(SQDataset(curr_path))
            elif task_id == 'SA':
                datasets_list.append(SQDataset(curr_path))
            else:
                raise ValueError("load_SSL_Dataset: task_id == "+str(task_id)+" not recognized")
    else:
        f = open(os.path.join(file_names_list), 'r')
        lines = f.readlines()
        print("load_SSL_Dataset: ASSUMING EACH LINE IN ", str(file_names_list), " REPRESENTS A DIFFERENT, INDIVIDUAL TEST SUBJECT")
        for i, line in enumerate(lines):
            record_name = line.strip()
            # print("load_SSL_Dataset: processing ", record_name)
            data_file = root+record_name+os.sep+record_name
            if task_id == 'RP':
                datasets_list.append(
                    RPDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            tpos=tpos_val, 
                            tneg=tneg_val, 
                            window_size=window_size, 
                            sfreq=sfreq, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'TS':
                datasets_list.append(
                    TSDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            tpos=tpos_val, 
                            tneg=tneg_val, 
                            window_size=window_size, 
                            sfreq=sfreq, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'CPC':
                datasets_list.append(
                    CPCDataset(None, 
                            path=data_file, 
                            Nc=Nc, 
                            Np=Np, 
                            Nb=Nb, # this used to be 2 not 4, but 4 would work better
                            max_Nb_iters=max_Nb_iters, 
                            total_points_factor=total_points_factor, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'PS':
                datasets_list.append(
                    PSDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            window_size=window_size, 
                            sfreq=sfreq, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'SQ':
                # print("task_id == SQ")
                datasets_list.append(
                    SQDataset(None, 
                              path=data_file, 
                              total_points=total_points_val, 
                              window_size=window_size, 
                              sfreq=sfreq, 
                              bw=bw, 
                              randomized_augmentation=randomized_augmentation, 
                              num_channels=num_channels, 
                              temporal_len=temporal_len, 
                              windowed_data_name=windowed_data_name, 
                              windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'SA':
                # print("task_id == SA")
                datasets_list.append(
                    SADataset(None, 
                              path=data_file, 
                              total_points=total_points_val, 
                              window_size=window_size, 
                              sfreq=sfreq, 
                              bw=bw, 
                              randomized_augmentation=randomized_augmentation, 
                              num_channels=num_channels, 
                              temporal_len=temporal_len, 
                              NUM_AUGMENTATIONS=NUM_AUGMENTATIONS, 
                              perturb_orig_signal=perturb_orig_signal, 
                              individual_id=i, 
                              num_individuals_in_label=len(lines), 
                              windowed_data_name=windowed_data_name, 
                              windowed_start_time_name=windowed_start_time_name
                    )
                )
            else:
                raise ValueError("load_SSL_Dataset: task_id == "+str(task_id)+" not recognized")
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

def load_SSL_Dataset_Based_On_Subjects(task_id, cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, 
                                        window_size=3, sfreq=1000, Nc=10, Np=16, Nb=4, max_Nb_iters=1000, total_points_factor=0.05, 
                                        bw=5, randomized_augmentation=False, num_channels=11, temporal_len=3000, # items relevant to SQ/SA data loading
                                        NUM_AUGMENTATIONS=2, perturb_orig_signal=True, # items unique to SA data loading
                                        windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                                        windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                                        data_root_name="Windowed_Data", file_names_list="training_names.txt", 
                                        train_portion=0.7, val_portion=0.2, test_portion=0.1, random_seed=0):
    assert task_id in ['RP', 'TS', 'CPC', 'PS', 'SQ', 'SA']
    root = os.path.join(data_folder_name, data_root_name, "")

    datasets_list = []

    print("load_SSL_Dataset_Based_On_Subjects: loading data")
    if cached_datasets_list_dir is not None:
        cached_datasets_file_names = os.listdir(cached_datasets_list_dir)
        for name in cached_datasets_file_names:
            curr_path = os.path.join(cached_datasets_list_dir, name, "")
            if task_id == 'RP':
                datasets_list.append(RPDataset(curr_path))
            elif task_id == 'TS':
                datasets_list.append(TSDataset(curr_path))
            elif task_id == 'CPC':
                datasets_list.append(CPCDataset(curr_path))
            elif task_id == 'PS':
                datasets_list.append(PSDataset(curr_path))
            elif task_id == 'SQ':
                datasets_list.append(SQDataset(curr_path))
            elif task_id == 'SA':
                datasets_list.append(SQDataset(curr_path))
            else:
                raise ValueError("load_SSL_Dataset_Based_On_Subjects: task_id == "+str(task_id)+" not recognized")
    else:
        f = open(os.path.join(file_names_list), 'r')
        lines = f.readlines()
        print("load_SSL_Dataset_Based_On_Subjects: ASSUMING EACH LINE IN ", str(file_names_list), " REPRESENTS A DIFFERENT, INDIVIDUAL TEST SUBJECT")
        for i, line in enumerate(lines):
            record_name = line.strip()
            # print("load_SSL_Dataset_Based_On_Subjects: processing ", record_name)
            data_file = root+record_name+os.sep+record_name
            if task_id == 'RP':
                datasets_list.append(
                    RPDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            tpos=tpos_val, 
                            tneg=tneg_val, 
                            window_size=window_size, 
                            sfreq=sfreq, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'TS':
                datasets_list.append(
                    TSDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            tpos=tpos_val, 
                            tneg=tneg_val, 
                            window_size=window_size, 
                            sfreq=sfreq, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'CPC':
                datasets_list.append(
                    CPCDataset(None, 
                            path=data_file, 
                            Nc=Nc, 
                            Np=Np, 
                            Nb=Nb, # this used to be 2 not 4, but 4 would work better
                            max_Nb_iters=max_Nb_iters, 
                            total_points_factor=total_points_factor, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'PS':
                datasets_list.append(
                    PSDataset(None, 
                            path=data_file, 
                            total_points=total_points_val, 
                            window_size=window_size, 
                            sfreq=sfreq, 
                            windowed_data_name=windowed_data_name, 
                            windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'SQ':
                # print("task_id == SQ")
                datasets_list.append(
                    SQDataset(None, 
                              path=data_file, 
                              total_points=total_points_val, 
                              window_size=window_size, 
                              sfreq=sfreq, 
                              bw=bw, 
                              randomized_augmentation=randomized_augmentation, 
                              num_channels=num_channels, 
                              temporal_len=temporal_len, 
                              windowed_data_name=windowed_data_name, 
                              windowed_start_time_name=windowed_start_time_name
                    )
                )
            elif task_id == 'SA':
                # print("task_id == SA")
                datasets_list.append(
                    SADataset(None, 
                              path=data_file, 
                              total_points=total_points_val, 
                              window_size=window_size, 
                              sfreq=sfreq, 
                              bw=bw, 
                              randomized_augmentation=randomized_augmentation, 
                              num_channels=num_channels, 
                              temporal_len=temporal_len, 
                              NUM_AUGMENTATIONS=NUM_AUGMENTATIONS, 
                              perturb_orig_signal=perturb_orig_signal, 
                              individual_id=i, 
                              num_individuals_in_label=len(lines), 
                              windowed_data_name=windowed_data_name, 
                              windowed_start_time_name=windowed_start_time_name
                    )
                )
            else:
                raise ValueError("load_SSL_Dataset_Based_On_Subjects: task_id == "+str(task_id)+" not recognized")
        f.close()
    
    shuffle_inds = [i for i in range(len(datasets_list))]
    random.Random(random_seed).shuffle(shuffle_inds) # see https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
    datasets_list = datasets_list[shuffle_inds]
    
    # combined_dataset = torch.utils.data.ConcatDataset(datasets_list)

    # data_len = len(combined_dataset)
    # train_len = int(data_len*train_portion)
    # val_len = int(data_len*val_portion)
    # test_len = int(data_len - (train_len+val_len))

    # # see https://pytorch.org/docs/stable/data.html
    # train_set, val_set, test_set = torch.utils.data.random_split(combined_dataset, 
    #                                                              [train_len, val_len, test_len], 
    #                                                              generator=torch.Generator().manual_seed(0)
    # )

    num_train_subjects = max(1, int(train_portion*len(datasets_list)))
    num_val_subjects = max(1, int(val_portion*len(datasets_list)))
    num_test_subjects = max(1, int((1. - (train_portion+val_portion))*len(datasets_list)))

    train_set = torch.utils.data.ConcatDataset(datasets_list[:num_train_subjects])
    train_set = torch.utils.data.random_split(train_set, 
                                              [len(train_set)], 
                                              generator=torch.Generator().manual_seed(random_seed)
    )
    val_set = torch.utils.data.ConcatDataset(datasets_list[num_train_subjects:num_train_subjects+num_val_subjects])
    val_set = torch.utils.data.random_split(val_set, 
                                            [len(val_set)], 
                                            generator=torch.Generator().manual_seed(random_seed)
    )
    test_set = torch.utils.data.ConcatDataset(datasets_list[num_train_subjects+num_val_subjects:])
    test_set = torch.utils.data.random_split(test_set, 
                                             [len(test_set)], 
                                             generator=torch.Generator().manual_seed(random_seed)
    )

    train_val_test_split_subjects = {"train_set_subjects": shuffle_inds[:num_train_subjects], 
                                     "val_set_subjects": shuffle_inds[num_train_subjects:num_train_subjects+num_val_subjects], 
                                     "test_set_subjects": shuffle_inds[num_train_subjects+num_val_subjects:], 
    }

    return train_set, val_set, test_set, train_val_test_split_subjects
# UPSTREAM DATA LOADER BLOCK #################################################################################


# DOWNSTREAM DATA LOADER BLOCK ###############################################################################
class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(self, cached_dataset_path, path=None, windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                 windowed_label_name="_Windowed_Label.npy"):
        if cached_dataset_path is not None:
            self.init_from_cached_data(cached_dataset_path)
        else:
            self.init_params_from_scratch(path, 
                                          windowed_data_name=windowed_data_name, 
                                          windowed_label_name=windowed_label_name
            )
        pass

    def remove_unknowns_from_dataset(self):
        # need to remove the -1 labels (unknown)
        unknown_locs = np.where(self.labels < 0)
        self.data = np.delete(self.data, unknown_locs, axis=self.SAMPLE_AXIS)
        self.labels = np.delete(self.labels, unknown_locs)
        print("DownstreamDataset.remove_unknowns_from_dataset(): removed ", len(unknown_locs[0]), " unknown entries from loaded data")
        pass

    def init_params_from_scratch(self, path, windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                 windowed_label_name="_Windowed_Label.npy"):
        self.SAMPLE_AXIS = 0

        data_path = path + windowed_data_name
        self.data = np.load(data_path)
        # print("init_params_from_scratch: data_path == ", data_path)
        # print("init_params_from_scratch: self.data.shape == ", self.data.shape)
        # print("init_params_from_scratch: self.data == ", self.data)
        data_path = path + windowed_label_name
        self.labels = np.load(data_path)
        # print("init_params_from_scratch: data_path == ", data_path)
        # print("init_params_from_scratch: self.labels.shape == ", self.labels.shape)
        # print("init_params_from_scratch: self.labels == ", self.labels)

        self.remove_unknowns_from_dataset()
        pass

    def init_from_cached_data(self, cached_dataset_path):
        cached_dataset = None
        with open(cached_dataset_path, 'rb') as infile:
            cached_dataset = pkl.load(infile)
        
        self.SAMPLE_AXIS = cached_dataset['SAMPLE_AXIS']
        self.data = cached_dataset['data']
        self.labels = cached_dataset['labels']

        self.remove_unknowns_from_dataset()
        pass

    def save_as_dictionary(self, path):
        with open(path, 'wb') as outfile:
            pkl.dump({
                'data': self.data,
                'labels': self.labels,
                'SAMPLE_AXIS': self.SAMPLE_AXIS,
            }, outfile)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index,:,:]).float()
        y = torch.from_numpy(np.array(self.labels[index])).long()
        return x, y

def load_Downstream_Dataset(cached_datasets_list_dir=None, n_cross_val_splits=5,  
                            windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                            data_folder_name="Mouse_Training_Data", data_root_name="Windowed_Data", 
                            windowed_label_name="_Windowed_Label.npy", 
                            file_names_list="training_names.txt", val_portion=0.2):
    root = os.path.join(data_folder_name, data_root_name, "")
    
    x_vals = []
    y_vals = []
    groups = []

    print("load_Downstream_Dataset: loading data")
    if cached_datasets_list_dir is not None:
        cached_datasets_file_names = os.listdir(cached_datasets_list_dir)
        for i, name in enumerate(cached_datasets_file_names):
            curr_path = os.path.join(cached_datasets_list_dir, name, "")
            data_set = DownstreamDataset(curr_path)
            x_vals.append(data_set.data)
            y_vals.append(data_set.labels)
            groups.append(np.ones(len(data_set.labels))*i)
    else:
        f = open(os.path.join(file_names_list), 'r')
        lines = f.readlines()
        for i, line in enumerate(lines):
            record_name = line.strip()
            # print("load_Downstream_Dataset: processing ", record_name)
            data_file = root+record_name+os.sep+record_name
            data_set = DownstreamDataset(None, 
                        path=data_file, 
                        windowed_data_name=windowed_data_name, 
                        windowed_label_name=windowed_label_name
            )
            x_vals.append(data_set.data)
            y_vals.append(data_set.labels)
            groups.append(np.ones(len(data_set.labels))*i)

        f.close()
        
    x_vals = np.vstack(x_vals)
    y_vals = np.concatenate(y_vals, axis=0)
    groups = np.concatenate(groups, axis=0)
    print("load_Downstream_Dataset: x_vals shape == ", x_vals.shape)
    print("load_Downstream_Dataset: y_vals shape == ", y_vals.shape)
    print("load_Downstream_Dataset: groups shape == ", groups.shape)

    cross_val_kfold = GroupKFold(n_splits=n_cross_val_splits)
    cross_val_kfold.get_n_splits(x_vals, y_vals, groups)

    cv_train_sets = []
    cv_val_sets = []
    cv_test_sets = []
    split_counter = 0
    for train_index, test_index in cross_val_kfold.split(x_vals, y_vals, groups):
        unique_test_mouse = np.unique(groups[test_index])
        print("load_Downstream_Dataset: cross val split number ", split_counter, " leaves out mouse number ", unique_test_mouse)
        
        orig_train_set = TensorDataset(torch.tensor(x_vals[train_index], dtype=torch.float), torch.tensor(y_vals[train_index], dtype=torch.long))
        train_data_len = int(len(orig_train_set)*(1. - val_portion))
        val_data_len = len(orig_train_set) - train_data_len
        
        train_set, val_set = torch.utils.data.random_split(orig_train_set, # see https://pytorch.org/docs/stable/data.html
                                                           [train_data_len, val_data_len], 
                                                           generator=torch.Generator().manual_seed(0)
        )

        cv_train_sets.append(train_set)
        cv_val_sets.append(val_set)
        cv_test_sets.append(TensorDataset(torch.tensor(x_vals[test_index], dtype=torch.float), torch.tensor(y_vals[test_index], dtype=torch.long)))

    return cv_train_sets, cv_val_sets, cv_test_sets