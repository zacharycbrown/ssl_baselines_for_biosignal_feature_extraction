# data_utils.py
"""
Zachary Brown

adapted from Jason Stranne's github.com/jstranne/mouse_self_supervision
"""
import os
import numpy as np
import sys
# from Extract_Data import loadSignals, extractWholeRecord, import_labels
from sklearn import preprocessing
from scipy import stats
import scipy.io
import gc
import mne


##### EXTRACT DATA SET CODE BLOCK #####
def list_training_files(out_file_name='training_names.txt', data_folder_name='Mouse_Training_Data', 
                        data_file_root_name='LFP_Data', data_file_extension='_LFP.mat', 
                        SKIP_MICE=True):
    print("list_training_files: creating list of training files")
    # directory = os.listdir('.'+os.sep+data_folder_name+os.sep+data_file_root_name)
    directory = os.listdir(data_folder_name+os.sep+data_file_root_name)
    f = open(out_file_name, 'w')
    mouse_list = []
    for folder in directory:
        print(folder)
        if folder.startswith('MouseCK'):
            mouse_list.append(folder[:folder.index(data_file_extension)])
    mouse_list.sort()

    last = '_'
    for file_name in mouse_list:
        if SKIP_MICE:
            if last[:last.index("_")] != file_name[:file_name.index("_")]:
                f.write(file_name+'\n')
            last = file_name
        else:
            f.write(file_name+'\n')
    f.close()
    print('list_training_files: Done')
    pass

def import_signals(file_name):
    return scipy.io.loadmat(file_name)['val']

def load_signals(record_name, data_path):
    signals = scipy.io.loadmat(data_path+record_name+'_LFP.mat')
    gc.collect()
    return signals

def load_time(record_name, data_path):
    print("Time path", data_path+record_name+'_TIME.mat')
    time = scipy.io.loadmat(data_path+record_name+'_TIME.mat')['INT_TIME'][0]
    gc.collect()
    return time

def convert_signal_to_array(x, time, downstream=True, fs=1000, first_fs_upper_bound=300):
    print("convert_signal_to_array: time == ", time)
    data_map = {}
    desired_keys=["PrL_Cx",     # I believe these are the channels assumed to be in the signal
                  "Md_Thal", 
                  "IL_Cx", 
                  "BLA", 
                  "Acb_Sh", 
                  "Acb_Core", 
                  "mSNC", 
                  "mDHip", 
                  "lSNC", 
                  "lDHip", 
                  "L_VTA", 
                  "R_VTA"
    ]
    for k in desired_keys:
        data_map[k] = []
    
    for key in x:
        if "_" in key and key[-1] != "_":
            new_key = key[:key.rindex("_")]
            if new_key in data_map:
                data_map[new_key].append(x[key])
    
    # add right and left VTA
    data_map["VTA"] = data_map["L_VTA"] + data_map["R_VTA"]
    del data_map["L_VTA"]
    del data_map["R_VTA"]
    print("convert_signal_to_array: VTA len == ", len(data_map["VTA"]))

    # average the lists
    for key in data_map:
        print("convert_signal_to_array: data_map[", key, "] shape == ", np.array(data_map[key]).shape)
        data_map[key] = np.mean(np.array(data_map[key]), axis=0).ravel()

    # find starts and stops
    print("convert_signal_to_array: time == ", time)
    starts_and_stops = np.r_[0:first_fs_upper_bound*fs, time[0]*fs:(time[0]+time[1])*fs, time[2]*fs:(time[2]+time[3])*fs]

    # compile into a single array
    ans = None
    if downstream:
        ans = np.array([data_map[k][starts_and_stops] for k in data_map])
    else:
        ans = np.array([data_map[k][:] for k in data_map])
    
    print("convert_signal_to_array: ans shape == ", ans.shape)
    gc.collect()
    return ans

def extract_whole_record(record_name, data_path, downstream=True, fs=1000, first_fs_upper_bound=300, 
                         l_freq=None, h_freq=55, method='fir', fir_window='hamming'):
    signals = load_signals(record_name, data_path+os.sep+"LFP_Data"+os.sep)
    time = load_time(record_name, data_path+os.sep+"INT_TIME"+os.sep)

    signals = convert_signal_to_array(signals, time, downstream=downstream, fs=fs, first_fs_upper_bound=first_fs_upper_bound)

    # 4th order butterworth
    signals = mne.filter.filter_data(data=signals, sfreq=fs, l_freq=l_freq, h_freq=h_freq, method=method, fir_window=fir_window)

    # fs=1000 -> 250Hz downsample
    # removing the downsample
    signals = signals[:,0::1]
    gc.collect()
    return np.transpose(signals)

def import_labels(record_name, data_path, fs=1000, label_interval=1200):
    # imports all the sleep stages as numbers in an array. -1 corresponds to an undefined label
    time = load_time(record_name, data_path+os.sep+"INT_TIME"+os.sep)
    print("import_labels: time == ", time)
    labels = np.zeros(label_interval*fs)
    labels[time[1]*fs:(2*time[1])*fs] = 1
    labels[2*time[1]*fs:(2*time[1] + time[3])*fs] = 2
    gc.collect()
    return labels

##### CREATE WINDOWED DATA SET CODE BLOCK #####
def preprocess_file(record_name, root_name="Mouse_Training_Data", sampling_rate=1000, window_size=3, channel_axis=0):
    # root = os.path.join(root_name, "")
    root = root_name
    # returns all channels of the eeg, 55Hz low pass filtered, with sampling_rate in Hz and window_size in seconds
    x = extract_whole_record(record_name=record_name, data_path=root)
    y = import_labels(record_name=record_name, data_path=root)
    total_windows = len(x) // (sampling_rate*window_size)

    print("preprocess_file: Before processing, shapes are: ")
    print("\t x shape == ", x.shape)
    print("\t y shape == ", y.shape)
    print("\t Total Windows == ", total_windows)

    x_windows = []
    sleep_labels = []

    for i in range(total_windows):
        x_val = x[sampling_rate*window_size*i : sampling_rate*window_size*(i+1)]
        y_val = y[sampling_rate*window_size*i : sampling_rate*window_size*(i+1)]

        mode, mode_count = stats.mode(y_val)

        #normalized channel size for zero mean and unit sd
        x_val = preprocessing.scale(x_val, axis=channel_axis)
        x_windows.append(x_val)
        sleep_labels.append(mode[0])
    
    x_rp = extract_whole_record(record_name=record_name, data_path=root,  downstream=False)
    total_windows  = len(x_rp)//(sampling_rate*window_size)
    print("preprocess_file: Before processing, shapes are: ")
    print("\t x_rp shape == ", x_rp.shape)
    print("\t Total Windows == ", total_windows)

    x_windows_rp = []
    start_times = []
    for i in range(total_windows):
        x_val = x_rp[sampling_rate*window_size*i : sampling_rate*window_size*(i+1)]
        x_val = preprocessing.scale(x_val, axis=channel_axis)
        x_windows_rp.append(x_val)
        start_times.append(window_size*i)
    
    print("preprocess_file: X_WINDOWS IS SIZE ", np.array(x_windows).shape)
    print("preprocess_file: X_WINDOWS_RP IS SIZE ", np.array(x_windows_rp).shape)

    root = os.path.join(root_name, "Windowed_Data", record_name, "")
    os.makedirs(root, exist_ok=True)
    np.save(file=root+os.sep+record_name+"_Windowed_Preprocess", arr=np.array(x_windows))
    np.save(file=root+os.sep+record_name+"_Windowed_Label", arr=np.array(sleep_labels))
    np.save(file=root+os.sep+record_name+"_Windowed_StartTime", arr=np.array(start_times))
    np.save(file=root+os.sep+record_name+"_Windowed_Pretext_Preprocess", arr=np.array(x_windows_rp))

def create_windowed_dataset(key_file_names_file="training_names.txt", data_root_directory=None):
    f = open(os.path.join(key_file_names_file), 'r')
    lines = f.readlines()
    for line in lines:
        print("create_windowed_dataset: stripped line == ", line.strip())
        if data_root_directory is not None:
            preprocess_file(line.strip(), root_name=data_root_directory)
        else:
            preprocess_file(line.strip())
    f.close()
    pass



###############################################

if __name__ == "__main__":
    root = os.path.join("Mouse_Training_Data", "")
    print("root == ", root)

    rec_name="MouseCKA1_030515_HCOFTS"

    x = extract_whole_record(record_name=rec_name, data_path=root)
    print("len x[0] == ", len(x[0]))

    y = import_labels(record_name=rec_name, data_path=root)
    print("len y == ", len(y))

    print("num0 == ", sum(y==0))
    print("num1 == ", sum(y==1))
    print("num2 == ", sum(y==2))