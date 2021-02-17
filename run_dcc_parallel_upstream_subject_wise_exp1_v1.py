# run_dcc_parallel_upstream_subject_wise_exp1_v1.py - see https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments

from train_with_discrete_subjects import train_RP_model, train_TS_model, train_CPC_model, train_PS_model, train_SQ_model, train_SA_model

import multiprocessing
from itertools import product
import os

def kick_off_training_run(alg_type, alg_args, curr_run_meta_params):
    """
    alg_type: str in ['RP', 'TS', 'CPC', 'PS', 'SQ', 'SA']
    alg_args: value from full_parameter_dict in __main__
    curr_run_meta_params: [train_set_proportion, root_save_dir, data_folder_name]
    """
    
    ROOT_SAVE_DIR_INDEX = 0
    DATA_FOLDER_INDEX = 40
    TRAIN_PORTION_INDEX = 43
    VAL_PORTION_INDEX = 44
    
    train_set_proportion = curr_run_meta_params[0]
    root_save_dir = curr_run_meta_params[1]
    data_folder_name = curr_run_meta_params[2]
    
    print("START RUNNING ", alg_type, " WITH TRAIN PORTION == ", train_set_proportion)

    # initialize train / val / test proportions
    assert train_set_proportion < 1.
    test_set_proportion = 0.1
    val_set_proportion = 1. - (train_set_proportion + test_set_proportion)

    # create the new directory for saving info to
    # see https://www.geeksforgeeks.org/python-os-path-exists-method/ and https://www.geeksforgeeks.org/python-os-mkdir-method/
    train_set_proportion_str_rep = str(train_set_proportion)[2:]
    if len(train_set_proportion_str_rep) == 1:
        train_set_proportion_str_rep = train_set_proportion_str_rep + "0"
    elif len(train_set_proportion_str_rep) >= 2: 
        train_set_proportion_str_rep = train_set_proportion_str_rep[:2]
    assert len(train_set_proportion_str_rep) == 2

    curr_run_root_save_dir = root_save_dir + alg_type + "_upstream_train_prop_" + train_set_proportion_str_rep + "_percent" + os.sep

    if not os.path.exists(curr_run_root_save_dir):
        os.mkdir(curr_run_root_save_dir)
    
    # tweak arguments to represent current run
    alg_args[ROOT_SAVE_DIR_INDEX] = curr_run_root_save_dir
    alg_args[DATA_FOLDER_INDEX] = data_folder_name
    alg_args[TRAIN_PORTION_INDEX] = train_set_proportion
    alg_args[VAL_PORTION_INDEX] = val_set_proportion

    # train the upstream model
    if alg_type == 'RP':
        train_RP_model(*alg_args)
    elif alg_type == 'TS':
        train_TS_model(*alg_args)
    elif alg_type == 'CPC':
        train_CPC_model(*alg_args)
    elif alg_type == 'PS':
        train_PS_model(*alg_args)
    elif alg_type == 'SQ':
        train_SQ_model(*alg_args)
    elif alg_type == 'SA':
        train_SA_model(*alg_args)
    else:
        raise ValueError("alg_type == "+str(alg_type)+" not recognized")

    print("DONE RUNNING ", alg_type, " WITH TRAIN PORTION == ", train_set_proportion)
    pass


if __name__=="__main__":
    print("<<< MAIN: START")
    root_save_dir = None
    data_folder_name = None

    full_parameter_dict = {
        'RP': [None, "final_RP_model.bin", 256, True, 100, 5e-4, (0.9, 0.999), 0.001, 6, 1, None, None, None, 11, 3000, 0.5, 100, None, None, None, None, None, None, None, None, None, None, None, 2000, 30, 120, 3, 1000, None, None, None, None, None, "_Windowed_Pretext_Preprocess.npy", "_Windowed_StartTime.npy", None, "Windowed_Data", "training_names.txt", None, None, 0.1, 0], 
        'TS': [None, "final_TS_model.bin", 256, True, 100, 5e-4, (0.9, 0.999), 0.001, 6, 1, None, None, None, 11, 3000, 0.5, 100, None, None, None, None, None, None, None, None, None, None, None, 2000, 30, 120, 3, 1000, None, None, None, None, None, "_Windowed_Pretext_Preprocess.npy", "_Windowed_StartTime.npy", None, "Windowed_Data", "training_names.txt", None, None, 0.1, 0], 
        'CPC': [None, "final_CPC_model.bin", 16, True, 100, 5e-4, (0.9, 0.999), 0.001, 6, 1, None, 100, 100, 11, 3000, 0.5, 100, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 10, 16, 4, 1000, 0.05, "_Windowed_Preprocess.npy", "_Windowed_StartTime.npy", None, "Windowed_Data", "training_names.txt", None, None, 0.1, 0], 
        'PS': [None, "final_PS_model.bin", 256, True, 100, 5e-4, (0.9, 0.999), 0.001, 6, 1, None, None, None, 11, 3000, 0.5, 100, None, None, None, None, None, None, None, None, None, None, None, 2000, 30, 120, 3, 1000, None, None, None, None, None, "_Windowed_Pretext_Preprocess.npy", "_Windowed_StartTime.npy", None, "Windowed_Data", "training_names.txt", None, None, 0.1, 0], 
        'SQ': [None, "final_SQ_model.bin", 32, True, 100, 5e-4, (0.9, 0.999), 0.001, 6, 1, None, None, None, 11, 3000, 0.5, 100, "simplified", 5, False, 32, 0.05, 2, True, None, 1., 0.999, None, 2000, 30, 120, 3, 1000, None, None, None, None, None, "_Windowed_Pretext_Preprocess.npy", "_Windowed_StartTime.npy", None, "Windowed_Data", "training_names.txt", None, None, 0.1, 0], 
        'SA': [None, "final_SA_model.bin", 32, True, 100, 5e-4, (0.9, 0.999), 0.001, 6, 1, None, None, None, 11, 3000, 0.5, 100, None, 5, False, 32, 0.05, 2, True, None, 1., 0.999, None, 2000, 30, 120, 3, 1000, None, None, None, None, None, "_Windowed_Pretext_Preprocess.npy", "_Windowed_StartTime.npy", None, "Windowed_Data", "training_names.txt", None, None, 0.1, 0], 
    }

    upstream_model_types = list(full_parameter_dict.keys()) # ['RP', 'TS', 'CPC', 'PS', 'SQ', 'SA']
    train_set_proportions = [0.1 + (i*0.2) for i in range(4)]
    parameters_to_be_parallelized = list(product(upstream_model_types, train_set_proportions))

    all_parameter_settings = [(x[0], full_parameter_dict[x[0]], [x[1], root_save_dir, data_folder_name]) for x in parameters_to_be_parallelized]

    pool = multiprocessing.Pool()
    pool.starmap(kick_off_training_run, all_parameter_settings)

    print("<<< MAIN: DONE!!!")
    pass