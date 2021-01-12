# train.py

import numpy as np
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import random
from models import *
from data_loaders import *
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pickle as pkl
from matplotlib import pyplot as plt

import itertools


def get_number_of_correct_preds(y_pred, y_true):
    return ((y_pred*y_true) > 0).float().sum().item()

def get_number_of_correct_cpc_preds(y_pred):
    arg_max = torch.argmax(y_pred, dim=2)
    correct = (torch.sum(arg_max==0)).float().sum().item()
    return correct, arg_max.nelement()

def plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, plot_series_name, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(avg_train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title(plot_series_name+": Average Training Losses")
    plt.legend()
    plt.draw()
    loss_plot_save_path = os.path.join(save_path, plot_series_name+"_loss_visualization.png")
    fig1.savefig(loss_plot_save_path)

    fig2, ax2 = plt.subplots()
    ax2.plot(avg_train_accs, label="training")
    ax2.plot(avg_val_accs, label="validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title(plot_series_name+": Average Prediction Accuracy")
    plt.legend()
    plt.draw()
    accuracy_plot_save_path = os.path.join(save_path, plot_series_name+"_accuracy_visualization.png")
    fig2.savefig(accuracy_plot_save_path)
    pass


def train_RP_model(save_dir_for_model, model_file_name="final_RP_model.bin", batch_size=256, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=4, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, ct_dim=None, h_dim=None, 
                    channels=11, dropout_rate=0.5, embed_dim=100, # hyper parameters for RP Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=None, 
                    windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1):
    
    # First, load the training, validation, and test sets
    # train_set, val_set, test_set = load_RPDataset(cached_datasets_list_dir=cached_datasets_list_dir, 
    #                                                 total_points_val=total_points_val, 
    #                                                 tpos_val=tpos_val, 
    #                                                 tneg_val=tneg_val, 
    #                                                 window_size=window_size, 
    #                                                 sfreq=sfreq, 
    #                                                 data_folder_name=data_folder_name, 
    #                                                 data_root_name=data_root_name, 
    #                                                 file_names_list=file_names_list, 
    #                                                 train_portion=train_portion, 
    #                                                 val_portion=val_portion, 
    #                                                 test_portion=test_portion
    # )
    train_set, val_set, test_set = load_SSL_Dataset('RP',
                                                    cached_datasets_list_dir=cached_datasets_list_dir, 
                                                    total_points_val=total_points_val, 
                                                    tpos_val=tpos_val, 
                                                    tneg_val=tneg_val, 
                                                    window_size=window_size, 
                                                    sfreq=sfreq, 
                                                    Nc=Nc, 
                                                    Np=Np, 
                                                    Nb=Nb, # this used to be 2 not 4, but 4 would work better
                                                    max_Nb_iters=max_Nb_iters, 
                                                    total_points_factor=total_points_factor, 
                                                    windowed_data_name=windowed_data_name,
                                                    windowed_start_time_name=windowed_start_time_name,
                                                    data_folder_name=data_folder_name, 
                                                    data_root_name=data_root_name, 
                                                    file_names_list=file_names_list, 
                                                    train_portion=train_portion, 
                                                    val_portion=val_portion, 
                                                    test_portion=test_portion
    )

    # initialize data loaders for training
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle#, num_workers=num_workers # see https://www.programmersought.com/article/93393550792/
    )
    val_loader = torch.utils.data.DataLoader(val_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle#, num_workers=num_workers
    )

    print("train_RP_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0

    # initialize model
    model = RPNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).to(device)
    if former_state_dict_file is not None:
        model.load_state_dict(torch.load(former_state_dict_file))

    print("train_RP_model: START OF TRAINING")
    # initialize training state
    min_val_inaccuracy = float("inf")
    min_state = None
    num_evaluations_since_model_saved = 0
    saved_model = None
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    # learning_rate = learning_rate
    # beta_vals = beta_vals
    optimizer = torch.optim.Adam(model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    # Iterate over epochs
    avg_train_losses = []
    avg_train_accs = []
    avg_val_accs = []
    for epoch in range(max_epochs):
        # print("train_RP_model: now starting epoch ", epoch, " of ", max_epochs)
        model.train()
        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        
        # iterate over training batches
        for x1, x2, y in train_loader:
            # transfer to GPU
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            # zero out any pre-existing gradients
            optimizer.zero_grad()

            # make prediction and compute resulting loss
            y_pred = model(x1, x2)
            loss = loss_fn(y_pred, y)

            # compute accuracy
            num_correct_train_preds += get_number_of_correct_preds(y_pred, y)
            total_num_train_preds += len(y)

            # update weights
            loss.backward()
            optimizer.step()

            # track loss
            running_train_loss += loss.item()

            # free up cuda memory
            del x1
            del x2
            del y
            del y_pred
            torch.cuda.empty_cache()
        
        # iterate over validation batches
        num_correct_val_preds = 0
        total_num_val_preds = 0
        with torch.no_grad():
            model.eval()

            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                # evaluate model
                y_pred = model(x1, x2)
                num_correct_val_preds += get_number_of_correct_preds(y_pred, y)
                total_num_val_preds += len(y)

                # free up cuda memory
                del x1
                del x2
                del y
                del y_pred
                torch.cuda.empty_cache()
        
        # record averages
        avg_train_accs.append(num_correct_train_preds / total_num_train_preds)
        avg_val_accs.append(num_correct_val_preds / total_num_val_preds)
        avg_train_losses.append(running_train_loss / len(train_loader))
        
        # check stopping criterion / save model
        incorrect_val_percentage = 1. - (num_correct_val_preds / total_num_val_preds)
        if incorrect_val_percentage < min_val_inaccuracy:
            num_evaluations_since_model_saved = 0
            min_val_inaccuracy = incorrect_val_percentage
            saved_model = model.state_dict()
        else:
            num_evaluations_since_model_saved += 1
            if num_evaluations_since_model_saved >= max_evals_after_saving:
                print("train_RP_model: EARLY STOPPING on epoch ", epoch)
                break
        
        # save intermediate state_dicts just in case
        if epoch % save_freq == 0:
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_RP_model_epoch"+str(epoch)+".bin")
            torch.save(model.state_dict(), temp_model_save_path)
            embedder_save_path = os.path.join(save_dir_for_model, "temp_embedder_epoch"+str(epoch)+".bin")
            torch.save(model.embed_model.state_dict(), embedder_save_path)
            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "epoch"+str(epoch), save_dir_for_model)

    print("train_RP_model: END OF TRAINING - now saving final model / other info")

    # save final model
    model.load_state_dict(saved_model)
    model_save_path = os.path.join(save_dir_for_model, model_file_name)
    torch.save(model.state_dict(), model_save_path)
    embedder_save_path = os.path.join(save_dir_for_model, "embedder_"+model_file_name)
    torch.save(model.embed_model.state_dict(), embedder_save_path)

    meta_data_save_path = os.path.join(save_dir_for_model, "meta_data_and_hyper_parameters.pkl")
    with open(meta_data_save_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": avg_train_losses, 
            "avg_train_accs": avg_train_accs, 
            "avg_val_accs": avg_val_accs, 
            "save_dir_for_model": save_dir_for_model, 
            "model_file_name": model_file_name, 
            "batch_size": batch_size, 
            "shuffle": shuffle, #"num_workers": num_workers, 
            "max_epochs": max_epochs, 
            "learning_rate": learning_rate, 
            "beta_vals": beta_vals, 
            "weight_decay": weight_decay, 
            "max_evals_after_saving": max_evals_after_saving, 
            "save_freq": save_freq, 
            "former_state_dict_file": former_state_dict_file, 
            "ct_dim": ct_dim, 
            "h_dim": h_dim, 
            "channels": channels, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "cached_datasets_list_dir": cached_datasets_list_dir, 
            "total_points_val": total_points_val, 
            "tpos_val": tpos_val, 
            "tneg_val": tneg_val, 
            "window_size": window_size,
            "sfreq": sfreq, 
            "Nc": Nc, 
            "Np": Np, 
            "Nb": Nb,
            "max_Nb_iters": max_Nb_iters, 
            "total_points_factor": total_points_factor, 
            "windowed_data_name": windowed_data_name,
            "windowed_start_time_name": windowed_start_time_name,
            "data_folder_name": data_folder_name, 
            "data_root_name": data_root_name, 
            "file_names_list": file_names_list, 
            "train_portion": train_portion, 
            "val_portion": val_portion, 
            "test_portion": test_portion, 
        }, outfile)

    plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "Final", save_dir_for_model)
    
    print("train_RP_model: DONE!")
    pass

def train_TS_model(save_dir_for_model, model_file_name="final_TS_model.bin", batch_size=256, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=4, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, ct_dim=None, h_dim=None, 
                    channels=11, dropout_rate=0.5, embed_dim=100, # hyper parameters for RP Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=None, 
                    windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1):
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset('TS',
                                                    cached_datasets_list_dir=cached_datasets_list_dir, 
                                                    total_points_val=total_points_val, 
                                                    tpos_val=tpos_val, 
                                                    tneg_val=tneg_val, 
                                                    window_size=window_size, 
                                                    sfreq=sfreq, 
                                                    Nc=Nc, 
                                                    Np=Np, 
                                                    Nb=Nb, # this used to be 2 not 4, but 4 would work better
                                                    max_Nb_iters=max_Nb_iters, 
                                                    total_points_factor=total_points_factor, 
                                                    windowed_data_name=windowed_data_name,
                                                    windowed_start_time_name=windowed_start_time_name,
                                                    data_folder_name=data_folder_name, 
                                                    data_root_name=data_root_name, 
                                                    file_names_list=file_names_list, 
                                                    train_portion=train_portion, 
                                                    val_portion=val_portion, 
                                                    test_portion=test_portion
    )

    # initialize data loaders for training
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle#, num_workers=num_workers # see https://www.programmersought.com/article/93393550792/
    )
    val_loader = torch.utils.data.DataLoader(val_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle#, num_workers=num_workers
    )

    print("train_TS_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0

    # initialize model
    model = TSNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).to(device)
    if former_state_dict_file is not None:
        model.load_state_dict(torch.load(former_state_dict_file))

    print("train_TS_model: START OF TRAINING")
    # initialize training state
    min_val_inaccuracy = float("inf")
    min_state = None
    num_evaluations_since_model_saved = 0
    saved_model = None
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    # learning_rate = learning_rate
    # beta_vals = beta_vals
    optimizer = torch.optim.Adam(model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    # Iterate over epochs
    avg_train_losses = []
    avg_train_accs = []
    avg_val_accs = []
    for epoch in range(max_epochs):
        # print("train_TS_model: now starting epoch ", epoch, " of ", max_epochs)
        model.train()
        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        
        # iterate over training batches
        for x1, x2, x3, y in train_loader:
            # transfer to GPU
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

            # zero out any pre-existing gradients
            optimizer.zero_grad()

            # make prediction and compute resulting loss
            y_pred = model(x1, x2, x3)
            loss = loss_fn(y_pred, y)

            # compute accuracy
            num_correct_train_preds += get_number_of_correct_preds(y_pred, y)
            total_num_train_preds += len(y)

            # update weights
            loss.backward()
            optimizer.step()

            # track loss
            running_train_loss += loss.item()

            # free up cuda memory
            del x1
            del x2
            del x3
            del y
            del y_pred
            torch.cuda.empty_cache()
        
        # iterate over validation batches
        num_correct_val_preds = 0
        total_num_val_preds = 0
        with torch.no_grad():
            model.eval()

            for x1, x2, x3, y in val_loader:
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

                # evaluate model
                y_pred = model(x1, x2, x3)
                num_correct_val_preds += get_number_of_correct_preds(y_pred, y)
                total_num_val_preds += len(y)

                # free up cuda memory
                del x1
                del x2
                del x3
                del y
                del y_pred
                torch.cuda.empty_cache()
        
        # record averages
        avg_train_accs.append(num_correct_train_preds / total_num_train_preds)
        avg_val_accs.append(num_correct_val_preds / total_num_val_preds)
        avg_train_losses.append(running_train_loss / len(train_loader))
        
        # check stopping criterion / save model
        incorrect_val_percentage = 1. - (num_correct_val_preds / total_num_val_preds)
        if incorrect_val_percentage < min_val_inaccuracy:
            num_evaluations_since_model_saved = 0
            min_val_inaccuracy = incorrect_val_percentage
            saved_model = model.state_dict()
        else:
            num_evaluations_since_model_saved += 1
            if num_evaluations_since_model_saved >= max_evals_after_saving:
                print("train_TS_model: EARLY STOPPING on epoch ", epoch)
                break
        
        # save intermediate state_dicts just in case
        if epoch % save_freq == 0:
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_TS_model_epoch"+str(epoch)+".bin")
            torch.save(model.state_dict(), temp_model_save_path)
            embedder_save_path = os.path.join(save_dir_for_model, "temp_embedder_epoch"+str(epoch)+".bin")
            torch.save(model.embed_model.state_dict(), embedder_save_path)
            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "epoch"+str(epoch), save_dir_for_model)

    print("train_TS_model: END OF TRAINING - now saving final model / other info")

    # save final model
    model.load_state_dict(saved_model)
    model_save_path = os.path.join(save_dir_for_model, model_file_name)
    torch.save(model.state_dict(), model_save_path)
    embedder_save_path = os.path.join(save_dir_for_model, "embedder_"+model_file_name)
    torch.save(model.embed_model.state_dict(), embedder_save_path)

    meta_data_save_path = os.path.join(save_dir_for_model, "meta_data_and_hyper_parameters.pkl")
    with open(meta_data_save_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": avg_train_losses, 
            "avg_train_accs": avg_train_accs, 
            "avg_val_accs": avg_val_accs, 
            "save_dir_for_model": save_dir_for_model, 
            "model_file_name": model_file_name, 
            "batch_size": batch_size, 
            "shuffle": shuffle, #"num_workers": num_workers, 
            "max_epochs": max_epochs, 
            "learning_rate": learning_rate, 
            "beta_vals": beta_vals, 
            "weight_decay": weight_decay, 
            "max_evals_after_saving": max_evals_after_saving, 
            "save_freq": save_freq, 
            "former_state_dict_file": former_state_dict_file, 
            "ct_dim": ct_dim, 
            "h_dim": h_dim, 
            "channels": channels, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "cached_datasets_list_dir": cached_datasets_list_dir, 
            "total_points_val": total_points_val, 
            "tpos_val": tpos_val, 
            "tneg_val": tneg_val, 
            "window_size": window_size,
            "sfreq": sfreq, 
            "Nc": Nc, 
            "Np": Np, 
            "Nb": Nb,
            "max_Nb_iters": max_Nb_iters, 
            "total_points_factor": total_points_factor, 
            "windowed_data_name": windowed_data_name,
            "windowed_start_time_name": windowed_start_time_name,
            "data_folder_name": data_folder_name, 
            "data_root_name": data_root_name, 
            "file_names_list": file_names_list, 
            "train_portion": train_portion, 
            "val_portion": val_portion, 
            "test_portion": test_portion, 
        }, outfile)

    plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "Final", save_dir_for_model)
    
    print("train_TS_model: DONE!")
    pass

def train_CPC_model(save_dir_for_model, model_file_name="final_CPC_model.bin", batch_size=16, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=12, 
                    max_evals_after_saving=6, save_freq=10, 
                    former_state_dict_file=None, ct_dim=100, h_dim=100, channels=11, dropout_rate=0.5, embed_dim=100, # hyper parameters for RP Model
                    cached_datasets_list_dir=None, total_points_val=None, tpos_val=None, tneg_val=None, window_size=None, #hyper parameters for data loaders
                    sfreq=None, Nc=10, Np=16, Nb=4, max_Nb_iters=1000, total_points_factor=0.05, 
                    windowed_data_name="_Windowed_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1):
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset('CPC',
                                                    cached_datasets_list_dir=cached_datasets_list_dir, 
                                                    total_points_val=total_points_val, 
                                                    tpos_val=tpos_val, 
                                                    tneg_val=tneg_val, 
                                                    window_size=window_size, 
                                                    sfreq=sfreq, 
                                                    Nc=Nc, 
                                                    Np=Np, 
                                                    Nb=Nb, # this used to be 2 not 4, but 4 would work better
                                                    max_Nb_iters=max_Nb_iters, 
                                                    total_points_factor=total_points_factor, 
                                                    windowed_data_name=windowed_data_name,
                                                    windowed_start_time_name=windowed_start_time_name,
                                                    data_folder_name=data_folder_name, 
                                                    data_root_name=data_root_name, 
                                                    file_names_list=file_names_list, 
                                                    train_portion=train_portion, 
                                                    val_portion=val_portion, 
                                                    test_portion=test_portion
    )

    # initialize data loaders for training
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size, 
                                                shuffle=shuffle#, num_workers=num_workers # see https://www.programmersought.com/article/93393550792/
    )
    val_loader = torch.utils.data.DataLoader(val_set, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle#, num_workers=num_workers
    )

    print("train_CPC_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0

    # initialize model
    model = CPCNet(Np=Np, 
                channels=channels, 
                ct_dim=ct_dim, 
                h_dim=h_dim, 
                dropout_rate=dropout_rate, 
                embed_dim=embed_dim
    ).to(device)
    if former_state_dict_file is not None:
        model.load_state_dict(torch.load(former_state_dict_file))

    print("train_CPC_model: START OF TRAINING")
    # initialize training state
    min_val_inaccuracy = float("inf")
    min_state = None
    num_evaluations_since_model_saved = 0
    saved_model = None
    
    # learning_rate = learning_rate
    # beta_vals = beta_vals
    optimizer = torch.optim.Adam(model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    # Iterate over epochs
    avg_train_losses = []
    avg_train_accs = []
    avg_val_accs = []
    for epoch in range(max_epochs):
        # print("train_CPC_model: now starting epoch ", epoch, " of ", max_epochs)
        model.train()
        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        
        # iterate over training batches
        for xc, xp, xb in train_loader:
            # transfer to GPU
            xc, xp, xb = xc.to(device), xp.to(device), xb.to(device)

            # zero out any pre-existing gradients
            optimizer.zero_grad()

            # make prediction and compute resulting loss
            y_pred = model(xc, xp, xb)
            loss = model.custom_cpc_loss(y_pred)

            # compute accuracy
            new_num_correct, new_num_total = get_number_of_correct_cpc_preds(y_pred)
            num_correct_train_preds += new_num_correct
            total_num_train_preds += new_num_total

            # update weights
            loss.backward()
            optimizer.step()

            # track loss
            running_train_loss += loss.item()

            # free up cuda memory
            del xc
            del xp
            del xb
            del y_pred
            torch.cuda.empty_cache()
        
        # iterate over validation batches
        num_correct_val_preds = 0
        total_num_val_preds = 0
        with torch.no_grad():
            model.eval()

            for xc, xp, xb in val_loader:
                xc, xp, xb = xc.to(device), xp.to(device), xb.to(device)

                # evaluate model
                y_pred = model(xc, xp, xb)
                new_num_correct, new_num_total = get_number_of_correct_cpc_preds(y_pred)
                num_correct_val_preds += new_num_correct
                total_num_val_preds += new_num_total

                # free up cuda memory
                del xc
                del xp
                del xb
                del y_pred
                torch.cuda.empty_cache()
        
        # record averages
        avg_train_accs.append(num_correct_train_preds / total_num_train_preds)
        avg_val_accs.append(num_correct_val_preds / total_num_val_preds)
        avg_train_losses.append(running_train_loss / len(train_loader))
        
        # check stopping criterion / save model
        incorrect_val_percentage = 1. - (num_correct_val_preds / total_num_val_preds)
        if incorrect_val_percentage < min_val_inaccuracy:
            num_evaluations_since_model_saved = 0
            min_val_inaccuracy = incorrect_val_percentage
            saved_model = model.state_dict()
        else:
            num_evaluations_since_model_saved += 1
            if num_evaluations_since_model_saved >= max_evals_after_saving:
                print("train_CPC_model: EARLY STOPPING on epoch ", epoch)
                break
        
        # save intermediate state_dicts just in case
        if epoch % save_freq == 0:
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_CPC_model_epoch"+str(epoch)+".bin")
            torch.save(model.state_dict(), temp_model_save_path)
            embedder_save_path = os.path.join(save_dir_for_model, "temp_embedder_epoch"+str(epoch)+".bin")
            torch.save(model.embed_model.state_dict(), embedder_save_path)
            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "epoch"+str(epoch), save_dir_for_model)

    print("train_CPC_model: END OF TRAINING - now saving final model / other info")

    # save final model
    model.load_state_dict(saved_model)
    model_save_path = os.path.join(save_dir_for_model, model_file_name)
    torch.save(model.state_dict(), model_save_path)
    embedder_save_path = os.path.join(save_dir_for_model, "embedder_"+model_file_name)
    torch.save(model.embed_model.state_dict(), embedder_save_path)

    meta_data_save_path = os.path.join(save_dir_for_model, "meta_data_and_hyper_parameters.pkl")
    with open(meta_data_save_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": avg_train_losses, 
            "avg_train_accs": avg_train_accs, 
            "avg_val_accs": avg_val_accs, 
            "save_dir_for_model": save_dir_for_model, 
            "model_file_name": model_file_name, 
            "batch_size": batch_size, 
            "shuffle": shuffle, #"num_workers": num_workers, 
            "max_epochs": max_epochs, 
            "learning_rate": learning_rate, 
            "beta_vals": beta_vals, 
            "weight_decay": weight_decay, 
            "max_evals_after_saving": max_evals_after_saving, 
            "save_freq": save_freq, 
            "former_state_dict_file": former_state_dict_file, 
            "ct_dim": ct_dim, 
            "h_dim": h_dim, 
            "channels": channels, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "cached_datasets_list_dir": cached_datasets_list_dir, 
            "total_points_val": total_points_val, 
            "tpos_val": tpos_val, 
            "tneg_val": tneg_val, 
            "window_size": window_size,
            "sfreq": sfreq, 
            "Nc": Nc, 
            "Np": Np, 
            "Nb": Nb,
            "max_Nb_iters": max_Nb_iters, 
            "total_points_factor": total_points_factor, 
            "windowed_data_name": windowed_data_name,
            "windowed_start_time_name": windowed_start_time_name,
            "data_folder_name": data_folder_name, 
            "data_root_name": data_root_name, 
            "file_names_list": file_names_list, 
            "train_portion": train_portion, 
            "val_portion": val_portion, 
            "test_portion": test_portion, 
        }, outfile)

    plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "Final", save_dir_for_model)
    
    print("train_CPC_model: DONE!")
    pass


def get_smallest_class_sample_num_for_downstream_training(train_set, num_classes):
    """
    returns the number of samples of the least-sampled class. For example, if class 1 had 3000 samples and class 2 had 500, 
    this method would return 500
    """
    indices = [[] for x in range(num_classes)]
    for i in range(len(train_set)):
        # puts each index in the list based on the label that it has
        indices[train_set[i][1].item()].append(i)

    smallest_num_class_samples = len(indices[0])
    for num in range(len(indices)):
        smallest_num_class_samples = min(smallest_num_class_samples, len(indices[num]))

    return smallest_num_class_samples

def restrict_training_size_per_class_for_downstream_training(train_set, num_samples_per_class, num_classes):
    random_seeds = [i for i in range(num_classes)]

    indices = [[] for x in range(num_classes)]
    for i in range(len(train_set)):
        # puts each index in the list based on the label that it has
        indices[train_set[i][1].item()].append(i)

    for num in range(len(indices)):
        random.Random(random_seeds[num]).shuffle(indices[num]) # https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
        indices[num] = indices[num][:num_samples_per_class]
    
    reduced_train_set = list(itertools.chain.from_iterable(indices))
    random.Random(random_seeds[-1]+1).shuffle(reduced_train_set)

    return torch.utils.data.Subset(train_set, reduced_train_set)

def get_number_of_correct_downstream_preds(y_pred, y):
    num_correct = (torch.argmax(y_pred, dim=1) == y).float().sum().item()
    num_total = len(y)
    return num_correct, num_total

def plot_cv_summary_for_downstream_tests(save_dir, final_plot_title, test_accuracies_per_num_train_samps_on_cv_sets, 
                                         nums_of_training_samples_per_class):
    fig1, ax1 = plt.subplots()
    for cv_num in range(len(test_accuracies_per_num_train_samps_on_cv_sets)):
        ax1.plot(nums_of_training_samples_per_class, test_accuracies_per_num_train_samps_on_cv_sets[cv_num], label="cv split "+str(cv_num))
        ax1.scatter(nums_of_training_samples_per_class, test_accuracies_per_num_train_samps_on_cv_sets[cv_num])
    ax1.set_xscale("log") # see https://stackoverflow.com/questions/773814/plot-logarithmic-axes-with-matplotlib-in-python
    ax1.set_xlabel("Number of Training Samples Per Class")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(final_plot_title)
    plt.legend()
    plt.draw()
    loss_plot_save_path = os.path.join(save_dir, final_plot_title+".png")
    fig1.savefig(loss_plot_save_path)
    pass

def train_downstream_model(save_dir, model_file_name="final_downstream_model.bin", batch_size=256, # hyper parameters for training loop
                    shuffle=True, max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=6, 
                    max_evals_after_saving=6, save_freq=10, sample_log_base=10, final_plot_title="cross_val_summary", 
                    former_state_dict_file=None, ct_dim=100, h_dim=100, channels=11, dropout_rate=0.5, embed_dim=100, # hyper parameters for Downstream Model
                    num_classes=3, path_to_pretrained_embedders=None, Np=16, update_embedders=False, 
                    cached_datasets_list_dir=None, n_cross_val_splits=5, #hyper parameters for data loaders
                    windowed_data_name="_Windowed_Preprocess.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", windowed_label_name="_Windowed_Label.npy", 
                    file_names_list="training_names.txt", val_portion=0.2):
    
    # First, load the training, validation, and test sets
    train_sets, val_sets, test_sets = load_Downstream_Dataset(cached_datasets_list_dir=cached_datasets_list_dir, 
                                                              n_cross_val_splits=n_cross_val_splits, 
                                                              windowed_data_name=windowed_data_name, 
                                                              data_folder_name=data_folder_name, 
                                                              data_root_name=data_root_name, 
                                                              file_names_list=file_names_list, 
                                                              val_portion=val_portion
    )

    # GET LIST OF THE NUMBERS OF SAMPLES PER CLASS ON A LOG SCALE
    max_num_available_samples_from_smallest_class = None
    for cv_split_num in range(len(train_sets)):
        train_set = train_sets[cv_split_num]
        smallest_num_class_samples = get_smallest_class_sample_num_for_downstream_training(train_set, num_classes)
        if max_num_available_samples_from_smallest_class is None or smallest_num_class_samples < max_num_available_samples_from_smallest_class:
            max_num_available_samples_from_smallest_class = smallest_num_class_samples

    nums_of_training_samples_per_class = []
    exponential_counter = 1
    while sample_log_base**exponential_counter < max_num_available_samples_from_smallest_class:
        nums_of_training_samples_per_class.append(sample_log_base**exponential_counter)
        exponential_counter += 1
    nums_of_training_samples_per_class.append(max_num_available_samples_from_smallest_class)

    # PERFORM EXPERIMENTS
    test_accuracies_per_num_train_samps_on_cv_sets = [] # will be list of lists for plotting num_train_samps vs test_acc for each cv split
    for cv_split_num in range(len(train_sets)):
        print("\ntrain_downstream_model: NOW PERFORMING CROSS-VAL EXPERIMENT NUMBER ", cv_split_num, " out of ", len(train_sets))
        # iterate over different levels of training samples
        train_set = train_sets[cv_split_num]
        val_set = val_sets[cv_split_num]
        test_set = test_sets[cv_split_num]
        curr_test_accs_per_num_train_samps = []

        for num_train_samps in nums_of_training_samples_per_class:
            save_dir_for_model = save_dir + os.sep + "DownstreamModel_CVSplit" + str(cv_split_num) + "_NumTrainSamps" + str(num_train_samps)
            os.mkdir(save_dir_for_model) # see geeksforgeeks.org/create-a-directory-in-python/

            print("train_downstream_model: now training with ", num_train_samps, " samples out of ", len(nums_of_training_samples_per_class), " different levels")

            # re-initialize data loaders
            train_set_reduced = restrict_training_size_per_class_for_downstream_training(train_set, num_train_samps, num_classes)
            data_loader_params = {'batch_size': batch_size, 
                                  'shuffle': shuffle
            }#, 'num_workers': num_workers}
            train_loader = torch.utils.data.DataLoader(train_set_reduced, **data_loader_params)
            val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
            test_loader = torch.utils.data.DataLoader(test_set, **data_loader_params)
                    
            # cuda setup if allowed
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0
            
            # initialize downstream model
            embedders = []
            if path_to_pretrained_embedders is None: # case where we are training embedders in tandem with down-stream model (vanilla auto-encoder)
                embedders = [RPNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model, 
                             TSNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model, 
                             CPCNet(Np=Np, channels=channels, ct_dim=ct_dim, h_dim=h_dim, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model, 
                ]
            else: # case where SSL techniques were used to train embedders previously
                embedder_file_names = [x for x in os.listdir(path_to_pretrained_embedders) if x.endswith(".bin")] # see stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
                for file_name in embedder_file_names:
                    if "RP" in file_name:
                        curr_model = RPNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    elif "TS" in file_name:
                        curr_model = TSNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    elif "CPC" in file_name:
                        curr_model = CPCNet(Np=Np, channels=channels, ct_dim=ct_dim, h_dim=h_dim, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    else:
                        raise NotImplementedError("train_downstream_model: A .bin file called "+str(file_name)+" has been found in the pretrained embedder directory which cannot be handled.")
                    curr_model.load_state_dict(torch.load(path_to_pretrained_embedders+os.sep+file_name))
                    embedders.append(curr_model)
            if update_embedders:
                for embedder in embedders:
                    for p in embedder.parameters():
                        p.requires_grad = False

            downstream_model = DownstreamNet(embedders, num_classes, embed_dim=embed_dim)
            if former_state_dict_file is not None:
                downstream_model.load_state_dict(torch.load(former_state_dict_file))
            downstream_model = downstream_model.to(device)
            
            # initialize objective function and optimizer
            objective_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(downstream_model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

            # initialize training state
            min_val_inaccuracy = float("inf")
            min_state = None
            num_evaluations_since_model_saved = 0
            saved_model = None

            avg_train_losses = []
            avg_train_accs = []
            avg_val_accs = []

            # perform training and validation
            for epoch in range(max_epochs):
                downstream_model.train()
                running_train_loss = 0
                num_correct_train_preds = 0
                total_num_train_preds = 0

                # update weights
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device) # transfer to device
                    optimizer.zero_grad()             # zero out any pre-existing gradients

                    y_pred = downstream_model(x)      # make prediction and compute resulting loss
                    loss = objective_function(y_pred, y)

                    # compute accuracy
                    new_num_correct, new_num_total = get_number_of_correct_downstream_preds(y_pred, y)
                    num_correct_train_preds += new_num_correct
                    total_num_train_preds += new_num_total

                    # update weights
                    loss.backward()
                    optimizer.step()

                    # track loss
                    running_train_loss += loss.item()

                    # free up cuda memory
                    del x
                    del y
                    del y_pred
                    torch.cuda.empty_cache()
                
                # perform validation
                num_correct_val_preds = 0
                total_num_val_preds = 0
                with torch.no_grad():
                    downstream_model.eval()

                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)

                        y_pred = downstream_model(x)

                        new_num_correct, new_num_total = get_number_of_correct_downstream_preds(y_pred, y)
                        num_correct_val_preds += new_num_correct
                        total_num_val_preds += new_num_total

                        # free up cuda memory
                        del x
                        del y
                        del y_pred
                        torch.cuda.empty_cache()
                        
                # record averages
                avg_train_losses.append(running_train_loss / len(train_loader))
                avg_train_accs.append(num_correct_train_preds / total_num_train_preds)
                avg_val_accs.append(num_correct_val_preds / total_num_val_preds)

                # check stopping criterion / stop
                incorrect_val_percentage = 1. - (num_correct_val_preds / total_num_val_preds)
                if incorrect_val_percentage < min_val_inaccuracy:
                    num_evaluations_since_model_saved = 0
                    min_val_inaccuracy = incorrect_val_percentage
                    saved_model = downstream_model.state_dict()
                else:
                    num_evaluations_since_model_saved += 1
                    if num_evaluations_since_model_saved >= max_evals_after_saving:
                        print("train_downstream_model: EARLY STOPPING on epoch ", epoch)
                        break

                # check if model should be saved / save model
                if epoch % save_freq == 0:
                    temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_Downstream_model_epoch"+str(epoch)+".bin")
                    torch.save(downstream_model.state_dict(), temp_model_save_path)
                    plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "epoch"+str(epoch), save_dir_for_model)

            print("train_downstream_model: END OF TRAINING - now saving final model / other info")

            # save final model for current experiment, along with hyper params / cv split num and num_train samps
            downstream_model.load_state_dict(saved_model)
            model_save_path = os.path.join(save_dir_for_model, model_file_name)
            torch.save(downstream_model.state_dict(), model_save_path)

            num_correct_test_preds = 0
            total_num_test_preds = 0
            with torch.no_grad():
                downstream_model.eval()

                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    y_pred = downstream_model(x)

                    new_num_correct, new_num_total = get_number_of_correct_downstream_preds(y_pred, y)
                    num_correct_test_preds += new_num_correct
                    total_num_test_preds += new_num_total

                    # free up cuda memory
                    del x
                    del y
                    del y_pred
                    torch.cuda.empty_cache()
            curr_avg_test_acc = num_correct_test_preds / total_num_test_preds
            curr_test_accs_per_num_train_samps.append(curr_avg_test_acc)

            meta_data_save_path = os.path.join(save_dir_for_model, "meta_data_and_hyper_parameters.pkl")
            with open(meta_data_save_path, 'wb') as outfile:
                pkl.dump({
                    "avg_train_losses": avg_train_losses, 
                    "avg_train_accs": avg_train_accs, 
                    "avg_val_accs": avg_val_accs, 
                    "curr_avg_test_acc": curr_avg_test_acc, 
                    "save_dir_for_model": save_dir_for_model, 
                    "model_file_name": model_file_name, 
                    "batch_size": batch_size, 
                    "shuffle": shuffle, #"num_workers": num_workers, 
                    "max_epochs": max_epochs, 
                    "learning_rate": learning_rate, 
                    "beta_vals": beta_vals, 
                    "weight_decay": weight_decay, 
                    "max_evals_after_saving": max_evals_after_saving, 
                    "save_freq": save_freq, 
                    "sample_log_base":sample_log_base, 
                    "former_state_dict_file": former_state_dict_file, 
                    "ct_dim": ct_dim, 
                    "h_dim": h_dim, 
                    "channels": channels, 
                    "dropout_rate": dropout_rate, 
                    "embed_dim": embed_dim,
                    "num_classes": num_classes, 
                    "path_to_pretrained_embedders": path_to_pretrained_embedders, 
                    "update_embedders": update_embedders, 
                    "cached_datasets_list_dir": cached_datasets_list_dir, 
                    "Np": Np, 
                    "windowed_data_name": windowed_data_name,
                    "data_folder_name": data_folder_name, 
                    "data_root_name": data_root_name, 
                    "windowed_label_name": windowed_label_name, 
                    "file_names_list": file_names_list, 
                    "val_portion": val_portion, 
                    "n_cross_val_splits": n_cross_val_splits, 
                    "curr_cross_val_split": cv_split_num, 
                    "num_train_samps": num_train_samps, 
                }, outfile)

            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "Final", save_dir_for_model)
            
            print("train_downstream_model: done with current experiment iteration on cv_split_num==", cv_split_num, " and num_train_samps==", num_train_samps)
            
        test_accuracies_per_num_train_samps_on_cv_sets.append(curr_test_accs_per_num_train_samps)

    # MAKE PLOTS / SUMMARIZE RESULTS OF EXPERIMENTS
    print("train_downstream_model: ALL EXPERIMENTS COMPLETE - SUMMARIZING RESULTS")
    plot_cv_summary_for_downstream_tests(save_dir, final_plot_title, test_accuracies_per_num_train_samps_on_cv_sets, nums_of_training_samples_per_class)
    test_acc_summary_save_path = os.path.join(save_dir, "final_cross_val_test_accuracy_summary.pkl")
    with open(test_acc_summary_save_path, 'wb') as outfile:
                pkl.dump({
                    "test_accuracies_per_num_train_samps_on_cv_sets": test_accuracies_per_num_train_samps_on_cv_sets, 
                    "nums_of_training_samples_per_class": nums_of_training_samples_per_class, 
                }, outfile)
    print("train_downstream_model: DONE!")
    pass