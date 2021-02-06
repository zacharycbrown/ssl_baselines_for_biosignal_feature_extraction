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

import copy


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

def train_PS_model(save_dir_for_model, model_file_name="final_PS_model.bin", batch_size=256, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=4, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, ct_dim=None, h_dim=None, 
                    channels=11, temporal_len=3000, dropout_rate=0.5, embed_dim=100, # hyper parameters for PS Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=None, tneg_val=None, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=None, 
                    windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1):
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset('PS',
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

    print("train_PS_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0

    # initialize model
    model = PSNet(num_channels=channels, temporal_len=temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim).to(device)
    if former_state_dict_file is not None:
        model.load_state_dict(torch.load(former_state_dict_file))

    print("train_PS_model: START OF TRAINING")
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
        # print("train_PS_model: now starting epoch ", epoch, " of ", max_epochs)
        model.train()
        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        
        # iterate over training batches
        # counter = 0
        for x, y in train_loader:
            # transfer to GPU
            x, y = x.to(device), y.to(device)
            # print("x == ", x)
            # print("y == ", y)

            # zero out any pre-existing gradients
            optimizer.zero_grad()

            # make prediction and compute resulting loss
            y_pred = model(x)
            # print("y_pred == ", y_pred)
            loss = loss_fn(y_pred, y)
            # print("loss == ", loss)

            # compute accuracy
            num_correct_train_preds += get_number_of_correct_preds(y_pred, y)
            total_num_train_preds += len(y)

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

            # counter += 1
            # if counter == 5:
            #     raise NotImplementedError()
        
        # iterate over validation batches
        num_correct_val_preds = 0
        total_num_val_preds = 0
        with torch.no_grad():
            model.eval()

            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # evaluate model
                y_pred = model(x)
                num_correct_val_preds += get_number_of_correct_preds(y_pred, y)
                total_num_val_preds += len(y)

                # free up cuda memory
                del x
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
                print("train_PS_model: EARLY STOPPING on epoch ", epoch)
                break
        
        # save intermediate state_dicts just in case
        if epoch % save_freq == 0:
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_PS_model_epoch"+str(epoch)+".bin")
            torch.save(model.state_dict(), temp_model_save_path)
            embedder_save_path = os.path.join(save_dir_for_model, "temp_embedder_epoch"+str(epoch)+".bin")
            torch.save(model.embed_model.state_dict(), embedder_save_path)
            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "epoch"+str(epoch), save_dir_for_model)

    print("train_PS_model: END OF TRAINING - now saving final model / other info")

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
            "temporal_len": temporal_len, 
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
    
    print("train_PS_model: DONE!")
    pass


class ContrastiveLoss(nn.Module):
    """
    see Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf 
    """
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.BATCH_DIM = 0
        self.tau = temperature
        self.cos_sim = torch.nn.CosineSimilarity(0)
        pass
    
    def forward(self, z1s, z2s):
        """
        z1s represents the (batched) representation(s) of the original signal(s)
        z2s represents the (batched) representation(s) of the augmented/perturbed signal(s)
        """
        loss = 0.
        curr_batch_size = z1s.size(self.BATCH_DIM)

        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        for i in range(curr_batch_size):
            # see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

            # compute loss contributions of orig-to-other pairings
            numerator1 = torch.exp(self.cos_sim(z1s[i,:], z2s[i,:]) / self.tau)
            denominator1 = 0.
            for k in range(curr_batch_size):
                denominator1 += torch.exp(self.cos_sim(z1s[i,:], z2s[k,:]) / self.tau) # compare orig ith signal with all augmented signals
                if k != i:                                                             # compare orig ith signal to all other orig signals, skipping the original ith signal
                    denominator1 += torch.exp(self.cos_sim(z1s[i,:], z1s[k,:]) / self.tau)
            loss += -1.*torch.log(numerator1/denominator1)
            
            # compute loss contributions of augmented-to-other pairings
            numerator2 = torch.exp(self.cos_sim(z2s[i,:], z1s[i,:]) / self.tau)
            denominator2 = 0.
            for k in range(curr_batch_size):
                denominator2 += torch.exp(self.cos_sim(z2s[i,:], z1s[k,:]) / self.tau) # compare augmented ith signal with all orig signals
                if k != i:                                                             # compare augmented ith signal to all other augmented signals, skipping the augmented ith signal
                    denominator2 += torch.exp(self.cos_sim(z2s[i,:], z2s[k,:]) / self.tau)
            loss += -1.*torch.log(numerator2/denominator2)

        loss = loss / (curr_batch_size*2.*(2.*curr_batch_size - 1.)) # take the average loss across the entire batches of both augmented and original signals
        return loss

    def get_number_of_correct_sq_reps(self, z1s, z2s):
        curr_batch_size = z1s.size(self.BATCH_DIM)

        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        num_correct_reps = 0.
        for i in range(curr_batch_size):
            # see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

            # compute accuracy contributions of orig-to-other pairings
            sim_measure_of_interest = self.cos_sim(z1s[i,:], z2s[i,:])
            representation_is_correct = True
            for k in range(curr_batch_size):
                other_sim_measure = self.cos_sim(z1s[i,:], z2s[k,:]) # compare orig ith signal with all augmented signals
                if other_sim_measure > sim_measure_of_interest:
                    representation_is_correct = False
                    break
                if k != i:                                                             # compare orig ith signal to all other orig signals, skipping the original ith signal
                    other_sim_measure = self.cos_sim(z1s[i,:], z1s[k,:])
                    if other_sim_measure > sim_measure_of_interest:
                        representation_is_correct = False
                        break
            if representation_is_correct:
                num_correct_reps += 1.
            
            # compute loss contributions of augmented-to-other pairings
            sim_measure_of_interest = self.cos_sim(z2s[i,:], z1s[i,:])
            representation_is_correct = True
            for k in range(curr_batch_size):
                other_sim_measure += self.cos_sim(z2s[i,:], z1s[k,:]) # compare augmented ith signal with all orig signals
                if other_sim_measure > sim_measure_of_interest:
                    representation_is_correct = False
                    break
                if k != i:                                                             # compare augmented ith signal to all other augmented signals, skipping the augmented ith signal
                    other_sim_measure += self.cos_sim(z2s[i,:], z2s[k,:])
                    if other_sim_measure > sim_measure_of_interest:
                        representation_is_correct = False
                        break
            if representation_is_correct:
                num_correct_reps += 1.

        return num_correct_reps

def train_SQ_model(save_dir_for_model, model_file_name="final_SQ_model.bin", batch_size=256, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=4, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, ct_dim=None, h_dim=None, 
                    channels=11, temporal_len=3000, dropout_rate=0.5, embed_dim=100, encoder_type="convolutional", bw=5, randomized_augmentation=False, num_upstream_decode_features=32, temperature=0.05, # hyper parameters for SQ Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=None, tneg_val=None, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=None, 
                    windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1):

    cudnn_actively_disabled = False
    if torch.backends.cudnn.enabled and encoder_type != "simplified":
        print("train_SQ_model: CUDNN BEING TEMPORARILY DISABLED FOR THIS TRAINING RUN TO ENABLE LSTM USAGE")
        torch.backends.cudnn.enabled = False # This line is needed for running LSTM on Windows 10 with pytorch - see https://github.com/pytorch/pytorch/issues/27837
        cudnn_actively_disabled = True
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset('SQ', 
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
                                                    bw=bw,                                              # items unique to SQ data loading
                                                    randomized_augmentation=randomized_augmentation,    # items unique to SQ data loading
                                                    num_channels=channels,                              # items unique to SQ data loading
                                                    temporal_len=temporal_len,                          # items unique to SQ data loading
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

    print("train_SQ_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0

    # initialize model
    model = SQNet(encoder_type, channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features).to(device)
    if former_state_dict_file is not None:
        model.load_state_dict(torch.load(former_state_dict_file))

    print("train_SQ_model: START OF TRAINING")
    # initialize training state
    min_val_inaccuracy = float("inf")
    min_state = None
    num_evaluations_since_model_saved = 0
    saved_model = None
    loss_fn = ContrastiveLoss(temperature=temperature)
    # learning_rate = learning_rate
    # beta_vals = beta_vals
    optimizer = torch.optim.Adam(model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    # Iterate over epochs
    avg_train_losses = []
    avg_train_accs = []
    avg_val_accs = []
    for epoch in range(max_epochs):
        print("train_SQ_model: epoch ", epoch, " of ", max_epochs)
        model.train()
        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        
        # iterate over training batches
        print("train_SQ_model: \tNow performing training updates")
        counter = 0
        for x1, x2 in train_loader:
            # transfer to GPU
            x1, x2 = x1.to(device), x2.to(device)
            # print("x1 == ", x1.shape)
            # print("x2 == ", x2.shape)

            # zero out any pre-existing gradients
            optimizer.zero_grad()

            # make prediction and compute resulting loss
            # print("train_SQ_model: \t\tembedding x1")
            x1_rep = model(x1)
            # print("train_SQ_model: \t\tembedding x2")
            x2_rep = model(x2)
            # print("x1_rep == ", x1_rep.shape)
            # print("x2_rep == ", x2_rep.shape)
            # print("train_SQ_model: \t\tcomputing loss")
            loss = loss_fn(x1_rep, x2_rep)
            # print("loss == ", loss)

            # compute accuracy
            # print("train_SQ_model: \t\tcomputing accuracy")
            num_correct_train_preds += loss_fn.get_number_of_correct_sq_reps(x1_rep, x2_rep)
            # print("train_SQ_model: \t\trecording accuracy")
            total_num_train_preds += 2.*len(x1_rep)

            # update weights
            # print("train_SQ_model: \t\tperforming backprop")
            loss.backward()
            # print("train_SQ_model: \t\tupdating weights")
            optimizer.step()

            # track loss
            # print("train_SQ_model: \t\trecording loss val")
            running_train_loss += loss.item()

            # free up cuda memory
            # print("train_SQ_model: \t\tclearing memory")
            del x1
            del x2
            del x1_rep
            del x2_rep
            del loss
            torch.cuda.empty_cache()

            if counter % 50 == 0:
                print("train_SQ_model: \t\tFinished batch ", counter)
            counter += 1
            # if counter == 5:
            #     raise NotImplementedError()
            # raise NotImplementedError()
        
        # iterate over validation batches
        print("train_SQ_model: \tNow performing validation")
        num_correct_val_preds = 0
        total_num_val_preds = 0
        with torch.no_grad():
            model.eval()

            for x1, x2 in val_loader:
                x1, x2 = x1.to(device), x2.to(device)

                # evaluate model
                x1_rep = model(x1)
                x2_rep = model(x2)
                num_correct_val_preds += loss_fn.get_number_of_correct_sq_reps(x1_rep, x2_rep)
                total_num_val_preds += 2.*len(x1_rep)

                # free up cuda memory
                del x1
                del x2
                del x1_rep
                del x2_rep
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
                print("train_SQ_model: EARLY STOPPING on epoch ", epoch)
                break
        
        # save intermediate state_dicts just in case
        if epoch % save_freq == 0:
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_SQ_model_epoch"+str(epoch)+".bin")
            torch.save(model.state_dict(), temp_model_save_path)
            embedder_save_path = os.path.join(save_dir_for_model, "temp_embedder_epoch"+str(epoch)+".bin")
            torch.save(model.embed_model.state_dict(), embedder_save_path)
            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "epoch"+str(epoch), save_dir_for_model)

    print("train_SQ_model: END OF TRAINING - now saving final model / other info")

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
            "temporal_len": temporal_len, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "encoder_type": encoder_type, 
            "bw": bw, 
            "randomized_augmentation": randomized_augmentation, 
            "num_upstream_decode_features": num_upstream_decode_features, 
            "temperature": temperature, 
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

    if cudnn_actively_disabled:
        torch.backends.cudnn.enabled = True # see beginning of function and https://github.com/pytorch/pytorch/issues/27837
    
    print("train_SQ_model: DONE!")
    pass



class SAContrastiveAdversarialLoss(nn.Module):
    """
    see Section 3.1 of arxiv.org/pdf/2007.04871.pdf
    """
    def __init__(self, temperature, adversarial_weighting_factor=1):
        super(SAContrastiveAdversarialLoss, self).__init__()
        self.BATCH_DIM = 0
        self.tau = temperature
        self.lam = adversarial_weighting_factor
        self.cos_sim = torch.nn.CosineSimilarity(0)
        self.log_noise = 1e-12 # 8 # see https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
        # self.contrastive_loss = ContrastiveLoss(temperature)
        pass
    
    def forward(self, z1s, z2s, z1_c_outs, z1_subject_labels):
        """
        z1s represents the (batched) representation(s) of the t1-transformed signal(s)
        z2s represents the (batched) representation(s) of the t2-transformed signal(s)
        z1_c_outs represents the (batched) subject predictions produced by the adversary
        z1_subject_labels represents the (batched) subject labels, representing the ground truth for the adversary

        see Sectoin 3.1 of arxiv.org/pdf/2007.04871.pdf
        """
        z1_c_outs = torch.nn.functional.normalize(z1_c_outs, p=2, dim=1) # see https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209

        loss = 0.
        curr_batch_size = z1s.size(self.BATCH_DIM)

        # get contrastive loss of representations
        # loss += self.contrastive_loss(z1s, z2s)
        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        for i in range(curr_batch_size):
            # see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

            # compute loss contributions of t1-to-other pairings
            numerator1 = torch.exp(self.cos_sim(z1s[i,:], z2s[i,:]) / self.tau)
            denominator1 = 0.
            for k in range(curr_batch_size):
                denominator1 += torch.exp(self.cos_sim(z1s[i,:], z2s[k,:]) / self.tau) # compare t1 ith signal with all t2 signals
                if k != i:                                                             # compare t1 ith signal to all other t1 signals, skipping the t1 ith signal
                    denominator1 += torch.exp(self.cos_sim(z1s[i,:], z1s[k,:]) / self.tau)
            loss += -1.*torch.log(self.log_noise + numerator1/denominator1)
            # print("SAContrastiveAdversarialLoss.forward: \t loss == ", loss)
            
            # SKIP loss contributions of t2-to-other pairings because they came from momentum-updated network
            # numerator2 = torch.exp(self.cos_sim(z2s[i,:], z1s[i,:]) / self.tau)
            # denominator2 = 0.
            # for k in range(curr_batch_size):
            #     denominator2 += torch.exp(self.cos_sim(z2s[i,:], z1s[k,:]) / self.tau) # compare augmented ith signal with all orig signals
            #     if k != i:                                                             # compare augmented ith signal to all other augmented signals, skipping the augmented ith signal
            #         denominator2 += torch.exp(self.cos_sim(z2s[i,:], z2s[k,:]) / self.tau)
            # loss += -1.*torch.log(numerator2/denominator2)

        loss = loss / (curr_batch_size*(2.*curr_batch_size - 1.)) # loss / (curr_batch_size*2.*(2.*curr_batch_size - 1.)) # take the average loss across the t1 signals 

        for i in range(curr_batch_size):
            j = torch.argmax(z1_subject_labels[i,:])
            loss += self.lam *(-1.)*torch.log(self.log_noise + (1. - z1_c_outs[i,j])) # see equation 3 of arxiv.org/pdf/2007.04871.pdf
            # print("SAContrastiveAdversarialLoss.forward: \t loss == ", loss)
        
        return loss

    def get_number_of_correct_reps(self, z1s, z2s, z1_c_outs, z1_subject_labels):
        curr_batch_size = z1s.size(self.BATCH_DIM)

        z1s = z1s.view(curr_batch_size, -1)
        z2s = z2s.view(curr_batch_size, -1)

        num_correct_reps = 0.
        for i in range(curr_batch_size):
            # see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

            # compute accuracy contributions of orig-to-other pairings
            sim_measure_of_interest = self.cos_sim(z1s[i,:], z2s[i,:])
            representation_is_correct = True
            for k in range(curr_batch_size):
                other_sim_measure = self.cos_sim(z1s[i,:], z2s[k,:]) # compare t1 ith signal with all augmented signals
                if other_sim_measure > sim_measure_of_interest:
                    representation_is_correct = False
                    break
                if k != i:                                           # compare t1 ith signal to all other orig signals, skipping the t1 ith signal
                    other_sim_measure = self.cos_sim(z1s[i,:], z1s[k,:])
                    if other_sim_measure > sim_measure_of_interest:
                        representation_is_correct = False
                        break
                
            if torch.argmax(z1_subject_labels[i,:]) == torch.argmax(z1_c_outs[i,:]):
                representation_is_correct = False

            if representation_is_correct:
                num_correct_reps += 1.
            
            # SKIP loss contributions of t2-to-other pairings because they were generated by momentum-updated network
            # sim_measure_of_interest = self.cos_sim(z2s[i,:], z1s[i,:])
            # representation_is_correct = True
            # for k in range(curr_batch_size):
            #     other_sim_measure += self.cos_sim(z2s[i,:], z1s[k,:]) # compare augmented ith signal with all orig signals
            #     if other_sim_measure > sim_measure_of_interest:
            #         representation_is_correct = False
            #         break
            #     if k != i:                                                             # compare augmented ith signal to all other augmented signals, skipping the augmented ith signal
            #         other_sim_measure += self.cos_sim(z2s[i,:], z2s[k,:])
            #         if other_sim_measure > sim_measure_of_interest:
            #             representation_is_correct = False
            #             break
            # if representation_is_correct:
            #     num_correct_reps += 1.

        return num_correct_reps

class SAAdversarialLoss(nn.Module):
    """
    see Section 3.1 of arxiv.org/pdf/2007.04871.pdf
    """
    def __init__(self):
        super(SAAdversarialLoss, self).__init__()
        self.BATCH_DIM = 0
        self.log_noise = 1e-12 # 8 # see https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
        pass
    
    def forward(self, z1_c_outs, z1_subject_labels):
        """
        z1_c_outs represents the (batched) subject predictions produced by the adversary
        z1_subject_labels represents the (batched) subject labels, representing the ground truth for the adversary

        see Sectoin 3.1 of arxiv.org/pdf/2007.04871.pdf
        """
        # print("z1_c_outs.shape == ", z1_c_outs.shape)
        # print("z1_c_outs == ", z1_c_outs)
        z1_c_outs = torch.nn.functional.normalize(z1_c_outs, p=2, dim=1) # see https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209
        # print("z1_c_outs == ", z1_c_outs)
        # print("z1_c_outs.shape == ", z1_c_outs.shape)

        loss = 0.
        curr_batch_size = z1_c_outs.size(self.BATCH_DIM)

        for i in range(curr_batch_size):
            j = torch.argmax(z1_subject_labels[i,:])
            loss += -1.*torch.log(self.log_noise + z1_c_outs[i,j]) # see equation 3 of arxiv.org/pdf/2007.04871.pdf
            # print("SAAdversarialLoss.forward: \t loss == ", loss, " (i,j) == ", (i,j), " z1_c_outs[i,j] == ", z1_c_outs[i,j])
        # raise NotImplementedError()
        return loss

    def get_number_of_correct_preds(self, z1_c_outs, z1_subject_labels):
        num_correct_preds = 0.
        curr_batch_size = z1_c_outs.size(self.BATCH_DIM)
        
        for i in range(curr_batch_size):
            if torch.argmax(z1_subject_labels[i,:]) == torch.argmax(z1_c_outs[i,:]):
                num_correct_preds += 1.

        return num_correct_preds

def momentum_model_parameter_update(momentum_factor, momentum_model, orig_model): # see https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/16
    for momentum_model_param, orig_model_param in zip(momentum_model.parameters(), orig_model.parameters()):
        momentum_model_param.copy_(momentum_factor*momentum_model_param.data + (1.-momentum_factor)*orig_model_param.data)
    return momentum_model

def train_SA_model(save_dir_for_model, model_file_name="final_SA_model.bin", batch_size=256, shuffle=True, # hyper parameters for training loop
                    max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, #num_workers=4, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, ct_dim=None, h_dim=None, 
                    channels=11, temporal_len=3000, dropout_rate=0.5, embed_dim=100, encoder_type=None, bw=5, # hyper parameters for SA Model
                    randomized_augmentation=False, num_upstream_decode_features=32, temperature=0.05, NUM_AUGMENTATIONS=2, perturb_orig_signal=True, former_adversary_state_dict_file=None, adversarial_weighting_factor=1., momentum=0.999, # hyper parameters for SA Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=None, tneg_val=None, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, Nc=None, Np=None, Nb=None, max_Nb_iters=None, total_points_factor=None, 
                    windowed_data_name="_Windowed_Pretext_Preprocess.npy", 
                    windowed_start_time_name="_Windowed_StartTime.npy", data_folder_name="Mouse_Training_Data", 
                    data_root_name="Windowed_Data", file_names_list="training_names.txt", train_portion=0.7, 
                    val_portion=0.2, test_portion=0.1):
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset('SA', 
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
                                                    bw=bw,                                              # items for SA data loading
                                                    randomized_augmentation=randomized_augmentation,    # items for SA data loading
                                                    num_channels=channels,                              # items for SA data loading
                                                    temporal_len=temporal_len,                          # items for SA data loading
                                                    NUM_AUGMENTATIONS=NUM_AUGMENTATIONS,                # items for SA data loading
                                                    perturb_orig_signal=perturb_orig_signal,            # items for SA data loading
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

    print("train_SA_model: len of the train_loader is ", len(train_loader))

    # cuda setup if allowed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0

    # initialize models - see Figure 1 of arxiv.org/pdf/2007.04871.pdf
    model = SACLNet(channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features)
    if former_state_dict_file is not None:
        model.load_state_dict(torch.load(former_state_dict_file))
    momentum_model = copy.deepcopy(model) # see https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492 and https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
    model = model.to(device)
    momentum_model = momentum_model.to(device)

    _, _, y0 = next(iter(train_loader))
    assert len(y0.shape) == 2 
    num_subjects = y0.shape[1]
    adversary = SACLAdversary(embed_dim, num_subjects, dropout_rate=dropout_rate).to(device)
    if former_adversary_state_dict_file is not None:
        adversary.load_state_dict(torch.load(former_adversary_state_dict_file))

    print("train_SA_model: START OF TRAINING")
    # initialize training state
    min_val_inaccuracy = float("inf")
    min_state = None
    num_evaluations_since_model_saved = 0
    saved_model = None
    saved_momentum_model = None
    loss_fn = SAContrastiveAdversarialLoss(temperature, adversarial_weighting_factor=adversarial_weighting_factor)
    # learning_rate = learning_rate
    # beta_vals = beta_vals
    optimizer = torch.optim.Adam(model.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    saved_adversary = None
    adversarial_loss_fn = SAAdversarialLoss()
    adversarial_optimizer = torch.optim.Adam(adversary.parameters(), betas=beta_vals, lr=learning_rate, weight_decay=weight_decay)

    # Iterate over epochs
    avg_train_losses = []
    avg_train_accs = []
    avg_val_accs = []
    avg_adversary_train_losses = []
    avg_adversary_train_accs = []
    avg_adversary_val_accs = []
    for epoch in range(max_epochs):
        print("train_SA_model: epoch ", epoch, " of ", max_epochs)

        model.train()
        momentum_model.train()
        adversary.train()

        running_train_loss = 0
        num_correct_train_preds = 0
        total_num_train_preds = 0
        running_adversary_train_loss = 0
        num_adversary_correct_train_preds = 0
        total_num_adversary_train_preds = 0
        
        # iterate over training batches
        print("train_SA_model: \tNow performing training updates")
        counter = 0
        for x_t1, x_t2, y in train_loader:
            # transfer to GPU
            x_t1, x_t2, y = x_t1.to(device), x_t2.to(device), y.to(device)
            # print("x_t1 == ", x_t1.shape)
            # print("x_t2 == ", x_t2.shape)
            # print("y == ", y.shape)

            # UPDATE ADVERSARY
            for p in model.parameters():
                p.requires_grad = False
            for p in momentum_model.parameters():
                p.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = True
            
            adversarial_optimizer.zero_grad()
            
            x_t1_initial_reps = model.embed_model(x_t1)
            x_t1_initial_subject_preds = adversary(x_t1_initial_reps)

            adversarial_loss = adversarial_loss_fn(x_t1_initial_subject_preds, y)
            num_adversary_correct_train_preds += adversarial_loss_fn.get_number_of_correct_preds(x_t1_initial_subject_preds, y)
            total_num_adversary_train_preds += len(x_t1_initial_subject_preds)

            adversarial_loss.backward()
            adversarial_optimizer.step()

            running_adversary_train_loss += adversarial_loss.item()

            del x_t1_initial_reps
            del x_t1_initial_subject_preds
            del adversarial_loss
            torch.cuda.empty_cache()

            # UPDATE MODEL - references Algorithm 1 of arxiv.org/pdf/1911.05722.pdf and Figure 1 of arxiv.org/pdf/2007.04871.pdf
            for p in model.parameters():
                p.requires_grad = True
            for p in momentum_model.parameters():
                p.requires_grad = False
            for p in adversary.parameters():
                p.requires_grad = False

            # zero out any pre-existing gradients
            optimizer.zero_grad()

            # make prediction and compute resulting loss
            # print("train_SA_model: \t\tembedding x1")
            x1_rep = model(x_t1)
            # print("train_SA_model: \t\tembedding x2")
            x2_rep = momentum_model(x_t2)
            # x2_rep.detatch()
            # print("x1_rep == ", x1_rep.shape)
            # print("x2_rep == ", x2_rep.shape)
            x1_embeds = model.embed_model(x_t1)
            x1_subject_preds = adversary(x1_embeds)
            # print("train_SA_model: \t\tcomputing loss")
            loss = loss_fn(x1_rep, x2_rep, x1_subject_preds, y)
            # print("loss == ", loss)

            # compute accuracy
            # print("train_SA_model: \t\tcomputing accuracy")
            num_correct_train_preds += loss_fn.get_number_of_correct_reps(x1_rep, x2_rep, x1_subject_preds, y)
            # print("train_SQ_model: \t\trecording accuracy")
            total_num_train_preds += len(x1_rep)

            # update weights
            # print("train_SA_model: \t\tperforming backprop")
            loss.backward()
            # print("train_SA_model: \t\tupdating weights")
            optimizer.step()

            # track loss
            # print("train_SA_model: \t\trecording loss val")
            running_train_loss += loss.item()

            # UPDATE MOMENTUM MODEL
            # momentum_model.parameters = momentum*momentum_model.parameters + (1.-momentum)*model.parameters
            momentum_model = momentum_model_parameter_update(momentum, momentum_model, model)

            # free up cuda memory
            # print("train_SA_model: \t\tclearing memory")
            del x_t1
            del x_t2
            del x1_rep
            del x2_rep
            del x1_embeds
            del x1_subject_preds
            del loss
            torch.cuda.empty_cache()

            if counter % 50 == 0:
                print("train_SA_model: \t\tFinished batch ", counter)
            counter += 1
            # if counter == 5:
            #     raise NotImplementedError()
            # raise NotImplementedError()
            # break # FOR DEBUGGING PURPOSES
        
        # iterate over validation batches
        print("train_SA_model: \tNow performing validation")
        num_correct_val_preds = 0
        total_num_val_preds = 0
        num_correct_adversarial_val_preds = 0
        total_num_adversarial_val_preds = 0
        with torch.no_grad():
            model.eval()
            momentum_model.eval()
            adversary.eval()

            for x_t1, x_t2, y in val_loader:
                x_t1, x_t2, y = x_t1.to(device), x_t2.to(device), y.to(device)

                # evaluate model and adversary
                x1_rep = model(x_t1)
                x2_rep = momentum_model(x_t2)
                x1_embeds = model.embed_model(x_t1)
                x1_subject_preds = adversary(x1_embeds)
                # x1_subject_preds = adversary(x1_rep)

                num_correct_val_preds += loss_fn.get_number_of_correct_reps(x1_rep, x2_rep, x1_subject_preds, y)
                total_num_val_preds += len(x1_rep)

                num_correct_adversarial_val_preds += adversarial_loss_fn.get_number_of_correct_preds(x1_subject_preds, y)
                total_num_adversarial_val_preds += len(x1_subject_preds)

                # free up cuda memory
                del x_t1
                del x_t2
                del x1_rep
                del x2_rep
                del x1_embeds
                del x1_subject_preds
                torch.cuda.empty_cache()
                # break # FOR DEBUGGING PURPOSES
        
        # record averages
        avg_train_accs.append(num_correct_train_preds / total_num_train_preds)
        avg_val_accs.append(num_correct_val_preds / total_num_val_preds)
        avg_train_losses.append(running_train_loss / len(train_loader))
        
        avg_adversary_train_accs.append(num_adversary_correct_train_preds / total_num_adversary_train_preds)
        avg_adversary_val_accs.append(num_correct_adversarial_val_preds / total_num_adversarial_val_preds)
        avg_adversary_train_losses.append(running_adversary_train_loss / len(train_loader))
        
        # check stopping criterion / save model
        incorrect_val_percentage = 1. - (num_correct_val_preds / total_num_val_preds)
        if incorrect_val_percentage < min_val_inaccuracy:
            num_evaluations_since_model_saved = 0
            min_val_inaccuracy = incorrect_val_percentage
            saved_model = model.state_dict()
            saved_momentum_model = momentum_model.state_dict()
            saved_adversary = adversary.state_dict()
        else:
            num_evaluations_since_model_saved += 1
            if num_evaluations_since_model_saved >= max_evals_after_saving:
                print("train_SA_model: EARLY STOPPING on epoch ", epoch)
                break
        
        # save intermediate state_dicts just in case
        if epoch % save_freq == 0:
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_SA_model_epoch"+str(epoch)+".bin")
            torch.save(model.state_dict(), temp_model_save_path)
            
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_SA_momentum_model_epoch"+str(epoch)+".bin")
            torch.save(momentum_model.state_dict(), temp_model_save_path)
            
            temp_model_save_path = os.path.join(save_dir_for_model, "temp_full_SA_adversary_epoch"+str(epoch)+".bin")
            torch.save(adversary.state_dict(), temp_model_save_path)

            embedder_save_path = os.path.join(save_dir_for_model, "temp_embedder_epoch"+str(epoch)+".bin")
            torch.save(model.embed_model.state_dict(), embedder_save_path)

            plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "model_epoch"+str(epoch), save_dir_for_model)
            plot_avgs(avg_adversary_train_losses, avg_adversary_train_accs, avg_adversary_val_accs, "adversary_epoch"+str(epoch), save_dir_for_model)
        # break # FOR DEBUGGING PURPOSES

    print("train_SA_model: END OF TRAINING - now saving final model / other info")

    # save final model(s)
    model.load_state_dict(saved_model)
    model_save_path = os.path.join(save_dir_for_model, model_file_name)
    torch.save(model.state_dict(), model_save_path)

    momentum_model.load_state_dict(saved_momentum_model)
    model_save_path = os.path.join(save_dir_for_model, "momentum_model_"+model_file_name)
    torch.save(model.state_dict(), model_save_path)

    adversary.load_state_dict(saved_adversary)
    model_save_path = os.path.join(save_dir_for_model, "adversary_"+model_file_name)
    torch.save(model.state_dict(), model_save_path)

    embedder_save_path = os.path.join(save_dir_for_model, "embedder_"+model_file_name)
    torch.save(model.embed_model.state_dict(), embedder_save_path)

    meta_data_save_path = os.path.join(save_dir_for_model, "meta_data_and_hyper_parameters.pkl")
    with open(meta_data_save_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": avg_train_losses, 
            "avg_train_accs": avg_train_accs, 
            "avg_val_accs": avg_val_accs, 
            "avg_adversary_train_losses": avg_adversary_train_losses, 
            "avg_adversary_train_accs": avg_adversary_train_accs, 
            "avg_adversary_val_accs": avg_adversary_val_accs, 
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
            "temporal_len": temporal_len, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "encoder_type": encoder_type, 
            "bw": bw, 
            "randomized_augmentation": randomized_augmentation, 
            "num_upstream_decode_features": num_upstream_decode_features, 
            "temperature": temperature, 
            "NUM_AUGMENTATIONS": NUM_AUGMENTATIONS, 
            "perturb_orig_signal": perturb_orig_signal, 
            "former_adversary_state_dict_file": former_adversary_state_dict_file, 
            "adversarial_weighting_factor": adversarial_weighting_factor, 
            "momentum": momentum, 
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

    plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, "Final_Model", save_dir_for_model)
    plot_avgs(avg_adversary_train_losses, avg_adversary_train_accs, avg_adversary_val_accs, "Final_Adversary", save_dir_for_model)
    
    print("train_SA_model: DONE!")
    pass

# DOWNSTREAM CODE BLOCK #################################################################################################

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
                    former_state_dict_file=None, ct_dim=100, h_dim=100, channels=11, temporal_len=3000, dropout_rate=0.5, embed_dim=100, # hyper parameters for Downstream Model
                    num_classes=3, path_to_pretrained_embedders=None, Np=16, update_embedders=False, sq_encoder_type='simplified', num_upstream_decode_features=32,
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
                embedders = [["RP", RPNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model], 
                             ["TS", TSNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model], 
                             ["CPC", CPCNet(Np=Np, channels=channels, ct_dim=ct_dim, h_dim=h_dim, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model], 
                             ["PS", PSNet(num_channels=channels, temporal_len=temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model], 
                             ["SQ", SQNet(sq_encoder_type, channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features).embed_model], 
                             ["SA", SACLNet(channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features).embed_model], 
                ]
            else: # case where SSL techniques were used to train embedders previously
                embedder_file_names = [x for x in os.listdir(path_to_pretrained_embedders) if x.endswith(".bin")] # see stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python
                for file_name in embedder_file_names:
                    curr_model_type = None
                    curr_model = None
                    if "RP" in file_name:
                        curr_model_type = "RP"
                        curr_model = RPNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    elif "TS" in file_name:
                        curr_model_type = "TS"
                        curr_model = TSNet(channels=channels, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    elif "CPC" in file_name:
                        curr_model_type = "CPC"
                        curr_model = CPCNet(Np=Np, channels=channels, ct_dim=ct_dim, h_dim=h_dim, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    elif "PS" in file_name:
                        curr_model_type = "PS"
                        curr_model = PSNet(num_channels=channels, temporal_len=temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim).embed_model
                    elif "SQ" in file_name:
                        curr_model_type = "SQ"
                        curr_model = SQNet(sq_encoder_type, channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features).embed_model
                        print("train_downstream_model: ASSUMING PRETRAINED SQ MODEL TYPE IS <<< ", sq_encoder_type, " >>> AND THAT THE STATE-DICT STORED AT ", str(path_to_pretrained_embedders+os.sep+file_name), " MATCHES THIS ARCHITECTURE.")
                    elif "SA" in file_name:
                        curr_model_type = "SA"
                        curr_model = SACLNet(channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim, num_upstream_decode_features=num_upstream_decode_features).embed_model
                    else:
                        raise NotImplementedError("train_downstream_model: A .bin file called "+str(file_name)+" has been found in the pretrained embedder directory which cannot be handled.")
                    curr_model.load_state_dict(torch.load(path_to_pretrained_embedders+os.sep+file_name))
                    embedders.append([curr_model_type, curr_model])
            if update_embedders:
                for _, embedder in embedders:
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
                    # print("train_downstream : x shape == ", x.shape)
                    # print("train_downstream : y shape == ", y.shape)

                    y_pred = downstream_model(x)      # make prediction and compute resulting loss
                    # print("train_downstream : y_pred shape == ", y_pred.shape)
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

                    # raise NotImplementedError()
                
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
                    "temporal_len": temporal_len, 
                    "dropout_rate": dropout_rate, 
                    "embed_dim": embed_dim,
                    "num_classes": num_classes, 
                    "path_to_pretrained_embedders": path_to_pretrained_embedders, 
                    "update_embedders": update_embedders, 
                    "sq_encoder_type": sq_encoder_type, 
                    "num_upstream_decode_features": num_upstream_decode_features,
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