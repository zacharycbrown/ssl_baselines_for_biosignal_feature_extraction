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


def get_number_of_correct_preds(y_pred, y_true):
    return ((y_pred*y_true) > 0).float().sum().item()


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
                    num_workers=4, max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, 
                    channels=11, dropout_rate=0.5, embed_dim=100, # hyper parameters for RP Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, data_folder_name="Mouse_Training_Data", data_root_name="Windowed_Data", 
                    file_names_list="training_names.txt", train_portion=0.7, val_portion=0.2, test_portion=0.1):
    
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
            "shuffle": shuffle, 
            "num_workers": num_workers, 
            "max_epochs": max_epochs, 
            "learning_rate": learning_rate, 
            "beta_vals": beta_vals, 
            "weight_decay": weight_decay, 
            "max_evals_after_saving": max_evals_after_saving, 
            "save_freq": save_freq, 
            "former_state_dict_file": former_state_dict_file, 
            "channels": channels, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "cached_datasets_list_dir": cached_datasets_list_dir, 
            "total_points_val": total_points_val, 
            "tpos_val": tpos_val, 
            "tneg_val": tneg_val, 
            "window_size": window_size,
            "sfreq": sfreq, 
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
                    num_workers=4, max_epochs=100, learning_rate=5e-4, beta_vals=(0.9, 0.999), weight_decay=0.001, 
                    max_evals_after_saving=6, save_freq=20, former_state_dict_file=None, 
                    channels=11, dropout_rate=0.5, embed_dim=100, # hyper parameters for RP Model
                    cached_datasets_list_dir=None, total_points_val=2000, tpos_val=30, tneg_val=120, window_size=3, #hyper parameters for data loaders
                    sfreq=1000, data_folder_name="Mouse_Training_Data", data_root_name="Windowed_Data", 
                    file_names_list="training_names.txt", train_portion=0.7, val_portion=0.2, test_portion=0.1):
    
    # First, load the training, validation, and test sets
    train_set, val_set, test_set = load_SSL_Dataset('TS',
                                                    cached_datasets_list_dir=cached_datasets_list_dir, 
                                                    total_points_val=total_points_val, 
                                                    tpos_val=tpos_val, 
                                                    tneg_val=tneg_val, 
                                                    window_size=window_size, 
                                                    sfreq=sfreq, 
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
            "shuffle": shuffle, 
            "num_workers": num_workers, 
            "max_epochs": max_epochs, 
            "learning_rate": learning_rate, 
            "beta_vals": beta_vals, 
            "weight_decay": weight_decay, 
            "max_evals_after_saving": max_evals_after_saving, 
            "save_freq": save_freq, 
            "former_state_dict_file": former_state_dict_file, 
            "channels": channels, 
            "dropout_rate": dropout_rate, 
            "embed_dim": embed_dim,
            "cached_datasets_list_dir": cached_datasets_list_dir, 
            "total_points_val": total_points_val, 
            "tpos_val": tpos_val, 
            "tneg_val": tneg_val, 
            "window_size": window_size,
            "sfreq": sfreq, 
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