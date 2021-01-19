# libraries

import torch
import torch.optim as optim
import os
from helpers import load_config, create_writer, add_zeros, freeze_module, unfreeze_module
from loss import loss_function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from testing import test
import numpy as np

def train(cfg_path, model, data_set, verbose = True):
    """
    Add documentation.
    :param cfg_path:
    :param model:
    :param data_set:
    :param verbose:
    :return:
    """
    cfg_file = load_config(cfg_path)

    num_epochs = cfg_file["train"]["num_epochs"]
    lr = cfg_file["train"]["lr"]
    device = cfg_file["model"]["device"]
    save = cfg_file["model"]["save"]
    batch_frequency_trace = cfg_file["tracing"]["batch_frequency"]
    batch_frequency_loss = cfg_file["train"]["batch_frequency_loss"]
    evolution = cfg_file["train"]["evolution"]
    epochs_frequency_evolution = cfg_file["train"]["epochs_frequency_evolution"]
    save_evolution = cfg_file["train"]["save_evolution"]
    batch_size = cfg_file["train"]["batch_size"]
    p_ref_opt = cfg_file["train"]["p_ref"]

    # make the generator train_loader for the train_dataset
    train_dataset, _ = data_set
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    N = len(train_dataset)
    num_batches = N // batch_size

    # TODO: Consider cases when the training hyperparamters are different,
    #  now the training is the same for all data sets

    # the dimension of the latent space corresponds to the number of clusters
    dim = cfg_file["model"]["latent_dim"]
    centers = torch.eye(dim) # centers based on the identity matrix

    alpha_ = cfg_file["train"]["alpha"] # temperature hyperparameter (alpha)
    beta_type = cfg_file["train"]["beta_type"] # hyperparameter to choose how beta should change
    beta_min = cfg_file["train"]["beta_min"]  # min value for hyperparameter beta
    beta_max = cfg_file["train"]["beta_max"]  # max value for hyperparameter (beta)
    beta_fixed = cfg_file["train"]["beta_fixed"] # fixed value for hyperparameter (beta)
    gamma_ = cfg_file["train"]["gamma"] # hyperparameter for loss of the last layer of the encoder

    type_loss = cfg_file["train"]["type_loss"]
    type_dist = cfg_file["train"]["type_dist"]
    # compute the increment (decrement) of beta by considering the number of epochs and the number of batches
    if beta_type == "up":
        beta_delta = (beta_max - beta_min)/(num_batches * num_epochs)
        beta_ = beta_min
    elif beta_type == "down":
        beta_delta = -1*(beta_max - beta_min)/(num_batches * num_epochs)
        beta_ = beta_max
    elif beta_type == "up_down":
        beta_delta = 2*(beta_max - beta_min)/(num_batches * num_epochs)
        beta_ = beta_min
    elif beta_type == "down_up":
        beta_delta = -2 * (beta_max - beta_min) / (num_batches * num_epochs)
        beta_ = beta_max
    elif beta_type == "fixed":
        beta_ = beta_fixed

    lambda_ = cfg_file["train"]["lambda"] # scalar for regularization

    # create a path for the log directory that includes the dates
    # TODO: include the other hyperparameters for the training
    path_log_dir = cfg_file["model"]["path"] + "log_training_" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    writer = create_writer(path_log_dir)


    print("Starting training on {}...".format(cfg_file["data"]["data_set"]))

    # use Adam optmizer
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = lambda_)

    model.train()
    lap = 1
    phase = 1
    for epoch in range(num_epochs):
        # test the current model
        # epoch = 0 ==> initialization of the model
        if evolution and epoch % epochs_frequency_evolution == 0:
            if save_evolution:
                lap_str = add_zeros(lap, num_epochs // epochs_frequency_evolution)
                path = cfg_file["model"]["evolution_path"] + cfg_file["model"]["name"] + "_lap_" + lap_str
                torch.save(model.state_dict(), path)

            test(cfg_path=cfg_path, model=model, data_set=data_set, mode_forced='train',
                 mode="evolution", lap=lap_str)
            lap += 1

        train_loss = 0

        if p_ref_opt == "random":
            p_ref_1 = np.random.rand()
            p_ref_2 = 1 - p_ref_1
            p_ref = torch.tensor([p_ref_1, p_ref_2])
        else:
            p_ref = torch.tensor(p_ref_opt)

        for batch_idx, data_batch in enumerate(train_loader):
            # get the data and labels from the generator
            x, y = data_batch

            #print("x shape: ", x.shape)
            #print("y shape: ", y.shape)

            x = x.to(device)

            # Resize the input accordingly
            x = x.view(-1, model.input_dim)
            optimizer.zero_grad()

            # Get the latent vector
            #h, dist, h1, h2 = model.encoder(x)

            h = model.encoder(x)
            dist = 0
            h1 = 0
            h2 = 0
            # Get the reconstruction from the auto-encoder
            x_reconstructed = model.decoder(h)

            # Compute the loss of the batch

            if type_loss == "dist":
                loss, loss_rec, loss_dist = loss_function(x, x_reconstructed, h, dist, centers, alpha_, beta_,
                                                        gamma_, type_dist, type_loss, p_ref)

                if batch_idx % batch_frequency_loss == 0:
                    writer.add_scalar('loss_training', loss.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_rec', loss_rec.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_dist', loss_dist.item(), epoch * N + batch_idx)

            elif type_loss == "entropy":
                loss, loss_rec, loss_ent, loss_KL = loss_function(x, x_reconstructed, h, dist, centers, alpha_, beta_,
                                                                  gamma_, type_dist, type_loss, p_ref)

                if batch_idx % batch_frequency_loss == 0:
                    writer.add_scalar('loss_training', loss.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_rec', loss_rec.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_ent', loss_ent.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_KL', loss_KL.item(), epoch * N + batch_idx)

            if beta_type == "up" or beta_type == "down":
                beta_ += beta_delta
            elif beta_type == "up_down" or beta_type == "down_up":
                if epoch < num_epochs // 2:
                    beta_ += beta_delta
                else:
                    beta_ -= beta_delta
            elif beta_type == "fixed":
                beta_ = beta_fixed


            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if verbose:
                if batch_idx % batch_frequency_trace == 0:
                    print(datetime.now(), end = '\t')
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, N,
                               100. * batch_idx / num_batches, loss.item()))




        if verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_batches))

    if save:
        print("Saving model...")
        path = cfg_file["model"]["path"] + cfg_file["model"]["name"]
        torch.save(model.state_dict(), path)


    print("Training DONE!")
    return None