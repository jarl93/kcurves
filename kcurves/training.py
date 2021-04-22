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
from initialization import PCA_proporcional_dist
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
    epochs_warmup = cfg_file["train"]["epochs_warmup"]
    lr = cfg_file["train"]["lr"]
    device = cfg_file["model"]["device"]
    save = cfg_file["model"]["save"]
    batch_frequency_trace = cfg_file["tracing"]["batch_frequency"]
    batch_frequency_loss = cfg_file["train"]["batch_frequency_loss"]
    evolution = cfg_file["train"]["evolution"]
    epochs_frequency_evolution = cfg_file["train"]["epochs_frequency_evolution"]
    save_evolution = cfg_file["train"]["save_evolution"]
    batch_size = cfg_file["train"]["batch_size"]
    #p_ref_opt = cfg_file["train"]["p_ref"]

    # make the generator train_loader for the train_dataset
    train_dataset, _ = data_set
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    N = len(train_dataset)
    num_batches = N // batch_size

    # TODO: Consider cases when the training hyperparamters are different,
    #  now the training is the same for all data sets

    alpha_init = cfg_file["train"]["alpha_init"] # temperature hyperparameter (alpha)
    alpha_type = cfg_file["train"]["alpha_type"]
    annealing_frequency_change = cfg_file["train"]["annealing_frequency_change"] # frequency to perform annealing
    beta_type = cfg_file["train"]["beta_type"]
    beta_init = cfg_file["train"]["beta_init"]
    gamma_type = cfg_file["train"]["gamma_type"]
    gamma_init = cfg_file["train"]["gamma_init"]
    lambda_ = cfg_file["train"]["lambda"]  # scalar for regularization
    type_loss = cfg_file["train"]["type_loss"]
    type_dist = cfg_file["train"]["type_dist"]

    ####################################### Frozen Code #########################################################
    # beta_min = cfg_file["train"]["beta_min"]  # min value for hyperparameter beta
    # beta_max = cfg_file["train"]["beta_max"]  # max value for hyperparameter (beta)
    # beta_fixed = cfg_file["train"]["beta_fixed"] # fixed value for hyperparameter (beta)
    # compute the increment (decrement) of beta by considering the number of epochs and the number of batches
    # if beta_type == "up":
    #     beta_delta = (beta_max - beta_min)/(num_batches * num_epochs)
    #     beta_ = beta_min
    # elif beta_type == "down":
    #     beta_delta = -1*(beta_max - beta_min)/(num_batches * num_epochs)
    #     beta_ = beta_max
    # elif beta_type == "up_down":
    #     beta_delta = 2*(beta_max - beta_min)/(num_batches * num_epochs)
    #     beta_ = beta_min
    # elif beta_type == "down_up":
    #     beta_delta = -2 * (beta_max - beta_min) / (num_batches * num_epochs)
    #     beta_ = beta_max
    # elif beta_type == "fixed":
    #     beta_ = beta_fixed



    # create a path for the log directory that includes the dates
    # TODO: include the other hyperparameters for the training
    path_log_dir = cfg_file["model"]["path"] + "log_training_" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    writer = create_writer(path_log_dir)


    print("Starting training on {}...".format(cfg_file["data"]["data_set"]))

    # use Adam optmizer
    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train()

    # code to fix the autoencoder
    # model.encoder.out.weight.requires_grad = False
    # model.decoder.out_linear.weight.requires_grad = False
    lap = 1
    alpha_ = alpha_init
    n_alpha = 2
    gamma_ = gamma_init
    n_gamma = 2
    beta_ = beta_init
    n_beta = 2
    for epoch in range(num_epochs):

        # test the current model
        # epoch = 0 ==> initialization of the model
        if evolution and epoch % epochs_frequency_evolution == 0:
            if save_evolution:
                lap_str = add_zeros(lap, num_epochs // epochs_frequency_evolution)
                path = cfg_file["model"]["evolution_path"] + cfg_file["model"]["name"] + "_lap_" + lap_str
                torch.save(model.state_dict(), path)

            _, _, _ = test(cfg_path=cfg_path, model=model, data_set=data_set, mode_forced='train',
                           mode="evolution", lap=lap_str)
            lap += 1


        # anneal hyperparameters following the scheme
        if epoch < epochs_warmup:
            beta_ = 0
            gamma_ = 0
            if epoch + 1 == epochs_warmup:

                rep_latent = np.load(cfg_file["data"]["train"] + "rep_latent.npy")
                rep = torch.tensor(rep_latent).type(torch.FloatTensor).to(device)
                # update the representatives
                model_dict = model.state_dict()
                model_dict['rep'] = rep
                model.load_state_dict(model_dict)

                path = cfg_file["model"]["evolution_path"] + cfg_file["model"]["name"] + "_lap_" + "initialization_centers"
                torch.save(model.state_dict(), path)
                _, _, _ = test(cfg_path=cfg_path, model=model, data_set=data_set, mode_forced='train',
                               mode="evolution", lap="initialization_centers")
                beta_ = beta_init
                gamma_ = gamma_init
        else:
            if (epoch + 1) % annealing_frequency_change == 0:
                if  alpha_type != "fixed":
                    alpha_ = 2 ** (1/np.log(n_alpha) ** 2) * alpha_
                    n_alpha += 1
                if gamma_type != "fixed":
                    gamma_ = 0.5 ** (1/np.log(n_gamma) ** 2) * gamma_
                    n_gamma += 1
                if beta_type != "fixed":
                    beta_ = 1.5 ** (1/np.log(n_beta)**2)  * beta_
                    n_beta += 1

        print("Train epoch = {}, beta = {}, gamma = {}, alpha = {}".format(epoch, beta_, gamma_, alpha_))

        train_loss = 0
        for batch_idx, data_batch in enumerate(train_loader):
            # get the data and labels from the generator
            x, y = data_batch

            #print("x shape: ", x.shape)
            #print("y shape: ", y.shape)

            x = x.to(device)

            # Resize the input accordingly
            x = x.view(-1, model.input_dim)
            optimizer.zero_grad()

            # encode and decode the input x
            h = model.encoder(x)
            x_reconstructed = model.decoder(h)

            # Compute the loss of the batch

            if type_loss == "dist" or type_loss == "log_dist":
                loss, loss_rec, loss_dist = loss_function(model, x, x_reconstructed, h, model.rep, alpha_, beta_,
                                                          gamma_, type_dist, type_loss)

                if batch_idx % batch_frequency_loss == 0:
                    writer.add_scalar('loss_training', loss.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_rec', loss_rec.item(), epoch * N + batch_idx)
                    writer.add_scalar('loss_dist', loss_dist.item(), epoch * N + batch_idx)

            #------------------------------------- frozen code ------------------------------------------------------#
            # elif type_loss == "entropy":
            #     loss, loss_rec, loss_ent, loss_KL = loss_function(x, x_reconstructed, h, dist, centers, alpha_, beta_,
            #                                                       gamma_, type_dist, type_loss, p_ref)
            #
            #     if batch_idx % batch_frequency_loss == 0:
            #         writer.add_scalar('loss_training', loss.item(), epoch * N + batch_idx)
            #         writer.add_scalar('loss_rec', loss_rec.item(), epoch * N + batch_idx)
            #         writer.add_scalar('loss_ent', loss_ent.item(), epoch * N + batch_idx)
            #         writer.add_scalar('loss_KL', loss_KL.item(), epoch * N + batch_idx)
            # if beta_type == "up" or beta_type == "down":
            #     beta_ += beta_delta
            # elif beta_type == "up_down" or beta_type == "down_up":
            #     if epoch < num_epochs // 2:
            #         beta_ += beta_delta
            #     else:
            #         beta_ -= beta_delta
            # elif beta_type == "fixed":
            #     beta_ = beta_fixed


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