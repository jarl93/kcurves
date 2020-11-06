# libraries

import torch
import torch.optim as optim
import os
from helpers import load_config, create_writer, get_regularization_hyperparameters
from loss import loss_function, loss_function_clusters
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


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
    batch_size = cfg_file["train"]["batch_size"]

    # make the generator train_loader for the train_dataset
    train_dataset, _ = data_set
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    N = len(train_dataset)
    num_batches = N // batch_size

    # get the regularization hyperparameters
    if cfg_file["data"]["data_set"] == "synthetic":
        dic_regularization_types, dic_scalar_hyperparameters = get_regularization_hyperparameters(cfg_path)
        regularization_types = [*dic_regularization_types.values()]
        scalar_hyperparameters = [*dic_scalar_hyperparameters.values()]
    elif cfg_file["data"]["data_set"] == "synthetic_clusters":
        # the dimension of the latent space corresponds to the number of clusters
        dim = cfg_file["model"]["latent_dim"]
        centers = torch.eye(dim) # centers based on the identity matrix

        alpha_min = cfg_file["train"]["alpha_min"] # min value for temperature hyperparameter (alpha)
        alpha_max = cfg_file["train"]["alpha_max"] # max value for temperature hyperparameter (alpha)
        # compute the increment of alpha by considering the number of epochs and the number of batches
        alpha_inc = (alpha_max - alpha_min)/(num_batches * num_epochs)
        alpha_ = alpha_min
        beta_ = cfg_file["train"]["beta"] # scalar for loss of membership and distance to the clusters
        lambda_ = cfg_file["train"]["lambda"] # scalar for L1 regularization

    # create a path for the log directory that includes the dates
    # TODO: include the other hyperparameters for the training
    path_log_dir = cfg_file["model"]["path"] + "log_training_" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    writer = create_writer(path_log_dir)


    print("Starting training on {}...".format(cfg_file["data"]["data_set"]))

    # use Adam optmizer
    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train()

    for epoch in range(num_epochs):
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

            # Get the reconstrunction from the auto-encoder
            x_reconstructed = model(x)

            # Get the latent vector
            h = model.encoder(x)

            # Compute the loss of the batch
            if cfg_file["data"]["data_set"] == "synthetic":
                loss = loss_function(x, x_reconstructed, h, model, scalar_hyperparameters, regularization_types)
            elif cfg_file["data"]["data_set"] == "synthetic_clusters":
                loss = loss_function_clusters(x, x_reconstructed, h, model, centers, alpha_, beta_, lambda_)
                alpha_ += alpha_inc

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if verbose:
                if batch_idx % batch_frequency_trace == 0:
                    print(datetime.now(), end = '\t')
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, N,
                               100. * batch_idx / num_batches, loss.item() / batch_size))

            if batch_idx % batch_frequency_loss == 0:
                writer.add_scalar('training loss', loss.item() / batch_size, epoch * N + batch_idx)

        if verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / N))

    if save:
        print("Saving model...")
        path = cfg_file["model"]["path"] + cfg_file["model"]["name"]
        torch.save(model.state_dict(), path)


    print("Training DONE!")
    # plt.plot(list_loss)
    # plt.title("Loss training")
    # plt.show()
    return None