# libraries

import torch
import torch.optim as optim
import os
from helpers import load_config, create_writer
from loss import loss_function
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
    lambda_ = cfg_file["train"]["lambda"]
    beta_ = cfg_file["train"]["beta"]
    regularization = cfg_file["train"]["regularization"]
    lr = cfg_file["train"]["lr"]
    device =  cfg_file["model"]["device"]
    save = cfg_file["model"]["save"]
    batch_frequency_trace = cfg_file["tracing"]["batch_frequency"]
    batch_frequency_loss = cfg_file["train"]["batch_frequency_loss"]
    batch_size = cfg_file["train"]["batch_size"]

    # create a path for the log directory
    path_log_dir = cfg_file["model"]["path"] + "log_training_" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    writer = create_writer(path_log_dir)


    print("Starting training on {}...".format(cfg_file["data"]["data_set"]))

    # Use Adam optmizer
    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.train()


    # make the generator train_loader for the train_dataset
    train_dataset, _ = data_set
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    N = len(train_dataset)
    num_batches = N // batch_size

    #list_loss = []
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

            # Get the reconstrunction from the autoencoder
            x_reconstructed = model(x)

            h = model.encoder(x)

            # Compute the loss of the batch
            loss = loss_function(x, x_reconstructed, h, model, lambda_, beta_, regularization)

            #list_loss.append(loss)

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