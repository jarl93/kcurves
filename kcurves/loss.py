# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from helpers import  pairwise_distances

# As long as PyTorch operations are employed the loss.backward should work

def get_hidden_layers(autoencoder):
    """
    Gets the hidden layers of a given autoencoder.
    :param autoencoder:
    :return: lyers: list with the hidden layers
    """
    # Get the hidden layers from the encoder and the decoder
    layers_encoder = list(autoencoder.encoder.children())
    layers_decoder = list(autoencoder.decoder.children())

    # Most of the layers are embedded in a nn.ModuleList(),
    # but we should consider as well the output layers
    # (linear and non-linear) from the encoder and decoder
    layers = list(layers_encoder[0]) + [layers_encoder[1], layers_encoder[2]] \
             + list(layers_decoder[0]) + [layers_decoder[1], layers_decoder[2]]

    return layers


def L1_regularization(autoencoder, x, lambda_):
    """
    Computes the L1 regularization loss for the autoencoder (sparse autoencoder).
    Arguments:
        autoencoder: auto-encoder in which the training is being run.
        x: input tensor.
        lambda_ : hyperparameter to scale the regularization term.
    Outputs:
        loss_L1: L1 regularization loss for the autoencoder.

    """
    loss_L1 = 0
    layers = get_hidden_layers(autoencoder)

    for layer in layers:
        x = layer(x)
        if isinstance(layer, nn.ReLU): # consider just the activation layers
            loss_L1 += torch.mean(torch.abs(x))  # get the mean of the batch

    # scale by lambda
    loss_L1 *= lambda_

    return loss_L1

def KL_regularization(autoencoder, x, gamma_, rho_scalar):
    """
    Computes the KL divergence regularization loss by means of a Bernoulli distribution.
    The hyperparameter rho is assumed to be the true average of the activation of a
    neuron in each activation layer, rho_hat is the estimated average over the sample batch,
    then, the KL divergence is computed between the two Bernoulli distributions parametrized by
    rho and rho_hat.
    :param autoencoder: auto-encoder in which the training is being run.
    :param x: input of the auto-encoder.
    :param gamma_: hyperparameter to scale the regularization term.
    :param rho_scalar: hyperparameter to define the true Bernoulli distribution rho.
    :return: KL divergence regularization loss.
    """
    layers = get_hidden_layers(autoencoder)
    loss_KL = 0
    for layer in layers:
        # print("Shape x: {}".format(x.shape))
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            # compute the KL divergence
            rho_hat = torch.mean(torch.sigmoid(x), 0)
            rho = rho_scalar * torch.ones_like(rho_hat)
            # print("rho: ", rho)
            # print("rho_hat: ", rho_hat)
            loss_KL += torch.sum(rho*torch.log(rho/rho_hat) + (1-rho)*torch.log((1-rho)/(1-rho_hat)))


    # sys.exit("Stopping script!")

    # scale by gamma
    loss_KL *= gamma_

    return loss_KL


def entropy_regularization(encoder, h, beta_):
    """
    Computes the entropy regularization loss for the last layer of the encoder
    by applying the softmax function to the last layer and then applying the entropy loss.
    Arguments:
        encoder: encoder in which the training is being run.
        h: output of the encoder (latent variable).
        beta_: hyperparameter to scale the regularization term.
    Outputs:
        loss_entropy: entropy regularization loss of the encoder (last layer).

    """
    # compute entropy loss for the last layer of the encoder
    h_ent = None
    if encoder.last_nn_layer_encoder_name == 'Identity':
        h_ent = F.softmax(h, dim = 1) * F.log_softmax(h, dim = 1)
        #TODO: add code for: h_ent = F.softmax(h, dim=1) * h
    elif encoder.last_nn_layer_encoder == 'Softmax':
        h_ent = h * torch.log(h)

    loss_entropy = -1.0 * h_ent.sum()

    rho_hat = F.softmax(h, dim = 1)
    rho = 0.5 * torch.ones_like(rho_hat)
    loss_KL = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    loss_entropy += loss_KL

    # scale by lambda
    loss_entropy *= beta_

    return loss_entropy


def loss_function(x, x_reconstructed, h, autoencoder, scalar_hyperparameters, regularization_types = None):
    """
    Computes the loss_function of the autoencoder.
    Arguments:
        x_reconstructed: reconstructed input.
        x: input tensor.
        h: output of the encoder (latent variable).
        autoencoder: auto-encoder in which the training is being run.
        scalar_hyperparameters: list of hyperparameters to scale the regularization terms, depending on
            the regularization type. The description of the parameters is the following:
            - lambda_: scalar for the  L1 regularization.
            - beta_: scalar for the entropy regularization.
            - gamma_: scalar for the KL divergence regularization.
            - rho_: hyperparamter for the true Bernoulli distribution used for the KL-divergence regularization.
        regularization_types: list of boolean values depending on whether the regularization is activated or not.
            The regularization types are the following:
            - reg_L1: boolean to activate the regularization L1.
            - reg_KL: boolean to activate the regularization KL-divergence.
            - reg_entropy: boolean to activate the entropy regularization.

    Outputs:
        loss_batch: loss of the batch depending on the type of the regularization used.

    """

    lambda_, beta_, gamma_, rho_ = scalar_hyperparameters
    reg_L1, reg_KL, reg_entropy = regularization_types

    # Compute the MSE loss between the input and the reconstruction
    loss_MSE = nn.MSELoss()
    loss_batch = loss_MSE(x, x_reconstructed)

    if regularization_types is not None:
        if reg_L1:
            loss_L1 = L1_regularization(autoencoder, x, lambda_)
            loss_batch += loss_L1
        if reg_KL:
            loss_KL = KL_regularization(autoencoder, x, gamma_, rho_)
            loss_batch += loss_KL
        if reg_entropy:
            loss_entropy = entropy_regularization(autoencoder.encoder, h, beta_)
            loss_batch += loss_entropy

    return loss_batch

def loss_function_clusters(x, x_reconstructed, h, centers, lambda_, alpha_):

    # Compute the MSE loss between the input and the reconstruction
    loss_MSE = nn.MSELoss()
    loss_batch = loss_MSE(x, x_reconstructed)

    # compute the distance matrix between the batch of points and the centers
    dist = pairwise_distances(h, centers)
    # relaxation of the Indicator function
    I_relaxed = F.softmax(-1.0 * alpha_ * dist, dim = 1)
    # compute the loss by multiplying the Indicator function and the distance
    loss_dist = torch.sum(I_relaxed * dist)

    # scale loss_dist with lambda_
    loss_dist *= lambda_

    loss_batch += loss_dist

    return loss_batch


