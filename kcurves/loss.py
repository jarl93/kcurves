# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# As long as PyTorch operations are employed the loss.backward should work

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
    # Get the hidden layers from the encoder and the decoder
    layers_encoder = list(autoencoder.encoder.children())
    layers_decoder = list(autoencoder.decoder.children())

    # Most of the layers are embedded in a nn.ModuleList(), therefore
    # we have to get them from the index 0 in the list of linear layers.
    hidden_layers = list(layers_encoder[0]) + [linear_layers_encoder[1], linear_layers_encoder[1]] \
                    + list(layers_decoder[0])

    for h_layer in hidden_layers:
        x = h_layer(x)
        loss_L1 += torch.mean(torch.abs(x))  # get the mean of the batch

    # scale by lambda
    loss_L1 *= lambda_

    return loss_L1

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
    if encoder.last_nn_layer_encoder == 'Identity':
        h_ent = F.softmax(h, dim = 1) * F.log_softmax(h, dim = 1)
        #TODO: add code for: h_ent = F.softmax(h, dim=1) * h
    elif encoder.last_nn_layer_encoder == 'Softmax':
        h_ent = h * torch.log(h)

    loss_entropy = -1.0 * h_ent.sum()

    # scale by lambda
    loss_entropy *= beta_

    return loss_entropy

def loss_function(x, x_reconstructed, h, autoencoder, lambda_, beta_, regularization=None):
    """
    Computes the loss_function of the autoencoder.
    Arguments:
        x_reconstructed: reconstructed input.
        x: input tensor.
        h: output of the encoder (latent varaible).
        autoencoder: auto-encoder in which the training is being run.
        lambda_: hyper parameter to scale the regularization term for L1 (or KL-divergence) loss.
        beta_ : hyper parameter to scale the regularization term for entropy loss.
        regularization: Type of regularization. It can take the following values:
            'None': Without regularization.
            'L1'  : L1 regularization.
            'KL_divergence': KL divergence regularization. (TODO)
    Outputs:
        loss_batch: loss of the batch depending on the type of the regularization used.

    """
    # Compute the MSE loss between the input and the reconstruction
    loss_MSE = nn.MSELoss()
    loss_batch = loss_MSE(x, x_reconstructed)

    if regularization is not None:
        if regularization == 'L1' or regularization == 'both':
            loss_L1 = L1_regularization(autoencoder, x, lambda_)
            loss_batch += loss_L1
        if regularization == 'KL_divergence':
            pass
        if regularization == 'entropy' or regularization == 'both':
            loss_entropy = entropy_regularization(autoencoder.encoder, h, beta_)
            loss_batch += loss_entropy

    return loss_batch