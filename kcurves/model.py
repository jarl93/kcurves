# libraries
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class AE(nn.Module):
    def __init__(self, input_dim, encoder_layer_sizes, decoder_layer_sizes, latent_dim,
                 last_nn_layer_encoder, last_nn_layer_decoder, alpha_):
        """
        Arguments:
            input_dim (int): dimension of the input.
            encoder_layer_sizes (list[int]): list with the sizes of the encoder layers.
            decoder_layer_sizes (list[int]): list with the sizes of the decoder layers.
            latent_dim (int): dimension of latent space/bottleneck.
            last_nn_layer_encoder (string): last non-linear layer of the encoder,
                                            the output will be the latent variable.
            last_nn_layer_decoder (string): last non-linear layer of the decoder,
                                            the output will be the reconstruction.
        """
        super(AE, self).__init__()

        # the output dim is the same as the input dim
        self.input_dim = input_dim

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim, last_nn_layer_encoder, alpha_)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim, last_nn_layer_decoder)

    def forward(self, x):
        """
        Forward Process of the whole AE.
        Arguments:
            x: tensor of dimension (batch_size, input_dim).
        Outputs:
            x_reconstructed: reconstructed input (output of the auto-encoder).
                Same dimension as x (batch_size, input_dim).
        """

        # Map the input to the latent space (encoding)
        z = self.encoder(x)

        # Map the latent variable to the input/output space (decoding),
        # i.e., get the reconstruction from the latent space
        x_reconstructed = self.decoder(z)

        return x_reconstructed


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim, last_nn_layer_encoder, alpha_):
        super(Encoder, self).__init__()
        """
        Arguments:
            layer_sizes (list[int]): list of sizes of the linear layers of the encoder.
            latent_dim (int): dimension of latent space, i.e. dimension out output of the encoder.
            last_nn_layer_encoder (string): last non-linear layer of the encoder, 
                                            the output will be the latent variable.
        """

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
            #self.hidden.append(nn.LeakyReLU())
            self.hidden.append(nn.ReLU())

        # Output layer from the encoder
        self.out = nn.Linear(layer_sizes[-1], latent_dim)

        self.last_nn_layer_encoder_name = last_nn_layer_encoder

        if last_nn_layer_encoder == 'ReLU':
            self.last_nn_layer_encoder = nn.ReLU()
        elif last_nn_layer_encoder == 'Identity':
            self.last_nn_layer_encoder = nn.Identity()
        elif last_nn_layer_encoder == 'Softmax':
            self.last_nn_layer_encoder = nn.Softmax(dim=1)

        #  heads for the encoder
        # self.head_encoder_1 = HeadEncoder(np.pi / 4)
        # self.head_encoder_2 = HeadEncoder(7 * np.pi / 4)
        # self.alpha_ = alpha_

    def forward(self, x):
        """
        Makes the forward for the Encoder.
        Arguments:
            x: input of the auto-encoder, tensor of dimension (batch_size, 28*28).

        Outputs:
            z: latent variable, tensor of dimension (batch_size, latent_dim).
        """

        # Do the forward for the hidden layers
        for layer in self.hidden:
            x = layer(x)


        # Do the forward for the output layer
        x = self.out(x)

        # Do the forward for the last non-linear layer
        z = self.last_nn_layer_encoder(x)


        # joining the output of the heads
        # z1 = self.head_encoder_1(z)
        # z2 = self.head_encoder_2(z)
        #
        # d1 = -torch.abs(z1)[:, 0] + torch.abs(z1)[:, 1]
        # d2 = torch.abs(z2)[:, 0] - torch.abs(z2)[:, 1]
        # dist = torch.cat((d1.view(-1, 1), d2.view(-1, 1)), 1)
        # p = F.softmax(self.alpha_ * dist, dim=1)
        # p1_ = p[:, 0][:, None]
        # p2_ = p[:, 1][:, None]
        # p1 = torch.cat((p1_, p1_), 1)
        # p2 =  torch.cat((p2_, p2_), 1)
        # z = p1*z1 + p2*z2
        #
        # return z, dist, z1, z2

        return z

class HeadEncoder(nn.Module):

    def __init__(self, init_angle):
        """

        :param init_angle:
        """
        super(HeadEncoder, self).__init__()
        # angle and vectors for the rotation and translation
        self.theta = nn.Parameter(torch.tensor(init_angle))
        self.v_in = nn.Parameter(torch.rand(2))
        self.v_out = nn.Parameter(torch.rand(2))

    def forward(self, z):
        """

        :param z:
        :return:
        """
        # rotation matrix
        r = torch.zeros(2, 2)
        r[0, 0] = torch.cos(self.theta)
        r[0, 1] = torch.sin(self.theta)
        r[1, 0] = -1*torch.sin(self.theta)
        r[1, 1] = torch.cos(self.theta)
        # apply transformations
        v_in  = self.v_in.repeat(z.shape[0]).view(-1, z.shape[1])
        v_out = self.v_out.repeat(z.shape[0]).view(-1, z.shape[1])
        z_t = torch.matmul(v_in + z, r) + v_out

        return z_t

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim, last_nn_layer_decoder):
        super(Decoder, self).__init__()
        """
        Arguments:
            layer_sizes (list[int]): list of sizes of the linear layers of the decoder.
            latent_dim (int): dimension of latent space, i.e. dimension out input of the decoder.
            last_nn_layer_decoder (string): last non-linear layer of the decoder,
                                            the output will be the reconstructed input.

        """

        # Hidden layers
        self.hidden = nn.ModuleList()

        if len(layer_sizes) > 1:
            # Append the first layer of the decoder
            self.hidden.append(nn.Linear(latent_dim, layer_sizes[0]))
            self.hidden.append(nn.ReLU())
            #self.hidden.append(nn.LeakyReLU())

            # Append the layers in between
            for k in range(len(layer_sizes) - 2):
                self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
                self.hidden.append(nn.ReLU())
                #self.hidden.append(nn.LeakyReLU())

            # Append the last layer which considers the last element
            # from the list of layer_sizes, this could be done in the for loop
            # but it's done that way for the sake of neatness
            self.out_linear = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        else:
            self.out_linear = nn.Linear(latent_dim, layer_sizes[-1])

        if last_nn_layer_decoder == 'ReLU':
            self.out_non_linear = nn.ReLU()
        elif  last_nn_layer_decoder == 'Identity':
            self.out_non_linear = nn.Identity()
        elif last_nn_layer_decoder == 'Softmax':
            self.out_non_linear = nn.Softmax(dim=1)

    def forward(self, z):
        """
        Makes the forward for the Decoder.
        Arguments:
            z: tensor of dimension (batch_size, latent_dim).
        Outputs:
            x: reconstructed input from the latent space of dimension (batch_size, input_dim).
        """
        # Do the feedforward for the hidden layers
        x = z
        for layer in self.hidden:
            x = layer(x)

        # Do the forward for the output layer
        # to get the reconstruction

        x = self.out_non_linear(self.out_linear(x))

        return x