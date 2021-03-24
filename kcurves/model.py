# libraries
import torch.nn as nn
import torch



class AE(nn.Module):
    def __init__(self, input_dim, encoder_layer_sizes, decoder_layer_sizes, latent_dim,
                 last_nn_layer_encoder, last_nn_layer_decoder, rep_init, rep_type):
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
            rep_init (numpy matrix): matrix with the initial representatives.
        """
        super(AE, self).__init__()

        # the output dim is the same as the input dim
        self.input_dim = input_dim

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim, last_nn_layer_encoder)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim, last_nn_layer_decoder)

        # initialize the representatives
        if rep_type == "points":
            centers_ = torch.tensor(rep_init).type(torch.FloatTensor)
            centers_latent = self.encoder(centers_)
            self.rep = nn.Parameter(centers_latent)
        elif rep_type == "segments":
            s_latent = self.get_rep(rep_init)
            self.set_rep(s_latent)

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

    def get_rep(self, rep):
        s1_ = torch.tensor(rep).type(torch.FloatTensor)[:, :self.input_dim]
        s2_ = torch.tensor(rep).type(torch.FloatTensor)[:, self.input_dim:]
        s1_latent = self.encoder(s1_)
        s2_latent = self.encoder(s2_)
        s_latent = torch.cat((s1_latent, s2_latent), 1)

        return s_latent
    def set_rep (self, rep):
        self.rep = nn.Parameter(rep)



class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim, last_nn_layer_encoder):
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
            #self.hidden.append(nn.ReLU())
            self.hidden.append(InverseSigmoid())

        # Output layer from the encoder
        self.out = nn.Linear(layer_sizes[-1], latent_dim)

        # code to fix the encoder
        # with torch.no_grad():
        #     self.out.weight.copy_(torch.eye(latent_dim))

        self.last_nn_layer_encoder_name = last_nn_layer_encoder

        if last_nn_layer_encoder == 'ReLU':
            self.last_nn_layer_encoder = nn.ReLU()
        elif last_nn_layer_encoder == 'Identity':
            self.last_nn_layer_encoder = nn.Identity()
        elif last_nn_layer_encoder == 'Softmax':
            self.last_nn_layer_encoder = nn.Softmax(dim=1)

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

        return z


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
            #self.hidden.append(nn.ReLU())
            self.hidden.append(nn.Sigmoid())

            # Append the layers in between
            for k in range(len(layer_sizes) - 2):
                self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
                #self.hidden.append(nn.ReLU())
                self.hidden.append(nn.Sigmoid())

            # Append the last layer which considers the last element
            # from the list of layer_sizes, this could be done in the for loop
            # but it's done that way for the sake of neatness
            self.out_linear = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        else:
            self.out_linear = nn.Linear(latent_dim, layer_sizes[-1])

            # code to fix the decoder
            # with torch.no_grad():
            #     self.out_linear.weight.copy_(torch.eye(latent_dim))

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


class InverseSigmoid(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        eps = 1e-6
        x = torch.clamp(x, eps, 1-eps)
        y = -torch.log((1 / x) - 1)

        if torch.isnan(y).any():
            raise ValueError("Output has nan values!!!")

        return y