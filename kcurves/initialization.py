# libraries

def init_model(encoder_layer_sizes, decoder_layer_sizes, input_dim, latent_dim, verbose=False):
    """
    Add documentation!
    """
    if verbose:
        print("Initialization of the model...")
    # Define the model as an autoencoder
    model = AE(input_dim=input_dim, encoder_layer_sizes=encoder_layer_sizes,
               decoder_layer_sizes=decoder_layer_sizes, latent_dim=latent_dim)

    model = model.to(device)

    if verbose:
        print("Model: ", model)

    return model