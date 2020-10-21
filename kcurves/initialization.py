# libraries
from helpers import load_config
from model import AE

def init_model(cfg_path, verbose=False):
    """
    Add documentation
    :param encoder_layer_sizes:
    :param decoder_layer_sizes:
    :param input_dim:
    :param latent_dim:
    :param verbose:
    :return:
    """
    cfg_file = load_config(cfg_path)
    encoder_layer_sizes = cfg_file["model"]["encoder"]["layer_sizes"]
    decoder_layer_sizes = cfg_file["model"]["decoder"]["layer_sizes"]
    input_dim = cfg_file["model"]["input_dim"]
    latent_dim = cfg_file["model"]["latent_dim"]
    last_nn_layer_encoder = cfg_file["model"]["encoder"]["last_nn_layer"]
    device = cfg_file["model"]["device"]

    if verbose:
        print("Initialization of the model...")
    # Define the model as an autoencoder
    model = AE(input_dim=input_dim, encoder_layer_sizes = encoder_layer_sizes,
               decoder_layer_sizes = decoder_layer_sizes, latent_dim = latent_dim,
               last_nn_layer_encoder = last_nn_layer_encoder)

    model = model.to(device)

    if verbose:
        print("Model: ", model)

    return model