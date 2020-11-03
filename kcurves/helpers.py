# libraries
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import yaml
from torch.utils.tensorboard import SummaryWriter

def imshow(list_images):
    """
    Plots the original image and the reconstructed image side by side.
    Arguments:
        img_original: original image from the dataset.
        img_reconstructed: reconstructed image, output of the autoencoder.
    """
    num_images = len(list_images)
    fig = plt.figure(figsize=(10, 10))

    for idx_img, pair_image in enumerate(list_images):
        img_original, img_reconstructed = pair_image

        # plot the original image
        ax1 = fig.add_subplot(num_images, 2, 2*idx_img+1)
        plt.imshow(img_original, cmap='gray')
        if idx_img == 0:
            ax1.set_title("Original image")
        ax1.set_xticks([])
        ax1.set_yticks([])
        # plot the reconstructed image
        ax2 = fig.add_subplot(num_images, 2, 2*idx_img + 2)
        plt.imshow(img_reconstructed, cmap='gray')
        if idx_img == 0:
            ax2.set_title("Reconstructed image")
        ax2.set_xticks([])
        ax2.set_yticks([])

    return fig

def plot_X2D_visualization(X_2D, labels, title, num_classes, cluster_centers=None):
    """
    Plots the visualization of the latent space, by separating the classes.
    Arguments:
        X_2D: data in the latent space(2D).
        labels: true labels of the data.
        title: string with the title of the graphic.
        num_classes: Number of classes.
        cluster_centers: Cluster centers, if any, to be plotted.
    """
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # loop to plot the clusters depending on the given labels
    for i in range(num_classes):
        idx = np.where(labels == i)
        ax.scatter(X_2D[idx, 0], X_2D[idx, 1], cmap='viridis')

    # plot the center of the clusters if any
    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, alpha=0.5)

    list_legend = ["class {}".format(i) for i in range(num_classes)]
    ax.legend(list_legend)
    ax.set_title(title)

    return fig

def plot_functions(list_functions, title):
    """
    Plots noisy data and function given a list of functions.
    Arguments:
        list_functions: list with all the functions to be plotted.
        title: title of the graphic.

    """
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    list_legends = []
    for idx, f in enumerate(list_functions):
        ax.plot(f.x, f.y_noisy, f.char_to_plot) # Plot noisy data
        list_legends.append("Noisy data from Function F{}".format(idx+1))
        ax.plot(f.x, f.y, color=f.color_to_plot) # Plot f(x)
        list_legends.append("Function F{}".format(idx+1))

    ax.set_title(title)
    ax.legend(list_legends)

    return fig

def transform_low_to_high(X_low, F_non_linear, list_W):
    """
    Maps the values from a low dimensional space to a higher dimensional space,
    by means of the following formula (applied multiple times):
        X_high = F_non_linear (U * F_non_linear (W * X_low))
    where X_high are the resulting vectors in the high dimensional space and
    X_low are the input vectors from the low dimensional space.
    See below for the other parameters.
    Arguments:
        X_low: data in low dimensional space.
        list_W: List of matrices for the linear transformation to be applied.
        F_non_linear: Non linear function applied in the formula.
    Outputs:
        X_high: data in a higher dimensional space.
    """
    # Apply the transformation to the normalized data
    # by using the matrices stored in the list
    vec_temp = X_low.T
    for W in list_W:
        vec_temp = F_non_linear(W @ vec_temp)

    X_high = vec_temp.T

    return X_high

def run_PCA(X, n_components=2):
    """
    Runs PCA given the data (numpy array).
    Arguments:
        X: data in the in high dimension.
        n_components: Number of components to keep.
     Outputs:
        X_pca: Projection of the data in n_components as a result of PCA.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit(X).transform(X)

    return X_pca

def sigmoid(x):
    """
    Applies the sigmoid function element-wise to a given numpy array.
    Arguments:
        x: numpy array.
    Output:
        : numpy array with the sigmoid function values computed.

    """
    return 1 / (1 + np.exp(-x))

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg

def create_writer(path_log_dir):
    """

    :param path_log_dir:
    :return:
    """

    if not os.path.isdir(path_log_dir):
        os.mkdir(path_log_dir)

    log_tensorboard = path_log_dir + "/tensorboard/"
    writer = SummaryWriter(log_dir=log_tensorboard)

    return writer

def get_regularization_hyperparameters(cfg_path):
    """
    Extracts the regularization hyperparameters: the regularization types and
    the scalar factors for each of those, from the config file.
    :param cfg_path: path for the config file.
    :return:
        - dic_regularization_types: dictionary with boolean values depending on whether the regularization is activated or not.
        - dic_scalar_hyperparameters: dictionary with hyperparameters to scale the regularization terms.
    """

    cfg_file = load_config(cfg_path)

    # types of regularization
    reg_L1 = cfg_file["train"]["reg_L1"]
    reg_KL = cfg_file["train"]["reg_KL"]
    reg_entropy = cfg_file["train"]["reg_entropy"]
    dic_regularization_types = {"reg_L1": reg_L1, "reg_KL": reg_KL, "reg_entropy": reg_entropy}

    # hyperparamters (scalar factors) for the regularization
    lambda_ = cfg_file["train"]["lambda"]
    beta_ = cfg_file["train"]["beta"]
    gamma_ = cfg_file["train"]["gamma"]
    rho_ = cfg_file["train"]["rho"]
    dic_scalar_hyperparameters = {"lambda": lambda_, "beta": beta_, "gamma": gamma_, "rho": rho_}

    return dic_regularization_types, dic_scalar_hyperparameters

def make_string_from_dic(dic):
    """
    Makes a string out of a dictionary, by relating the keys and values with "=" symbol and by
    using a "_" as separator.
        For example:
        dic = {key1: val1, key2: val2, key3: val3 } => str_ = "key1=val1_key2=val2_key3=val3"
    :param dic: dictionary to be applied the transformation.
    :return: str_: string that represents the dictionary given.
    """
    str_ = ""
    for key, value in dic.items():
        str_ += key + "=" +str(value)+"_"

    str_ = str_[:-1] # remove the last character of the string

    return str_
