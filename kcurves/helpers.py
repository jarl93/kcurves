# libraries
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F

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

def plot_X2D_visualization(X, labels, title, num_classes, cluster_centers=None):
    """
    Plots the visualization of the data in 2D. If the dimension of the data is higher
    than 2, then it applies PCA to the data.
    Arguments:
        X: data to be plotted.
        labels: true labels of the data.
        title: string with the title of the graphic.
        num_classes: Number of classes.
        cluster_centers: Cluster centers, if any, to be plotted.
    """
    if X.shape[1] > 2:
        X_2D = run_PCA(X)
    else:
        X_2D = X

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

def plot_2D_visualization_clusters(list_X, list_vars, labels, titles, num_classes, levels_contour, accuracy):
    """
    Visualizes outputs (inputs) of the auto-encoder.
    :param list_X: List [X_input, X_reconstructed, H, softmax(H)], where:
        -X_input is the input data
        -X_reconstructed is the reconstructed data(output of the auto-encoder)
        -H is the latent vector.
        -softmax(H) is the softmax over the latent vector.
    :param list_var: values x_i, y_i and z_ent and z_dist tp plot the objective functions.
    :param labels: true labels of the data.
    :param titles: strings with the titles for each graphic.
    :param num_classes: Number of classes.
    :return: fig: figure with the plots.
    """

    for i in range(len(list_X)):
        if list_X[i].shape[1] > 2:
            print("Running PCA...")
            list_X[i] = run_PCA(list_X[i])

    list_cx = [0, 0, 1, 1, 2, 2]
    list_cy = [0, 1, 0, 1, 0, 1]

    plt.switch_backend('agg')
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    for i in range(len(list_X)):
        cx = list_cx[i]
        cy = list_cy[i]
        X  = list_X[i]
        title = titles[i]
        # loop to plot the clusters depending on the given labels
        for j in range(num_classes):
            idx = np.where(labels == j)
            axs[cx, cy].scatter(X[idx, 0], X[idx, 1], cmap='viridis')
            axs[cx, cy].set_title(title)
        if i == 3:
            axs[cx, cy].legend(accuracy)



    # -----------------------------Frozen code-------------------------------
    # x_i, y_i, z_i_ent, z_i_dist = list_vars
    # # plot graphic for entropy
    # im_20 = axs[2, 0].contourf(x_i ,y_i, z_i_ent, levels = levels_contour)
    # axs[2, 0].set_title(titles[4])
    # fig.colorbar(im_20, ax = axs[2, 0])
    # # plot  graphic for distance
    # im_21 = axs[2, 1].contourf(x_i, y_i, z_i_dist, levels=levels_contour)
    # axs[2, 1].set_title(titles[5])
    # fig.colorbar(im_21, ax=axs[2, 1])

    return fig

def plot_2D_visualization_generation_functions(list_functions, X, labels, num_classes, labels_kmeans,
                                               cluster_centers, titles):
    """
    Visualizes generation of synthetic data via functions (functions and data generated)
    and the result of applying k-means on the synthetic data.
    :param list_functions:
    :param X:
    :param labels: true labels.
    :param num_classes: number of classes (clusters).
    :param labels_kmeans: labels after applying vanilla k-means.
    :param cluster_centers: cluster centers.
    :param titles: titles for each plot.
    :return:
        -fig: Figure with the plots.
    """

    if X.shape[1] > 2:
        X = run_PCA(X)

    plt.switch_backend('agg')
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # make plot for functions
    list_legends = []
    for idx, f in enumerate(list_functions):
        axs[0, 0].plot(f.x, f.y_noisy, f.char_to_plot) # Plot noisy data
        list_legends.append("Noisy data from Function F{}".format(idx+1))
        axs[0, 0].plot(f.x, f.y, color=f.color_to_plot) # Plot f(x)
        list_legends.append("Function F{}".format(idx+1))

    axs[0,0].set_title(titles[0])
    axs[0,0].legend(list_legends)

    # make plot for synthetic data (training or test data)
    for i in range(num_classes):
        idx = np.where(labels == i)
        axs[0, 1].scatter(X[idx, 0], X[idx, 1], cmap='viridis')
        axs[0, 1].set_title(titles[1])

    # make plot for k-means on synthetic data (training or test data)
    for i in range(num_classes):
        idx = np.where(labels_kmeans == i) # plot the clusters depending on the given labels
        axs[1, 0].scatter(X[idx, 0], X[idx, 1], cmap='viridis')

    # plot the center of the clusters if any
    if cluster_centers is not None:
        axs[1, 0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, alpha=0.5)

    list_legend = ["class {}".format(i) for i in range(num_classes)]
    axs[1, 0].legend(list_legend)
    axs[1, 0].set_title(titles[2])

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

def get_softmax(X_2D):
    """
    Computes the softmax of a numpy array or a torch tensor.
    :param X_2D: 2-dimensional array (N, 2).
    :return:
        - X_2D_softmax: numpy array with the softmax of X_2D.
    """

    if type(X_2D) is np.ndarray:
        T_2D = torch.from_numpy(X_2D)
    else:
        T_2D = X_2D

    T_2D_softmax = F.softmax(T_2D, dim = 1)
    X_2D_softmax = T_2D_softmax.detach().numpy()

    return X_2D_softmax


def softmax_pair(x, y, absolute =True, alpha=None):
    """
    Computes the softmax of the pair (x,y).
    :param x: x-coordinate (float).
    :param y: y-coordinate (float).
    :param alpha: temperature parameter (float).
    :return:
        -a: softmax x-coordinate
        -b: softmax y-coordinate
    """
    if absolute:
        x = np.absolute(x)
        y = np.absolute(y)
    if alpha is not None:
        a = np.exp(-alpha * x) / (np.exp(-alpha * x) + np.exp(-alpha * y))
        b = np.exp(-alpha * y) / (np.exp(-alpha * x) + np.exp(-alpha * y))
    else:
        a = np.exp(x) / (np.exp(x) + np.exp(y))
        b = np.exp(y) / (np.exp(x) + np.exp(y))

    return a, b


def entropy_pair(x, y):
    """
    Computes the cross entropy of a pair(x,y)
    :param x: x-coordinate (float).
    :param y: y-coordinate (float).
    :return: entropy of the pair.
    """
    return -x * np.log(x) - y * np.log(y)

def distance_axes_pair(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    a, b = softmax_pair(x, y)
    return a * np.absolute(x) + b * np.absolute(y)

def pairwise_distances(x, y):
    '''
    Arguments: x is a (N,d) tensor
           y is a (M,d) tensor
    Output: dist is a (N,M) tensor where dist[i,j] is the square L2-norm
            between x[i,:] and y[j,:] => dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)

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

def get_hyperparameters(cfg_path):
    """
    Extracts the hyperparameters to make a dictionary out of it.
    :param cfg_path: path for the config file.
    :return:
        - dic_hyperparameters: dictionary with the hyperparameters.
    """

    cfg_file = load_config(cfg_path)

    lr = cfg_file["train"]["lr"]
    num_epochs = cfg_file["train"]["num_epochs"]
    type_dist = cfg_file["train"]["type_dist"]
    type_loss = cfg_file["train"]["type_loss"]
    alpha_ = cfg_file["train"]["alpha"]
    beta_type = cfg_file["train"]["beta_type"]
    beta_fixed = cfg_file["train"]["beta_fixed"]
    beta_min = cfg_file["train"]["beta_min"]
    beta_max = cfg_file["train"]["beta_max"]
    gamma_ = cfg_file["train"]["gamma"]
    lambda_ = cfg_file["train"]["lambda"]

    dic_hyperparameters = {"type_dist": type_dist, "type_loss": type_loss,
                           "beta_type": beta_type, "beta_fixed": beta_fixed,
                           "beta_min": beta_min, "beta_max": beta_max,
                           "gamma": gamma_, "lambda": lambda_,
                           "num_epochs": num_epochs,
                           "alpha": alpha_, "lr": lr}

    return dic_hyperparameters

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

    str_ = str_[:-1] # remove the last '_' from the string

    return str_

def add_zeros (n, lim):
    """
    Adds zeros at the beginning of a number w.r.t to lim.
    :param n: number to add zeros at the beginning.
    :param lim: upper bound for n.
    :return:
        - num: string with the number with zeros at the beginning.
    """
    num = ""
    # put 0's at the beginning w.r.t. lim
    dif = int(np.log10(lim)) - int(np.log10(n))
    for j in range(dif):
        num += "0"
    # add the number given by n
    num += str(n)

    return num

def freeze_module (module):
    """
    :param module:
    :return:
    """
    for param in module.parameters():
        param.requires_grad = False

    return None


def unfreeze_module(module):
    """
    :param module:
    :return:
    """
    for param in module.parameters():
        param.requires_grad = True

    return None