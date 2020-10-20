# libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def imshow(img_original, img_reconstructed):
    """
    Plots the original image and the reconstructed image side by side.
    Arguments:
        img_original: original image from the dataset.
        img_reconstructed: reconstructed image, output of the autoencoder.
    """
    f = plt.figure()

    # plot the original image
    f.add_subplot(1, 2, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("Original image")
    plt.xticks([])
    plt.yticks([])
    # plot the reconstructed image
    f.add_subplot(1, 2, 2)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title("Reconstructed image")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return None

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
    plt.figure(figsize=(10, 10))
    # loop to plot the clusters depending on the given labels
    for i in range(num_classes):
        idx = np.where(labels == i)
        plt.scatter(X_2D[idx, 0], X_2D[idx, 1], cmap='viridis')

    # plot the center of the clusters if any
    if cluster_centers is not None:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, alpha=0.5)

    list_legend = ["class {}".format(i) for i in range(num_classes)]
    plt.legend(list_legend)
    plt.title(title)
    plt.show()
    return None

def plot_functions(list_functions, title):
    """
    Plots noisy data and function given a list of functions.
    Arguments:
        list_functions: list with all the functions to be plotted.
        title: title of the graphic.

    """
    plt.figure(figsize=(15, 10))
    for f in list_functions:
        plt.plot(f.x, f.y_noisy, f.char_to_plot)  # Plot noisy data
        plt.plot(f.x, f.y, color=f.color_to_plot)  # Plot f(x)

    plt.title(title)
    plt.show()

    return None

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




