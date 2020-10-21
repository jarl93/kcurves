import os
import argparse
from helpers import transform_low_to_high, load_config, sigmoid, plot_functions, plot_X2D_visualization
from data_management import normalize_data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from functions import FunctionSin
from datetime import datetime
from clustering import k_means

# def plot_synthetic(X, Y, title):
#     """
#     Add documentation
#
#     :param X:
#     :param Y:
#     :param title:
#     :return:
#     """
#     # Plot data according to the labels
#     idx0 = np.where(Y == 0)
#     idx1 = np.where(Y == 1)
#     plt.scatter(X[idx0, 0], X[idx0, 1], cmap='viridis')
#     plt.scatter(X[idx1, 0], X[idx1, 1], cmap='viridis')
#     plt.title(title)
#     plt.show()
#
#     return None

def generate_synthetic(cfg_path, verbose = True):
    """
    Add documentation

    :param cfg_path:
    :return:
    """

    # laod config file
    cfg_file = load_config(cfg_path)

    interval_F1 = tuple(cfg_file["F1"]["interval"])
    interval_F2 = tuple(cfg_file["F2"]["interval"])


    # create a path for the log directory
    path_log_dir = cfg_file["data"]["plots"]["path"] + datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    if not os.path.isdir(path_log_dir):
        os.mkdir(path_log_dir)

    log_tensorboard = path_log_dir + "/tensorboard/"

    writer = SummaryWriter(log_dir = log_tensorboard)

    if verbose:
        print ("Defining functions with parameters from config file: {}".format(cfg_file["name"]))

    # functions for training data
    F1_train = FunctionSin(cfg_file["F1"]["amp"], cfg_file["F1"]["frec"], interval_F1, cfg_file["F1"]["shift"],
                        cfg_file["F1"]["char_to_plot"], cfg_file["F1"]["color_to_plot"])

    F2_train = FunctionSin(cfg_file["F2"]["amp"], cfg_file["F2"]["frec"], interval_F2, cfg_file["F2"]["shift"],
                           cfg_file["F2"]["char_to_plot"], cfg_file["F2"]["color_to_plot"])

    # functions for test data
    F1_test = FunctionSin(cfg_file["F1"]["amp"], cfg_file["F1"]["frec"], interval_F1, cfg_file["F1"]["shift"],
                          cfg_file["F1"]["char_to_plot"], cfg_file["F1"]["color_to_plot"])

    F2_test = FunctionSin(cfg_file["F2"]["amp"], cfg_file["F2"]["frec"], interval_F2, cfg_file["F2"]["shift"],
                          cfg_file["F2"]["char_to_plot"], cfg_file["F2"]["color_to_plot"])


    if verbose:
        print("Generating data...")

    # generation of the training and test data
    F1_train.generate_data(cfg_file["F1"]["train_num_samples"])
    F2_train.generate_data(cfg_file["F1"]["train_num_samples"])
    F1_test.generate_data(cfg_file["F1"]["test_num_samples"])
    F2_test.generate_data(cfg_file["F1"]["test_num_samples"])

    # define lists with the defined functions
    list_F_train = [F1_train, F2_train]
    list_F_test = [F1_test, F2_test]

    if verbose:
        print("Creating non-linear transformation...")

    # apply non-linear transformation to bring lower-dimensional data to a higher-dimensional space
    list_dimensions = []
    for dimension in cfg_file["transformation"]["list_dimensions"]:
        list_dimensions.append(tuple(dimension))

    num_transformations = len(list_dimensions)
    list_W = []

    # loop to store the linear transformations in a list
    for i in range(num_transformations):
        W = np.random.normal(0, 1, size = list_dimensions[i])
        list_W.append(W)


    # TODO: add more non-linear functions if required

    if cfg_file["transformation"]["non_linear"] == "sigmoid":
        F_non_linear = sigmoid

    # TODO: adapt code for list of training functions if required

    # training data
    X_train_low = np.vstack((F1_train.vec, F2_train.vec))
    if cfg_file["data"]["train"]["normalize"]:
        X_train_low_normalized = normalize_data(X_train_low, verbose=True)
        # save the training data normalized
        if cfg_file["data"]["save"]:
            np.save(cfg_file["data"]["train"]["path"] + "X_low_normalized", X_train_low_normalized)
        if verbose:
            print("Applying transformation to normalized training data...")

        X_train = transform_low_to_high(X_train_low_normalized, F_non_linear, list_W)

    else:
        if verbose:
            print("Applying transformation to training data...")
        X_train = transform_low_to_high(X_train_low, F_non_linear, list_W)

    # labels of training data
    labels_train0 = np.zeros(F1_train.vec.shape[0])
    labels_train1 = np.ones(F2_train.vec.shape[0])
    Y_train = np.hstack((labels_train0, labels_train1))

    if verbose:
        print("Transformation of training data completed!")

    # TODO: adapt code for list of test functions if required

    # test data
    X_test_low = np.vstack((F1_test.vec, F2_test.vec))
    if cfg_file["data"]["test"]["normalize"]:
        X_test_low_normalized = normalize_data(X_test_low, verbose=True)

        # save the test data nomalized
        if cfg_file["data"]["save"]:
            np.save(cfg_file["data"]["test"]["path"] + "X_low_normalized", X_test_low_normalized)
        if verbose:
            print("Applying transformation to normalized test data...")

        X_test = transform_low_to_high(X_test_low_normalized, F_non_linear, list_W)
    else:
        if verbose:
            print("Applying transformation to test data...")
        X_test = transform_low_to_high(X_test_low, F_non_linear, list_W)

    # labels of test data
    labels_test0 = np.zeros(F1_test.vec.shape[0])
    labels_test1 = np.ones(F2_test.vec.shape[0])
    Y_test = np.hstack((labels_test0, labels_test1))

    if verbose:
        print("Transformation of test data completed!")

    # run k-means on the training data
    centers_train_low, labels_tain_low = k_means(X_train_low, n_clusters = len(list_F_train))
    centers_train_low_normalized, labels_tain_low_normalized = \
        k_means(X_train_low_normalized, n_clusters = len(list_F_train))

    # run k-means on the test data
    centers_test_low, labels_test_low = k_means(X_test_low, n_clusters = len(list_F_test))
    centers_test_low_normalized, labels_test_normalized = \
        k_means(X_test_low_normalized, n_clusters=len(list_F_test))



    if cfg_file["data"]["save"]:
        if verbose:
            print("Saving data...")
        # save the training data
        np.save(cfg_file["data"]["train"]["path"] + "X_low", X_train_low)
        np.save(cfg_file["data"]["train"]["path"] + "X", X_train)
        np.save(cfg_file["data"]["train"]["path"] + "Y", Y_train)
        # save the test data
        np.save(cfg_file["data"]["test"]["path"] + "X_low", X_test_low)
        np.save(cfg_file["data"]["test"]["path"] + "X", X_test)
        np.save(cfg_file["data"]["test"]["path"] + "Y", Y_test)

    if cfg_file["data"]["plot"]:

        # plot functions and gaussian noisy data
        writer.add_figure('01 Functions and gaussian noisy training data',
                          plot_functions([F1_train, F2_train], "Train data generated"))

        writer.add_figure('02 Functions and gaussian noisy test data',
                          plot_functions([F1_test, F2_test], "Test data generated"))


        # visualize training data in 2D

        title = "k-means on synthetic training data in low-dimensional space"
        writer.add_figure('03 Visualization 2D training data ',
                          plot_X2D_visualization(X_train_low, labels_tain_low,
                                                 title = title, num_classes = len(list_F_train),
                                                 cluster_centers = centers_train_low))

        title = "k-means on synthetic normalized training data in low-dimensional space"
        writer.add_figure('03.1 Visualization 2D normalized training data',
                           plot_X2D_visualization(X_train_low_normalized, labels_tain_low_normalized,
                                                  title = title, num_classes = len(list_F_train),
                                                  cluster_centers = centers_train_low_normalized))

        # visualize test data in 2D
        title = "k-means on synthetic test data in low-dimensional space"
        writer.add_figure('04 Visualization 2D test data ',
                          plot_X2D_visualization(X_test_low, labels_test_low,
                                                 title = title, num_classes = len(list_F_test),
                                                 cluster_centers = centers_test_low))

        title = "k-means on synthetic normalized test data in low-dimensional space"
        writer.add_figure('04.1 Visualization 2D normalized test data',
                          plot_X2D_visualization(X_test_low_normalized, labels_test_normalized,
                                                 title = title, num_classes = len(list_F_test),
                                                 cluster_centers = centers_test_low_normalized))

        # leave the repeated code, because otherwise the last image does not appear.
        # It seems to be a bug from tensorboard, although more investigation is required.
        writer.add_figure('04.1 Visualization 2D normalized test data',
                          plot_X2D_visualization(X_test_low_normalized, labels_test_normalized,
                                                 title = title, num_classes = len(list_F_test),
                                                 cluster_centers = centers_test_low_normalized))


    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")
    args = ap.parse_args()

    generate_synthetic(cfg_path = args.config_path)