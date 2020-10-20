from k_curves.helpers import transform_low_to_high
from k_curves.data-management import normalize_data
import numpy as np
import matplotlib.pyplot as plt
from k_curves.functions FunctionSin

def plot_synthetic(X, Y, title):
    """
    Add documentation

    :param X:
    :param Y:
    :param title:
    :return:
    """
    # Plot data according to the labels
    idx0 = np.where(Y == 0)
    idx1 = np.where(Y == 1)
    plt.scatter(X[idx0, 0], X[idx0, 1], cmap='viridis')
    plt.scatter(X[idx1, 0], X[idx1, 1], cmap='viridis')
    plt.title(title)
    plt.show()

    return None

def generate_synthetic(cfg_file):
    """
    Add documentation

    :param cfg_file:
    :return:
    """

    # functions for training data
    F1_train = FunctionSin(cfg_file["F1"]["amp"], cfg_file["F1"]["frec"], cfg_file["F1"]["interval"], cfg_file["F1"]["shift"],
                        cfg_file["F1"]["char_to_plot"], cfg_file["F1"]["color_to_plot"])
    F1_train.generate_data(cfg_file["F1"]["train_num_samples"])

    F2_train = FunctionSin(cfg_file["F2"]["amp"], cfg_file["F2"]["frec"], cfg_file["F2"]["interval"], cfg_file["F2"]["shift"],
                        cfg_file["F2"]["char_to_plot"], cfg_file["F2"]["color_to_plot"])

    F2_train.generate_data(cfg_file["F1"]["train_num_samples"])

    list_functions_train = [F1_train, F2_train]

    # functions for test data
    F1_test = FunctionSin(cfg_file["F1"]["amp"], cfg_file["F1"]["frec"], cfg_file["F1"]["interval"], cfg_file["F1"]["shift"],
                        cfg_file["F1"]["char_to_plot"], cfg_file["F1"]["color_to_plot"])

    F1_test.generate_data(cfg_file["F1"]["test_num_samples"])

    F2_test = FunctionSin(cfg_file["F2"]["amp"], cfg_file["F2"]["frec"], cfg_file["F2"]["interval"], cfg_file["F2"]["shift"],
                        cfg_file["F2"]["char_to_plot"], cfg_file["F2"]["color_to_plot"])
    F2_test.generate_data(cfg_file["F1"]["test_num_samples"])

    list_functions_test = [F1_test, F2_test]

    list_dimensions = cfg_file["transformation"]["list_dimensions"]
    num_transformations = len(list_dimensions)

    list_W = []

    # loop to store the linear transformations in a list
    for i in range(num_transformations):
        W = np.random.normal(0, 1, size=list_dimensions[i])
        list_W.append(W)

    # training data
    X_train_low = np.vstack((F1_train.vec, F2_train.vec))
    if cfg_file["data"]["normalize"]:
        X_train_low_normalized = normalize_data(X_train_low, verbose=True)
        # save the training data normalized
        if cfg_file["data"]["save"]:
            np.save(cfg_file["data"]["train"]["path"], X_train_low_normalized)
        X_train = transform_low_to_high(X_train_low_normalized, cfg_file["transformation"]["non_linear"], list_W)

    else:
        X_train = transform_low_to_high(X_train_low, cfg_file["transformation"]["non_linear"], list_W)

    # labels of training data
    labels_train0 = np.zeros(F1_train.vec.shape[0])
    labels_train1 = np.ones(F2_train.vec.shape[0])
    Y_train = np.hstack((labels_train0, labels_train1))

    # test data
    X_test_low = np.vstack((F1_test.vec, F2_test.vec))
    if cfg_file["data"]["normalize"]:
        X_test_low_normalized = normalize_data(X_test_low, verbose=True)
        X_test = transform_low_to_high(X_test_low_normalized, F_non_linear, list_W)
        # save the test data nomalized
        if cfg_file["data"]["save"]:
            np.save(cfg_file["data"]["test"]["path"], X_test_low_normalized)
    else:
        X_test = transform_low_to_high(X_test_low, cfg_file["transformation"]["non_linear"], list_W)

    # labels of test data
    labels_test0 = np.zeros(F1_test.vec.shape[0])
    labels_test1 = np.ones(F2_test.vec.shape[0])
    Y_test = np.hstack((labels_test0, labels_test1))

    if cfg_file["data"]["save"]:
        # save the training data
        np.save(cfg_file["data"]["train"]["path"], X_train_low)
        np.save(cfg_file["data"]["train"]["path"], X_train)
        np.save(cfg_file["data"]["train"]["path"], Y_train)
        # save the test data
        np.save(cfg_file["data"]["test"]["path"], X_test_low)
        np.save(cfg_file["data"]["test"]["path"], X_test)
        np.save(cfg_file["data"]["test"]["path"], Y_test)

    if cfg_file["data"]["plot"]:
        # plot training data
        print("Plotting training data...")
        plot_synthetic(X_train_low, Y_train, "Synthetic data in low-dimensional space")
        plot_synthetic(X_train_low_normalized, Y_train, "Synthetic normalized data in low-dimensional space")
        # plot test data
        print("Plotting test data...")
        plot_synthetic(X_test_low, Y_test, "Synthetic data in low-dimensional space")
        plot_synthetic(X_test_low_normalized, Y_test,  "Synthetic normalized data in low-dimensional space")

    return X_train, Y_train, X_test, Y_test
