# libraries
import numpy as np
from helpers import load_config, Read_Two_Column_File, Read_One_Column_File
from data import SyntheticDataset
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
import torch
def load_data_set(cfg_path, verbose = True):
    """
    Add documentation
    :param cfg_path:
    :return:
    """
    # laod config file
    cfg_file = load_config(cfg_path)

    if verbose:
        print("Loading dataset...")

    if cfg_file["data"]["data_set"] == "mnist":

        #path_data = cfg_file["data"]["train"]

        mnist = fetch_openml('mnist_784', version=1, cache=True)

        # scale data
        X = mnist.data / 255.0
        Y = mnist.target.astype(np.int64)

        full_training = cfg_file["data"]["full_training"]
        validation = cfg_file["data"]["validation"]

        if full_training:
            if validation:
                idx_test = Read_One_Column_File('./split/mnist/validation', 'int')
            else:
                idx_test = Read_One_Column_File('./split/mnist/test', 'int')
            X_train = X
            Y_train = Y
        else:
            idx_train = Read_One_Column_File('./split/mnist/train_20', 'int')
            if validation:
                idx_test = Read_One_Column_File('./split/mnist/validation_20', 'int')
            else:
                idx_test = Read_One_Column_File('./split/mnist/test_20', 'int')
            X_train = X[idx_train]
            Y_train = Y[idx_train]

        X_test = X[idx_test]
        Y_test = Y[idx_test]

        train_dataset = SyntheticDataset(data = X_train, labels = Y_train)
        test_dataset = SyntheticDataset(data = X_test, labels = Y_test)

        if verbose:
            print("Shape X_train: ", X_train.shape)
            print("Shape Y_train: ", Y_train.shape)
            print("Shape X_test: ", X_test.shape)
            print("Shape Y_test: ", Y_test.shape)


        # centers
        num_classes = cfg_file["data"]["num_classes"]
        input_dim = cfg_file["model"]["input_dim"]
        centers_train = np.zeros((num_classes, input_dim))
        np.save(cfg_file["data"]["train"]+ "centers_train", centers_train)
        np.save(cfg_file["data"]["test"] + "centers_test", centers_train)


    elif cfg_file["data"]["data_set"] == "synthetic_functions" or \
         cfg_file["data"]["data_set"] == "synthetic_functions" or \
         cfg_file["data"]["data_set"] == "synthetic_lines":

        X_train = np.load(cfg_file["data"]["train"]+"X_train.npy")
        Y_train = np.load(cfg_file["data"]["train"]+"Y_train.npy")
        X_test =  np.load(cfg_file["data"]["test"]+"X_test.npy")
        Y_test = np.load(cfg_file["data"]["test"]+ "Y_test.npy")

        train_dataset = SyntheticDataset(data = X_train, labels = Y_train)
        test_dataset = SyntheticDataset(data = X_test, labels = Y_test)

        if verbose:
            print("Shape X_train: ", X_train.shape)
            print("Shape Y_train: ", Y_train.shape)
            print("Shape X_test: ", X_test.shape)
            print("Shape Y_test: ", Y_test.shape)

    elif cfg_file["data"]["data_set"] == "basic_benchmarking":
        X_train = Read_Two_Column_File(cfg_file["data"]["train_data"], 'float')
        centers_train = Read_Two_Column_File(cfg_file["data"]["train_centers"], 'float')
        X_train, centers_train = normalize_data(X_train, centers_train)
        Y_train = Read_One_Column_File(cfg_file["data"]["train_labels"], 'float')
        # use labels starting at 0
        Y_train = Y_train - 1

        # save the matrices with the training data
        np.save(cfg_file["data"]["train"]+ "centers_train", centers_train)
        np.save(cfg_file["data"]["test"] + "centers_test", centers_train)
        np.save(cfg_file["data"]["train"] + "X_train", X_train)
        np.save(cfg_file["data"]["train"] + "Y_train", Y_train)

        # test and train are the same
        train_dataset = SyntheticDataset(data=X_train, labels=Y_train)
        test_dataset = SyntheticDataset(data=X_train, labels=Y_train)
        if verbose:
            print("Shape X_train: ", X_train.shape)
            print("Shape Y_train: ", Y_train.shape)




    data_set = (train_dataset, test_dataset)

    if verbose:
        print("Dataset loaded!")

    return data_set

def normalize_data(X, centers = None, verbose=False):
    """
    Normalize the data by subtracting the mean and dividing by the variance.
    Arguments:
        X: numpy array with the data.
        verbose: bool variable to print out sanity checks.

    Output: X normalized
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    if verbose:
        print("Function: ", normalize_data.__name__)
        print("Mean: ", mean)
        print("Std: ", std)

    X = (X - mean) / std
    if centers is not None:
        centers = (centers - mean) / std
        return X, centers
    else:
        return X
def scale_data(X, scale_factor, verbose = False):
    """
    Scales data, such that all the points are in the square bottom_left = (-1,-1), upper_right = (1,1).
    :param X: numpy array with the data.
    :param scale_factor: factor to scale all the values.
    :param verbose: boolean varaible to print sanity checks.
    :return: X_scaled: numpy array with the data scaled.
    """
    X_abs = np.absolute(X)
    x_max = np.max(X_abs)
    X_scaled = scale_factor * X / x_max

    return X_scaled

def split_data_loader(data_loader):
    """
    Splits the data loader into data (X) and labels(Y).
    Arguments:
        data_loader: data loader.
    Outputs:
        X: data.
        Y: true labels.
    """
    X = None  # array to store data
    Y = None  # array to store labels
    for batch_idx, data in enumerate(data_loader):
        x, y = data
        x = x.reshape(x.shape[0], -1)
        if batch_idx == 0:
            X = x
            Y = y
        else:
            X = np.vstack((X, x))
            Y = np.hstack((Y, y))

    return X, Y