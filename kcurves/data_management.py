# libraries
import numpy as np
from helpers import load_config
from data import SyntheticDataset
from torchvision import datasets, transforms
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
        path_data = cfg_file["data"]["train"]

        train_dataset = datasets.MNIST(root = path_data, train = True,
                                       transform = transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.1306,), (0.3081,))]),
                                       download = True)
        test_dataset = datasets.MNIST(root = path_data, train=False,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((0.1325,), (0.3105,))]),
                                      download = False)


    elif cfg_file["data"]["data_set"] == "synthetic":
        X_train = np.load(cfg_file["data"]["train"]+"X.npy")
        Y_train = np.load(cfg_file["data"]["train"]+"Y.npy")
        X_test =  np.load(cfg_file["data"]["test"]+"X.npy")
        Y_test = np.load(cfg_file["data"]["test"]+ "Y.npy")

        train_dataset = SyntheticDataset(data = X_train, labels = Y_train)
        test_dataset = SyntheticDataset(data = X_test, labels = Y_test)

        if verbose:
            print("Shape X_train: ", X_train.shape)
            print("Shape Y_train: ", Y_train.shape)
            print("Shape X_test: ", X_test.shape)
            print("Shape Y_test: ", Y_test.shape)

    data_set = (train_dataset, test_dataset)

    if verbose:
        print("Dataset loaded!")

    return data_set

def normalize_data(X, verbose=False):
    """
    Function to normalize the data by subtracting the mean and dividing by the variance.
    Arguments:
        X: numpy array with the data.
        verbose: bool variable to print out sanity checks.

    """
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    if verbose:
        print("Function: ", normalize_data.__name__)
        print("Mean: ", mean)
        print("Varaince: ", var)

    X = (X - mean) / var

    return X

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