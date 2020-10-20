# libraries
import numpy as np

def init_data_set(X_train, Y_train, X_test, Y_test, X_val, Y_val, verbose=True):
    """
    Add documentation!
    """
    if verbose:
        print("Loading dataset...")
    # Load the data by defining an instance of the class Data
    data_set = Data(X_train, Y_train, X_test, Y_test)

    if verbose:
        print("Shape X_train: ", X_train.shape)
        print("Shape Y_train: ", Y_train.shape)
        print("Shape X_test: ", X_test.shape)
        print("Shape Y_test: ", Y_test.shape)

    if verbose:
        print("Dataset created!")

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