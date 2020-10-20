# libraries
import numpy as np

class Data():
    def __init__(self, X_train, Y_train, X_test, Y_test, X_val=None, Y_val=None, batch_size=128):
        """
        Arguments:
            X_train: training data.
            Y_train: truth training labels.
            X_test: test data.
            Y_test: truth test labels.
            X_val: validation data.
            Y_val: truth validation labels.
            batch_size: size of the batch.

        """
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test
        self.batch_size = batch_size
        self.data = None
        self.labels = None
        self.N = None
        self.num_batches = None
        self.mode = None

    def set_mode(self, mode, shuffle=True):
        """
        Sets the mode of the data, it could be either train, validation or test mode.
        Arguments:
            mode: string variable to indicate whether the training,
                validation or test data is required, mode can take one
                of the following values: {'train', 'validation', 'test'}.
            shuffle: bool variable to indicate if data should be shuffled.
        """
        self.mode = mode

        if mode == 'train':
            self.N = len(self.Y_train)
            if shuffle:
                idx = np.random.permutation(self.N)
            else:
                idx = np.arange(self.N)
            self.data = self.X_train[idx]
            self.labels = self.Y_train[idx]

        elif mode == 'validation':
            self.N = len(self.Y_val)
            if shuffle:
                idx = np.random.permutation(self.N)
            else:
                idx = np.arange(self.N)
            self.data = self.X_val[idx]
            self.labels = self.Y_val[idx]

        elif mode == 'test':
            self.N = len(self.Y_test)
            if shuffle:
                idx = np.random.permutation(self.N)
            else:
                idx = np.arange(self.N)
            self.data = self.X_test[idx]
            self.labels = self.Y_test[idx]

        # Compute the number of batches given the size
        # of the data and the size of the batch
        self.num_batches = self.N // self.batch_size
        if self.N % self.batch_size > 0:
            self.num_batches += 1

        return None

    def get_batch(self):
        """
        Gets batches of a specific size once the mode was set.
        Arguments:
            None.
        Outputs:
            Return a batch from size of batch_size, the batch is a pair (data, labels).
        """
        for idx in range(0, self.N, self.batch_size):
            yield (self.data[idx:min(idx + self.batch_size, self.N)],
                   self.labels[idx:min(idx + self.batch_size, self.N)])