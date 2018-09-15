import numpy as np
import pandas as pd
import random
import time

random.seed(time.time())


class CurrencyDataSet(object):
    """
    A Model to represent the time series exchange rate data for a given
    currency pair.
    """

    def __init__(self,
                 raw_currency_df,
                 currency_pair="GBP_CAD",
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1):

        self.currency_pair = currency_pair
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.raw_currency_df = raw_currency_df

        self.raw_seq = np.array(self.raw_currency_df['Rate'].tolist())
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.\
            _prepare_data(self.raw_seq)

    def info(self):
        return "CurrencyDataSet [%s] train: %d test: %d" % (
            self.currency_pair, len(self.train_X), len(self.test_Y))

    def _prepare_data(self, seq):
        # split into items of input_size
        seq = [
            np.array(seq[i * self.input_size: (i + 1) * self.input_size])
            for i in range(len(seq) // self.input_size)
        ]

        seq = [seq[0] / seq[0][0] - 1.0] + [
            curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in
                      range(len(seq) - self.num_steps)])
        Y = np.array([seq[i + self.num_steps] for i in
                      range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_Y, test_Y = Y[:train_size], Y[train_size:]

        return train_X, train_Y, test_X, test_Y

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size

        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = range(num_batches)
        random.shuffle(batch_indices)

        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_Y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}

            yield batch_X, batch_y

    def generate_test_data(self, batch_size):
        last_test_batch_index  = int(len(self.test_X)) - batch_size
        return self.test_X[last_test_batch_index:], \
               self.test_Y[last_test_batch_index:]
