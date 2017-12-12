import math

import numpy as np


class BatchGenerator(object):
    def __init__(self):
        pass

    def get_batch_indexes(self, batch_size, shuffle=False):
        raise NotImplementedError


class SimpleBatch(BatchGenerator):
    def __init__(self, size):
        super(SimpleBatch, self).__init__()
        self.size = size

    def get_batch_indexes(self, batch_size, shuffle=False):
        indexes = np.arange(self.size, dtype='int32')
        if shuffle:
            np.random.shuffle(indexes)
        batches = []
        num_splits = int(math.ceil(1. * self.size / batch_size))
        splits = np.array_split(indexes, num_splits)
        batches.extend(splits)
        return batches
