import math

import numpy as np

from batch_generators import BatchGenerator
from kmean import kmean


class Bucket(object):
    def __init__(self, length):
        self.length = length
        self.indexes = list()

    def add(self, idx):
        self.indexes.append(idx)

    def size(self):
        return len(self.indexes)


class Bucketing(BatchGenerator):
    def __init__(self, num_buckets, lengths):
        super(Bucketing, self).__init__()
        self.num_buckets = num_buckets
        self.buckets = self.init_buckets(lengths)

    def init_buckets(self, lengths):
        length2bucket, bucket_lengths = kmean(self.num_buckets, lengths)
        buckets = [Bucket(length) for length in bucket_lengths]
        for idx, length in enumerate(lengths):
            bucket = buckets[length2bucket[length]]
            bucket.add(idx)
        for idx, bucket in enumerate(buckets):
            print 'Bucket %d: %dx%d' % (idx, bucket.size(), bucket.length)
        return buckets

    def get_batch_indexes(self, batch_size, shuffle=False):
        """
        :param batch_size: number of tokens in a batch
        :param shuffle:
        :return:
        """
        batches = list()
        for bucket in self.buckets:
            num_tokens = bucket.size() * bucket.length
            num_splits = min(int(math.ceil(1. * num_tokens / batch_size)), bucket.size())
            indexes = np.asarray(bucket.indexes).astype('int32')
            if shuffle:
                np.random.shuffle(indexes)
            splits = np.array_split(indexes, num_splits)
            batches.extend(splits)
        if shuffle:
            np.random.shuffle(batches)
        return batches
