from batch_generators import SimpleBatch


class Dataset(object):
    def __init__(self, data, raw_data=None, batch_generator=None):
        self.data = data
        self.raw_data = raw_data
        if not batch_generator:
            size = len(data[0])
            batch_generator = SimpleBatch(size)
        self.batch_generator = batch_generator

    def get_batches(self, batch_size, shuffle=False):
        return self.batch_generator.get_batch_indexes(batch_size, shuffle)
