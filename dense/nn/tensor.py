class Tensor(object):
    def __init__(self, tensor, history=None):
        self.tensor = tensor
        self.history = set() if history is None else history.copy()

    def __getitem__(self, item):
        return Tensor(self.tensor.__getitem__(item), self.history)

    def __add__(self, other):
        tensor = Tensor(self.tensor + other.tensor)
        tensor.history.update(self.history)
        tensor.history.update(other.history)
        return tensor

    def __sub__(self, other):
        tensor = Tensor(self.tensor - other.tensor)
        tensor.history.update(self.history)
        tensor.history.update(other.history)
        return tensor

    def __mul__(self, other):
        tensor = Tensor(self.tensor * other.tensor)
        tensor.history.update(self.history)
        tensor.history.update(other.history)
        return tensor

    def __div__(self, other):
        tensor = Tensor(self.tensor / other.tensor)
        tensor.history.update(self.history)
        tensor.history.update(other.history)
        return tensor
