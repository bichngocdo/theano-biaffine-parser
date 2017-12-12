from core import Layer


class Reverse(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Reverse, self).__init__(**kwargs)
        self.axis = axis

    def compute(self, x):
        axis = x.ndim - 1 if self.axis == -1 else self.axis
        slices = [slice(None, None, -1) if i == axis else slice(None, None, None) for i in range(x.ndim)]
        return x[slices]
