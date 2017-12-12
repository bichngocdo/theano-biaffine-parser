import theano.tensor as T

from core import Layer


class _Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(_Concatenate, self).__init__(**kwargs)
        self.axis = axis

    def compute(self, *x):
        return T.concatenate(x, axis=self.axis)


def Concatenate(axis=-1, **kwargs):
    return _Concatenate(axis=axis, **kwargs)
