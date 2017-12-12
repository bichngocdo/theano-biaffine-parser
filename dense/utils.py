import numpy as np
import theano


def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)
