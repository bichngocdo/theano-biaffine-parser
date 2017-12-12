import math

import numpy as np


def glorot_variance_scaling():
    return variance_scaling(scale=2., mode='fan_avg', distribution='uniform')


def he_variance_scaling():
    return variance_scaling(scale=2., mode='fan_in', distribution='normal')


class variance_scaling(object):
    def __init__(self, scale=1.0, mode='fan_in', distribution='uniform'):
        self.scale = scale
        if mode in ['fan_in', 'fan_out', 'fan_avg']:
            self.mode = mode
        else:
            raise TypeError
        if distribution in ['uniform', 'normal']:
            self.distribution = distribution
        else:
            raise TypeError

    def __call__(self, shape):
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        n = 1.
        if self.mode == 'fan_in':
            n = fan_in
        elif self.mode == 'fan_out':
            n = fan_out
        elif self.mode == 'fan_avg':
            n = (fan_in + fan_out) / 2.
        if self.distribution == 'uniform':
            limit = math.sqrt(3. * self.scale / n)
            return np.random.uniform(-limit, limit, shape)
        else:
            std = math.sqrt(1. * self.scale / n)
            return np.random.normal(0., std, shape)


class constant(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        assert shape == self.value.shape
        return self.value


def orthogonal(shape):
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

    # Generate a random matrix
    a = np.random.normal(0., 1., flat_shape)
    # Compute the qr factorization
    q, r = np.linalg.qr(a, mode='reduced')
    # Make Q uniform
    d = np.diag(r)
    ph = d / np.abs(d)
    q *= ph
    if num_rows < num_cols:
        q = np.transpose(q)
    return np.reshape(q, shape)


def orthonormal_dozat(input_size, output_size):
    """
    https://github.com/tdozat/Parser-v1/blob/master/lib/linalg.py#L35
    :param input_size:
    :param output_size:
    :return:
    """
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in xrange(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss) or True:
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return Q.astype(np.float32)
