import dill
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import dense.nn.initializers as initializers
from dense.nn import is_training
from dense.nn.tensor import Tensor

random_stream = RandomStreams(np.random.randint(10000))


class Layer(object):
    def __init__(self, name=None):
        self.name = name
        self.params = list()

    def compute(self, *tensor_args, **tensor_kwargs):
        """
        :param X: an input theano tensor or a list of input theano tensors
        :return: an output theano tensor or a list of output theano tensors
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        :param X: an input tensor or a list of input tensors
        :return: an output tensor or a list of output tensors
        """
        history = set()
        input_args = list()
        input_kwargs = dict()

        for arg in args:
            if isinstance(arg, Tensor):
                input_args.append(arg.tensor)
                history.update(arg.history)
            else:
                input_args.append(arg)

        for kw, arg in kwargs.items():
            if isinstance(arg, Tensor):
                input_kwargs[kw] = arg.tensor
                history.update(arg.history)
            else:
                input_kwargs[kw] = arg

        Y = self.compute(*input_args, **input_kwargs)
        history.add(self)
        if isinstance(Y, list):
            output_tensors = list()
            for y in Y:
                output_tensors.append(Tensor(tensor=y, history=history))
            return output_tensors
        else:
            return Tensor(tensor=Y, history=history)

    def regularization(self):
        return 0.


class Dense(Layer):
    def __init__(self, input_dim, output_dim, activation=None,
                 weight_regularizer=None, bias_regularizer=None,
                 weight_initializer=initializers.he_variance_scaling(), bias_initializer=np.zeros, **kwargs):
        super(Dense, self).__init__(**kwargs)
        weight = weight_initializer((input_dim, output_dim))
        bias = bias_initializer(output_dim)
        self.W = theano.shared(value=weight.astype(theano.config.floatX),
                               name='%s_W' % self.name,
                               borrow=True)
        self.b = theano.shared(value=bias.reshape(1, -1).astype(theano.config.floatX),
                               name='%s_b' % self.name,
                               borrow=True,
                               broadcastable=(True, False))
        self.activation = activation
        self.params.extend([self.W, self.b])
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer

    def compute(self, x):
        y = T.dot(x, self.W) + self.b
        return y if self.activation is None else self.activation(y)

    def regularization(self):
        cost = 0.
        if self.weight_regularizer:
            cost += self.weight_regularizer(self.W)
        if self.bias_regularizer:
            cost += self.bias_regularizer(self.b)
        return cost


class DenseMLP(Layer):
    def __init__(self, dims, activations=None,
                 weight_regularizer=None, bias_regularizer=None,
                 input_dropout=False, output_dropout=False,
                 dropout_mask=None, **kwargs):
        super(DenseMLP, self).__init__(**kwargs)

        self.layers = list()
        if input_dropout and dropout_mask is not None:
            self.layers.append(dropout_mask)
        self.layers.append(Dense(dims[0], dims[1], activation=activations[0],
                                 weight_regularizer=weight_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 name='%s_dense0' % self.name))
        for i in range(1, len(dims) - 1):
            if dropout_mask is not None:
                self.layers.append(dropout_mask)
            input_dim = dims[i]
            output_dim = dims[i + 1]
            activation = activations[i]
            self.layers.append(Dense(input_dim, output_dim, activation=activation,
                                     weight_regularizer=weight_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     name='%s_dense%d' % (self.name, i)))
        if output_dropout and dropout_mask is not None:
            self.layers.append(dropout_mask)

        for layer in self.layers:
            self.params.extend(layer.params)

    def compute(self, x):
        for layer in self.layers:
            x = layer.compute(x)
        return x

    def regularization(self):
        cost = 0.
        for layer in self.layers:
            cost += layer.regularization()
        return cost


class Dropout(Layer):
    def __init__(self, dropout_rate, noise_shape=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.keeping_rate = 1.0 - dropout_rate
        if noise_shape is None:
            self.axes = None
            self.noise_shape = None
        else:
            self.axes = list()
            self.noise_shape = list()
            k = 0
            for i, axis in enumerate(noise_shape):
                if axis is None:
                    self.axes.append(i)
                    self.noise_shape.append(k)
                    k += 1
                elif axis == 1:
                    self.noise_shape.append('x')
                else:
                    raise TypeError('Only \'None\' and 1 in noise_shape')

    def compute(self, x):
        if 0. <= self.keeping_rate < 1.:
            if self.noise_shape is None:
                m = random_stream.binomial(n=1, p=self.keeping_rate, size=x.shape, dtype='floatX') / self.keeping_rate
            else:
                assert x.ndim == len(self.noise_shape)
                shape = tuple([x.shape[i] for i in self.axes])
                m = (random_stream.binomial(n=1, p=self.keeping_rate, size=shape,
                                            dtype='floatX') / self.keeping_rate).dimshuffle(self.noise_shape)
            return T.switch(is_training, x * m, x)
        else:
            return x


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, regularizer=None,
                 initializer=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        if initializer is None:
            embeddings = np.random.normal(size=(input_dim, output_dim))
        else:
            embeddings = initializer((input_dim, output_dim))
        self.E = theano.shared(value=embeddings.astype(theano.config.floatX),
                               name='%s_E' % self.name,
                               borrow=True)
        self.params.append(self.E)
        self.regularizer = regularizer

    def compute(self, idx):
        return self.E[idx, :]

    def regularization(self):
        if self.regularizer:
            return self.regularizer(self.E)
        else:
            return 0.


class Lambda(Layer):
    def __init__(self, function, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.function = function

    def compute(self, *X):
        return self.function(*X)

    def __getstate__(self):
        return dill.dumps(self.function)

    def __setstate__(self, state):
        self.function = dill.loads(state)
