import numpy as np
import theano
import theano.tensor as T

import dense.nn.initializers as initializers
from core import Layer, random_stream
from dense.nn import is_training


class LSTM(Layer):
    def __init__(self, input_dim, hidden_dim, go_backwards=False,
                 weight_regularizer=None, bias_regularizer=None,
                 dropout_rate=0., cell_dropout_rate=0.,
                 weight_initializer=initializers.glorot_variance_scaling(),
                 bias_initializer=np.zeros, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        weights = list()
        for _ in range(4):
            weights.append(weight_initializer((input_dim + hidden_dim, hidden_dim)))
        weight = np.concatenate(weights, axis=-1)
        bias = bias_initializer(4 * hidden_dim)
        self.W = theano.shared(value=weight.astype(theano.config.floatX),
                               name='%s_W' % self.name,
                               borrow=True)
        self.b = theano.shared(value=bias.reshape(1, -1).astype(theano.config.floatX),
                               name='%s_b' % self.name,
                               borrow=True,
                               broadcastable=(True, False))
        self.size = hidden_dim
        self.go_backwards = go_backwards

        self.params = [self.W, self.b]
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer
        self.dropout_rate = dropout_rate
        self.cell_dropout_rate = cell_dropout_rate

    def step(self, x_t, m_t, h_t_prev, c_t_prev, m_h=None, m_c=None):
        def split(matrix, size, no_splits):
            results = list()
            for i in range(no_splits):
                results.append(matrix[:, i * size:(i + 1) * size])
            return results

        if m_h is not None:
            h_t_prev_drop = T.switch(is_training, h_t_prev * m_h, h_t_prev)
        else:
            h_t_prev_drop = h_t_prev
        if m_c is not None:
            c_t_prev_drop = T.switch(is_training, c_t_prev * m_c, c_t_prev)
        else:
            c_t_prev_drop = c_t_prev
        input_t = T.concatenate((x_t, h_t_prev_drop), axis=-1)
        output_t = T.dot(input_t, self.W) + self.b
        splits = split(output_t, self.size, 4)
        f = T.nnet.sigmoid(splits[0])
        i = T.nnet.sigmoid(splits[1])
        c = T.tanh(splits[2])
        o = T.nnet.sigmoid(splits[3])
        c_t = f * c_t_prev_drop + i * c
        c_t = m_t[:, None] * c_t + (1. - m_t)[:, None] * c_t_prev
        h_t = T.tanh(c_t) * o
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_prev
        return h_t, c_t

    def compute(self, x, mask=None):
        ndim = x.ndim
        axes = [1, 0] + list(range(2, ndim))
        x = x.dimshuffle(axes)
        if mask is not None:
            mask_ndim = mask.ndim
            mask_axes = [1, 0] + list(range(2, mask_ndim))
            mask = mask.dimshuffle(mask_axes)

        h0 = T.zeros([x.shape[i] for i in range(1, x.ndim - 1)] + [self.size])
        c0 = T.zeros([x.shape[i] for i in range(1, x.ndim - 1)] + [self.size])
        if mask is None:
            mask = T.ones((x.shape[0], 1))
        if 0. < self.dropout_rate <= 1.:
            keeping_rate = 1 - self.dropout_rate
            m_h = T.cast(random_stream.binomial(n=1, p=keeping_rate, size=h0.shape) / keeping_rate, 'floatX')
        else:
            m_h = T.ones_like(h0)
        if 0. < self.cell_dropout_rate <= 1.:
            keeping_rate = 1 - self.cell_dropout_rate
            m_c = T.cast(random_stream.binomial(n=1, p=keeping_rate, size=c0.shape) / keeping_rate, 'floatX')
        else:
            m_c = T.ones_like(c0)
        if 0. < self.dropout_rate <= 1. or 0. < self.cell_dropout_rate <= 1.:
            [h, c], _ = theano.scan(fn=self.step,
                                    sequences=[x, mask],
                                    non_sequences=[m_h, m_c],
                                    outputs_info=[h0, c0],
                                    go_backwards=self.go_backwards)
        else:
            [h, c], _ = theano.scan(fn=self.step,
                                    sequences=[x, mask],
                                    outputs_info=[h0, c0],
                                    go_backwards=self.go_backwards)

        if self.go_backwards:
            h = h[::-1]
        h = h.dimshuffle(axes)
        return h

    def regularization(self):
        cost = 0.
        if self.weight_regularizer:
            cost += self.weight_regularizer(self.W)
        if self.bias_regularizer:
            cost += self.bias_regularizer(self.b)
        return cost


def RNN(input_dim, hidden_dim, cls=LSTM,
        weight_regularizer=None, bias_regularizer=None, dropout_rate=0.,
        weight_initializer=initializers.glorot_variance_scaling(),
        bias_initializer=np.zeros, **kwargs):
    return cls(input_dim, hidden_dim, go_backwards=False,
               weight_regularizer=weight_regularizer,
               bias_regularizer=bias_regularizer,
               dropout_rate=dropout_rate,
               weight_initializer=weight_initializer,
               bias_initializer=bias_initializer,
               **kwargs)


class BiRNN(Layer):
    def __init__(self, input_dim, hidden_dim, cls=LSTM,
                 weight_regularizer=None, bias_regularizer=None, dropout_rate=0.,
                 weight_initializer=initializers.glorot_variance_scaling(),
                 bias_initializer=np.zeros, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.forward_layer = cls(input_dim, hidden_dim, go_backwards=False,
                                 weight_regularizer=weight_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 dropout_rate=dropout_rate,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 name='%s_f' % self.name)
        self.backward_layer = cls(input_dim, hidden_dim, go_backwards=True,
                                  weight_regularizer=weight_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  dropout_rate=dropout_rate,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='%s_b' % self.name)
        self.params = self.forward_layer.params + self.backward_layer.params

    def compute(self, x, mask=None):
        h_f = self.forward_layer.compute(x, mask)
        h_b = self.backward_layer.compute(x, mask)
        return T.concatenate((h_f, h_b), axis=-1)

    def regularization(self):
        return self.forward_layer.regularization() + self.backward_layer.regularization()


class MultiRNN(Layer):
    def __init__(self, input_dim, hidden_dim, cls=LSTM, num_layers=1, bi_directional=False,
                 weight_regularizer=None, bias_regularizer=None,
                 dropout_rate=0., recurrent_dropout_rate=0.,
                 input_dropout=False, output_dropout=False,
                 weight_initializer=initializers.glorot_variance_scaling(),
                 bias_initializer=np.zeros,
                 **kwargs):
        super(MultiRNN, self).__init__(**kwargs)
        assert num_layers >= 1

        self.layers = list()
        self.params = list()
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout

        inner_input_dim = 2 * hidden_dim if bi_directional else hidden_dim
        wrapper_cls = BiRNN if bi_directional else RNN
        if input_dropout:
            self.layers.append(RecurrentDropout(dropout_rate))
        self.layers.append(wrapper_cls(input_dim, hidden_dim, cls,
                                       weight_regularizer, bias_regularizer, recurrent_dropout_rate,
                                       weight_initializer, bias_initializer,
                                       name='%s_lstm0' % self.name))
        for i in range(num_layers - 1):
            self.layers.append(RecurrentDropout(dropout_rate))
            self.layers.append(wrapper_cls(inner_input_dim, hidden_dim, cls,
                                           weight_regularizer, bias_regularizer, recurrent_dropout_rate,
                                           weight_initializer, bias_initializer,
                                           name='%s_lstm%d' % (self.name, (i + 1))))
        if output_dropout:
            self.layers.append(RecurrentDropout(dropout_rate))

        for layer in self.layers:
            self.params.extend(layer.params)

    def compute(self, x, mask=None):
        rnn_pos = 1 if self.input_dropout else 0
        for i, layer in enumerate(self.layers):
            if i % 2 == rnn_pos:
                x = layer.compute(x, mask)
            else:
                x = layer.compute(x)
        return x

    def regularization(self):
        cost = 0.
        for layer in self.layers:
            cost += layer.regularization()
        return cost


class RecurrentDropout(Layer):
    def __init__(self, dropout_rate, axis=1, **kwargs):
        super(RecurrentDropout, self).__init__(**kwargs)
        self.keeping_rate = 1.0 - dropout_rate
        self.axis = axis

    def compute(self, x):
        if self.axis is not None:
            axes = list(range(x.ndim))
            del axes[self.axis]
            shape = tuple([x.shape[i] for i in axes])
            mask_axes = list(range(x.ndim - 1))
            mask_axes.insert(self.axis, 'x')
            m = (random_stream.binomial(n=1, p=self.keeping_rate, size=shape,
                                        dtype=theano.config.floatX) / self.keeping_rate).dimshuffle(mask_axes)
        else:
            m = random_stream.binomial(n=1, p=self.keeping_rate, size=x.shape, dtype=theano.config.floatX)
        return T.switch(is_training, x * m, x)
