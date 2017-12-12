import numpy as np
import theano.tensor as T

from activations import softmax, masking_softmax


class categorical_cross_entropy(object):
    def __init__(self, ignore_value=None):
        """
        :param ignore_value: value indicates not to use in the gold distribution
        :return:
        """
        self.ignore_value = ignore_value

    def __call__(self, y_true, y_pred):
        if y_true.ndim == y_pred.ndim:
            y_pred = T.clip(y_pred, 1e-9, 1 - 1e-9)
            if self.ignore_value is not None:
                log_prob = y_true * T.switch(T.neq(y_true, self.ignore_value), T.log(y_pred), 0)
                batch_size = T.sum(y_true * T.neq(y_true, self.ignore_value))
                return T.cast(-T.sum(log_prob) / batch_size, 'floatX')
            else:
                return -T.mean(y_true * T.log(y_pred), axis=-1)
        elif y_true.ndim == y_pred.ndim - 1:
            no_classes = y_pred.shape[-1]
            total_dim = 1
            for d in y_pred.shape[:-1]:
                total_dim *= d
            y_pred = y_pred.reshape((-1, no_classes))
            y_true = y_true.reshape((-1, 1))
            prob = y_pred[T.arange(total_dim).dimshuffle(0, 'x'), y_true]
            prob = T.clip(prob, 1e-9, 1 - 1e-9)
            if self.ignore_value is not None:
                log_prob = T.switch(T.neq(y_true, self.ignore_value), T.log(prob), 0)
                batch_size = T.sum(T.neq(y_true, self.ignore_value))
            else:
                log_prob = T.log(prob)
                batch_size = total_dim
            return T.cast(-T.sum(log_prob) / batch_size, 'floatX')


class softmax_categorical_cross_entropy(object):
    def __init__(self, ignore_value=None):
        """
        :param ignore_value: value indicates not to use in the gold distribution
        :return:
        """
        self.ignore_value = ignore_value

    def __call__(self, y_true, y_pred):
        if y_true.ndim == y_pred.ndim:
            if self.ignore_value is not None:
                mask = T.neq(y_true, self.ignore_value)
                logit = masking_softmax(y_pred, mask)
                logit = T.clip(logit, 1e-9, 1 - 1e-9)
                log_prob = y_true * T.switch(T.neq(y_true, self.ignore_value), T.log(logit), 0)
                batch_size = T.sum(y_true * T.neq(y_true, self.ignore_value))
                return T.cast(-T.sum(log_prob) / batch_size, 'floatX')
            else:
                logit = softmax(y_pred)
                logit = T.clip(logit, 1e-9, 1 - 1e-9)
                return -T.mean(y_true * T.log(logit), axis=-1)
        elif y_true.ndim == y_pred.ndim - 1:
            no_classes = y_pred.shape[-1]
            total_dim = 1
            for d in y_pred.shape[:-1]:
                total_dim *= d
            y_pred = y_pred.reshape((-1, no_classes))
            y_true = y_true.reshape((-1, 1))
            if self.ignore_value is not None:
                mask = T.neq(y_true, self.ignore_value)
                logit = masking_softmax(y_pred, mask)
                prob = logit[T.arange(total_dim).dimshuffle(0, 'x'), y_true]
                prob = T.clip(prob, 1e-9, 1 - 1e-9)
                log_prob = T.switch(T.neq(y_true, self.ignore_value), T.log(prob), 0)
                batch_size = T.sum(T.neq(y_true, self.ignore_value))
            else:
                prob = y_pred[T.arange(total_dim).dimshuffle(0, 'x'), y_true]
                prob = T.clip(prob, 1e-9, 1 - 1e-9)
                log_prob = T.log(prob)
                batch_size = total_dim
            return T.cast(-T.sum(log_prob) / batch_size, 'floatX')


def to_one_hot(idx, no_classes=None, ignore_value=None):
    if no_classes is None:
        no_classes = np.max(idx) + 1
    total_dim = 1
    for dim in idx.shape:
        total_dim *= dim
    one_hot = np.zeros((total_dim, no_classes)).astype('int64')
    idx = idx.reshape(-1, 1)
    one_hot[np.arange(total_dim).reshape(-1, 1), idx] = 1
    if ignore_value is not None:
        one_hot = np.where(np.not_equal(idx, ignore_value), one_hot, ignore_value)
    return one_hot.reshape(idx.shape + (no_classes,))


def binary_crossentropy(y_true, y_pred):
    return T.mean(T.nnet.binary_crossentropy(y_pred, y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return T.mean(T.square(y_pred - y_true), axis=-1)
