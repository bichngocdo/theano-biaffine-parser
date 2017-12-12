import theano.tensor as T


def binary_accuracy(y_true, y_pred):
    return T.eq(y_true, T.round(y_pred))


def categorical_accuracy(y_true, y_pred):
    if y_true.ndim == y_pred.ndim:
        return T.eq(T.argmax(y_true, axis=-1),
                    T.argmax(y_pred, axis=-1))
    elif y_true.ndim == y_pred.ndim - 1:
        return T.eq(y_true, T.argmax(y_pred, axis=-1))
    else:
        raise Exception('Rank mismatched')


def precision(y_true, y_pred):
    y_pred = T.round(y_pred)
    tp = T.eq(y_true, y_pred) * y_pred
    return (1. * T.sum(tp) / T.sum(y_pred) + 1e-9).dimshuffle('x')


def recall(y_true, y_pred):
    y_pred = T.round(y_pred)
    tp = T.eq(y_true, y_pred) * y_true
    return (1. * T.sum(tp) / T.sum(y_true) + 1e-9).dimshuffle('x')


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)
