import theano.tensor as T


def softmax(x):
    if x.ndim == 2:
        return T.nnet.softmax(x)
    elif x.ndim > 2:
        e_x = T.exp(x - x.max(axis=-1, keepdims=True))
        s = e_x.sum(axis=-1, keepdims=True)
        return e_x / s
    else:
        raise Exception('Rank mismatched')


def masking_softmax(x, mask):
    """
    This softmax function ignores some value
    :param x:
    :return:
    """
    x_max = T.max(x, axis=-1, keepdims=True)
    e_x = T.switch(T.neq(mask, 0), T.exp(x - x_max), 0)
    s = e_x.sum(axis=-1, keepdims=True)
    out = T.switch(T.neq(s, 0), e_x / s, 0)
    return out
