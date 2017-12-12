import theano.tensor as T


def bilinear(x1, x2, W, bias1=False, bias2=False):
    """

    :param x1: tensor with shape (bxn1xd1)
    :param x2: tensor with shape (bxn2xd2)
    :param W: tensor with shape (d1xd2) or (rxd1xd2)
    :param bias1:
    :param bias2:
    :return: tensor with shape (bxn1xn2) or (bxn1xn2xr)
    """
    if bias1:
        batch_size, max_length, _ = x1.shape
        x1 = T.concatenate([x1, T.ones((batch_size, max_length, 1))], axis=-1)
    if bias2:
        batch_size, max_length, _ = x2.shape
        x2 = T.concatenate([x2, T.ones((batch_size, max_length, 1))], axis=-1)
    result = T.batched_dot(T.dot(x1, W), x2.dimshuffle(0, 2, 1))
    return result if result.ndim == 3 else result.dimshuffle(0, 1, 3, 2)


def bilinear_dozat(x1, x2, W, bias1=False, bias2=False):
    """

    :param x1: tensor with shape (bxn1xd1)
    :param x2: tensor with shape (bxn2xd2)
    :param W: tensor with shape (d1xd2) or (rxd1xd2)
    :param bias1:
    :param bias2:
    :return: tensor with shape (bxn1xn2) or (bxn1xn2xr)
    """
    batch_size, max_length_1, x1_size = x1.shape
    _, max_length_2, x2_size = x2.shape
    output_size = W.shape[0] if W.ndim == 3 else 1
    W = W.dimshuffle(1, 0, 2)
    if bias1:
        x1_size += 1
        x1 = T.concatenate([x1, T.ones((batch_size, max_length_1, 1))], axis=-1)
    if bias2:
        x2_size += 1
        x2 = T.concatenate([x2, T.ones((batch_size, max_length_2, 1))], axis=-1)
    # (b,n1 x d1) (d1 x r,d2) -> (b,n1 x r,d2)
    lin = T.dot(T.reshape(x1, (-1, x1_size)), T.reshape(W, (x1_size, -1)))
    # (b x n1,r x d2) (b x n2 x d2)T -> (b x n1,r x n2)
    bilin = T.batched_dot(
        T.reshape(lin, (batch_size, max_length_1 * output_size, x2_size)), x2.transpose(0, 2, 1))
    # (b,n1 x r x n2)
    bilin = T.reshape(bilin, (-1, output_size, max_length_2))
    # (b x n1 x r x n2)
    bilin = T.reshape(bilin, (batch_size, max_length_1, output_size, max_length_2))
    return bilin.dimshuffle(0, 1, 3, 2)
