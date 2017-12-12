import numpy as np
import theano


def prepare_matrix(sequences, root_id=1, none_id=0, mask=False, max_length=None):
    """
    :param sequences:
    :param root_id:
    :param none_id:
    :param mask:
    :return: matrix with shape (batch_size, max_sequence_length)
    """
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    if max_length is None:
        max_length = max_len
    elif max_length < max_len:
        raise Exception('Not supported: max_length = %d < real maximum length = %d' % (max_length, max_len))
    batch_size = len(sequences)
    if batch_size == 0:
        return None
    if root_id is None:
        x = np.zeros((batch_size, max_length)).astype('int32') + none_id
        x_mask = np.zeros((batch_size, max_length)).astype(theano.config.floatX)
        for batch_idx, sequence in enumerate(sequences):
            x[batch_idx, 0:lengths[batch_idx]] = sequence
            x_mask[batch_idx, 0:lengths[batch_idx]] = 1.
    else:
        x = np.zeros((batch_size, max_length + 1)).astype('int32') + none_id
        x_mask = np.zeros((batch_size, max_length + 1)).astype(theano.config.floatX)
        x[:, 0] = root_id
        x_mask[:, 0] = 1.
        for batch_idx, sequence in enumerate(sequences):
            x[batch_idx, 1:lengths[batch_idx] + 1] = sequence
            x_mask[batch_idx, 1:lengths[batch_idx] + 1] = 1.
    if not mask:
        return x
    else:
        return x, x_mask
