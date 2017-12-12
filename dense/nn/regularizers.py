import theano.tensor as T


class L1L2(object):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, weights):
        return self.l1 * T.sum(T.abs_(weights)) + self.l2 * T.sum(T.sqr(weights))


def l1(l1):
    return L1L2(l1=l1)


def l2(l2):
    return L1L2(l2=l2)


def l1_l2(l1, l2):
    return L1L2(l1, l2)
