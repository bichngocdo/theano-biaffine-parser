import theano
import theano.tensor as T
from theano.compat import OrderedDict

import dense.utils as utils


def l2_norm(params):
    return T.sqrt(T.sum(map(lambda x: T.sqr(x).sum(), params)))


def clip_norm(grads, max_norm):
    if max_norm is None or max_norm <= 0.:
        return grads
    grad_norm = l2_norm(grads)
    return [T.switch(T.gt(grad_norm, max_norm), grad * max_norm / grad_norm, grad) for grad in grads]


def clip_value(grads, max_value):
    if max_value is None or max_value <= 0.:
        return grads
    return [T.clip(grad, -max_value, max_value) for grad in grads]


class Optimizer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = None

    def get_gradients(self, params, loss):
        grads = T.grad(loss, params)
        if hasattr(self, 'clip_norm'):
            grads = clip_norm(grads, self.clip_norm)
        if hasattr(self, 'clip_value'):
            grads = clip_value(grads, self.clip_value)
        return grads


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super(AdamOptimizer, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def get_updates(self, params, loss):
        grads = self.get_gradients(params, loss)
        self.updates = OrderedDict()

        if isinstance(self.learning_rate, LearningRateDecay):
            lr = self.learning_rate.learning_rate
            self.updates.update(self.learning_rate.get_updates())
        else:
            lr = self.learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        one = T.constant(utils.floatX(1.))
        beta1_t = theano.shared(utils.floatX(beta1), name='beta1_t')
        beta2_t = theano.shared(utils.floatX(beta2), name='beta2_t')
        for param, grad in zip(params, grads):
            momentum = theano.shared(param.get_value() * 0.,
                                     broadcastable=param.broadcastable,
                                     name='momentum')
            velocity = theano.shared(param.get_value() * 0.,
                                     broadcastable=param.broadcastable,
                                     name='velocity')
            m_t = beta1 * momentum + (one - beta1) * grad
            v_t = beta2 * velocity + (one - beta2) * grad ** 2
            m_hat = m_t / (1. - beta1_t)
            v_hat = v_t / (1. - beta2_t)
            step = lr * m_hat / (T.sqrt(v_hat) + eps)
            self.updates[momentum] = m_t
            self.updates[velocity] = v_t
            self.updates[param] = param - step
        self.updates[beta1_t] = beta1_t * beta1
        self.updates[beta2_t] = beta2_t * beta2
        return self.updates


class LearningRateDecay(object):
    def __init__(self, initial_rate, **kwargs):
        self.learning_rate = theano.shared(utils.floatX(initial_rate), name='leaning_rate')
        self.updates = None

    def get_updates(self):
        raise NotImplementedError()


class StepDecay(LearningRateDecay):
    def __init__(self, initial_rate, decay_rate=0.5, decay_step=5000):
        super(StepDecay, self).__init__(initial_rate)
        self.initial_rate = utils.floatX(initial_rate)
        self.decay_rate = utils.floatX(decay_rate)
        self.step = decay_step

    def get_updates(self):
        self.updates = OrderedDict()
        epoch = theano.shared(1, name='epoch_t')
        self.updates[epoch] = epoch + 1
        self.updates[self.learning_rate] = T.cast(self.initial_rate * self.decay_rate ** (epoch / self.step),
                                                  theano.config.floatX)
        return self.updates
