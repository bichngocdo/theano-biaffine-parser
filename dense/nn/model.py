import sys

import numpy as np
import theano

import dense.nn


def get_batch_indexes(size, batch_size, shuffle=False):
    """
    Use to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(size, dtype='int32')

    if shuffle:
        np.random.shuffle(idx_list)

    batches = []
    batch_start = 0
    for i in range(size // batch_size):
        batches.append(idx_list[batch_start:batch_start + batch_size])
        batch_start += batch_size

    if batch_start != size:
        batches.append(idx_list[batch_start:])

    return batches


def make_list(lst):
    if lst is None:
        return list()
    if not isinstance(lst, list):
        return [lst]
    return lst


class Model(object):
    def __init__(self, input, output):
        """
        :param input: a tensor wrapper or a list of tensor wrappers
        :param output: a tensor wrapper or a list of tensor wrappers
        """
        self.inputs = None
        self.outputs = None
        self.build(input, output)

        self.params = None
        self.regularization = None

        self.cost = None
        self.metric = None
        self.updates = None

        self.predict_function = None

        self.current_epoch = 0
        self.max_epoch = 0
        self.best_score = 0.

        self.is_training = dense.nn.is_training

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['predict_function']
        if self.updates is not None:
            del state['train_function']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.predict_function = None
        if self.updates is not None:
            self.train_function = None
            self.eval_function = None
            self.check_function = None

    def build(self, input, output):
        self.inputs = list()
        self.outputs = list()

        inputs = make_list(input)
        for input in inputs:
            self.inputs.append(input.tensor)

        if isinstance(output, dict):
            self.outputs = dict()
            for name, tensor in output.iteritems():
                self.outputs[name] = tensor.tensor
        else:
            self.outputs = list()
            outputs = make_list(output)
            for output in outputs:
                self.outputs.append(output.tensor)

    def compile(self, loss, optimizer):
        self.params = list()
        self.regularization = 0.

        layers = set()
        layers.update(loss.history)
        for layer in layers:
            self.regularization += layer.regularization()
            self.params.extend(layer.params)

        self.cost = loss.tensor
        self.cost += self.regularization
        self.updates = optimizer.get_updates(self.params, self.cost)

        self.train_function = None

    def __make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise Exception('Model should be compiled before training')
        if self.train_function is None:
            print >> sys.stderr, 'Compile training function'
            input_vars = list()
            input_vars.extend(self.inputs)
            input_vars.append(theano.In(self.is_training, value=1))
            if isinstance(self.outputs, dict):
                output_vars = dict()
                output_vars['loss'] = self.cost
                output_vars.update(self.outputs)
            else:
                output_vars = list()
                output_vars.append(self.cost)
                output_vars.extend(self.outputs)
            self.train_function = theano.function(input_vars, output_vars, updates=self.updates,
                                                  on_unused_input='ignore')
        return self.train_function

    def __make_predict_function(self):
        if self.predict_function is None:
            print >> sys.stderr, 'Compile predict function'
            vars = list()
            vars.extend(self.inputs)
            vars.append(theano.In(self.is_training, value=0))
            self.predict_function = theano.function(vars, self.outputs, on_unused_input='ignore')
        return self.predict_function

    def train_batch(self, *vars):
        """
        Train one batch/iteration and return the loss
        :param vars:
        :return:
        """
        train_function = self.__make_train_function()
        output = train_function(*vars)
        return output

    def run_batch(self, *vars):
        predict_function = self.__make_predict_function()
        output = predict_function(*vars)
        return output
