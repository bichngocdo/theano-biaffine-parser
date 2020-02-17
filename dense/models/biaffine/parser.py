import cPickle
import sys
import time

import numpy as np
import theano
import theano.tensor as T

import dense.nn.initializers as initializers
import dense.nn.losses as losses
import dense.nn.regularizers as regularizers
import linalg
from data_prepare import prepare_matrix
from dense.nn.activations import masking_softmax
from dense.nn.layers import Layer, Embedding, Dropout, Concatenate, Dense
from dense.nn.layers.lstm import MultiRNN, LSTM
from dense.nn.model import Model
from dense.nn.optimizers import AdamOptimizer, StepDecay
from dense.nn.tensor import Tensor


class HeadClassifier(Layer):
    def __init__(self, input_dim, weight_regularizer=None, **kwargs):
        super(HeadClassifier, self).__init__(**kwargs)
        weight = np.zeros((input_dim + 1, input_dim))
        self.W = theano.shared(value=weight.astype(theano.config.floatX),
                               name='%s_W' % self.name,
                               borrow=True)
        self.params.append(self.W)
        self.weight_regularizer = weight_regularizer

    def compute(self, x_dep, x_head, mask):
        y = linalg.bilinear(x_dep, x_head, self.W, bias1=True, bias2=False)
        y = masking_softmax(y, mask)
        y_pred = T.argmax(y, axis=-1)
        return y, y_pred

    def regularization(self):
        if self.weight_regularizer:
            return self.weight_regularizer(self.W)
        else:
            return 0.


class LabelClassifier(Layer):
    def __init__(self, input_dim, output_dim, weight_regularizer=None, **kwargs):
        super(LabelClassifier, self).__init__(**kwargs)
        weight = np.zeros((output_dim, input_dim + 2, input_dim + 2))
        self.W = theano.shared(value=weight.astype(theano.config.floatX),
                               name='%s_W' % self.name,
                               borrow=True)
        self.params.append(self.W)
        self.weight_regularizer = weight_regularizer

    def compute(self, x_dep, x_head, mask, head_gold, head_pred):
        batch_size, max_length, _ = x_dep.shape
        x_dep = T.concatenate([x_dep, T.ones((batch_size, max_length, 1))], axis=-1)
        x_head = T.concatenate([x_head, T.ones((batch_size, max_length, 1))], axis=-1)
        y = linalg.bilinear(x_dep, x_head, self.W, bias1=True, bias2=True)

        y_gold_dist = y[T.tile(T.arange(batch_size).dimshuffle(0, 'x'), (1, max_length)),
                        T.tile(T.arange(max_length), (batch_size, 1)),
                        head_gold]
        y_gold_dist = masking_softmax(y_gold_dist, mask)

        y_pred_dist = y[T.tile(T.arange(batch_size).dimshuffle(0, 'x'), (1, max_length)),
                        T.tile(T.arange(max_length), (batch_size, 1)),
                        head_pred]
        y_pred_dist = masking_softmax(y_pred_dist, mask)

        y_pred = T.argmax(y_pred_dist, axis=-1)

        return y_gold_dist, y_pred

    def regularization(self):
        if self.weight_regularizer:
            return self.weight_regularizer(self.W)
        else:
            return 0.


class JointMLP(Layer):
    def __init__(self, input_size, output_sizes, depth=1, activation=None,
                 weight_regularizer=None, bias_regularizer=None, dropout_rate=None,
                 input_dropout=False, output_dropout=False, **kwargs):
        super(JointMLP, self).__init__(**kwargs)
        assert depth >= 1
        output_size = 0
        self.split_indices = [0]
        for size in output_sizes:
            output_size += size
            self.split_indices.append(output_size)
        dims = [input_size] + [output_size] * depth
        dropout_mask = Dropout(dropout_rate) if 0 < dropout_rate <= 1 else None

        self.layers = list()
        if input_dropout and dropout_mask is not None:
            self.layers.append(dropout_mask)
        self.layers.append(Dense(dims[0], dims[1], activation=activation,
                                 weight_regularizer=weight_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 name='%s_dense0' % self.name))
        for i in range(1, len(dims) - 1):
            if dropout_mask is not None:
                self.layers.append(dropout_mask)
            input_dim = dims[i]
            output_dim = dims[i + 1]
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
        results = list()
        ndim = x.ndim
        for i in range(len(self.split_indices) - 1):
            idx = [slice(None, None, None)] * (ndim - 1) + [slice(self.split_indices[i], self.split_indices[i + 1])]
            results.append(x[idx])
        return results

    def regularization(self):
        cost = 0.
        for layer in self.layers:
            cost += layer.regularization()
        return cost


class SplitMLP(Layer):
    def __init__(self, input_size, output_size, no_splits=1, depth=1, activation=None,
                 weight_regularizer=None, bias_regularizer=None, dropout_rate=None,
                 input_dropout=False, output_dropout=False, **kwargs):
        super(SplitMLP, self).__init__(**kwargs)
        assert depth >= 1
        assert no_splits >= 1

        dims = [input_size] + [output_size * no_splits] * depth
        dropout_mask = Dropout(dropout_rate, noise_shape=(None, 1, None)) if 0 < dropout_rate <= 1 else None

        self.layers = list()
        if input_dropout and dropout_mask is not None:
            self.layers.append(dropout_mask)
        mat = initializers.orthogonal((input_size, output_size))
        mat = np.concatenate([mat] * no_splits, axis=-1)
        self.layers.append(Dense(dims[0], dims[1], activation=activation,
                                 weight_regularizer=weight_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 weight_initializer=initializers.constant(mat),
                                 name='%s_dense0' % self.name))
        for i in range(1, len(dims) - 1):
            if dropout_mask is not None:
                self.layers.append(dropout_mask)
            input_dim = dims[i]
            output_dim = dims[i + 1]
            mat = initializers.orthogonal((input_size, output_size))
            mat = np.concatenate([mat] * no_splits, axis=-1)
            self.layers.append(Dense(input_dim, output_dim, activation=activation,
                                     weight_regularizer=weight_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     weight_initializer=initializers.constant(mat),
                                     name='%s_dense%d' % (self.name, i)))
        if output_dropout and dropout_mask is not None:
            self.layers.append(dropout_mask)

        for layer in self.layers:
            self.params.extend(layer.params)

        self.no_splits = no_splits
        self.splits = [output_size] * no_splits

    def compute(self, x):
        for layer in self.layers:
            x = layer.compute(x)
        return T.split(x, self.splits, self.no_splits, axis=-1)

    def regularization(self):
        cost = 0.
        for layer in self.layers:
            cost += layer.regularization()
        return cost


class EmbeddingConcatenation(Layer):
    def __init__(self, **kwargs):
        super(EmbeddingConcatenation, self).__init__(**kwargs)

    def compute(self, x):
        pass


class CountTokens(Layer):
    def __init__(self, **kwargs):
        super(CountTokens, self).__init__(**kwargs)

    def compute(self, x):
        return T.sum(T.ge(x, 0))


class CountCorrectTokens(Layer):
    def __init__(self, **kwargs):
        super(CountCorrectTokens, self).__init__(**kwargs)

    def compute(self, x_gold, x_pred):
        return T.sum(T.ge(x_gold, 0) * T.eq(x_gold, x_pred))


class CountCorrectLabels(Layer):
    def __init__(self, **kwargs):
        super(CountCorrectLabels, self).__init__(**kwargs)

    def compute(self, x_gold, x_pred, x_label_gold, x_label_pred):
        correct_head = T.ge(x_gold, 0) * T.eq(x_gold, x_pred)
        correct_label = T.eq(x_label_gold, x_label_pred)
        return T.sum(T.bitwise_and(correct_head, correct_label))


def leaky_relu(x):
    return T.nnet.relu(x, alpha=.1)


class LossLayer(Layer):
    def __init__(self, **kwargs):
        super(LossLayer, self).__init__(**kwargs)

    def compute(self, gold_edge, edge_dist, gold_label, label_dist):
        loss_func = losses.categorical_cross_entropy(ignore_value=-1)
        return loss_func(gold_edge, edge_dist) + loss_func(gold_label, label_dist)


class ComputeMasks(Layer):
    def __init__(self):
        super(ComputeMasks, self).__init__()

    def compute(self, x):
        input_mask = T.ge(x, 0)
        edge_mask = input_mask[:, None, :] * input_mask[:, :, None]
        edge_mask = T.set_subtensor(edge_mask[:, 0], 0)
        label_mask = input_mask[:, :, None]
        label_mask = T.set_subtensor(label_mask[:, 0], 0)
        return input_mask, edge_mask, label_mask


class FixEmbedding(Layer):
    def __init__(self, embeddings, **kwargs):
        super(FixEmbedding, self).__init__(**kwargs)
        embeddings = embeddings/np.std(embeddings)
        self.E = theano.shared(value=embeddings.astype(theano.config.floatX),
                               name='%s_E' % self.name,
                               borrow=True)

    def compute(self, idx):
        return self.E[idx, :]


class DeNSeParser(object):
    def __init__(self, args):
        x_word = Tensor(T.imatrix('x_word'))
        x_pretrain = Tensor(T.imatrix('x_pretrain'))
        x_tag = Tensor(T.imatrix('x_tag'))
        # x_mask = Tensor(T.matrix('x_mask'))
        y_head = Tensor(T.imatrix('y_head'))
        y_label = Tensor(T.imatrix('y_label'))

        input_mask, edge_mask, label_mask = ComputeMasks()(x_word)

        if hasattr(args, 'embeddings'):
            Ew = Embedding(args.no_words, args.word_dim,
                           regularizer=regularizers.l2(args.word_l2),
                           initializer=np.zeros,
                           name='Ew')(x_word)
            Ew_pt = FixEmbedding(args.embeddings,
                                 name='E_pretrained')(x_pretrain)
            word_dim = args.word_dim
            if args.pretrained_mode == 'concat':
                Ew = Concatenate(axis=-1)(Ew, Ew_pt)
                word_dim = args.word_dim + args.embeddings.shape[1]
            elif args.pretrained_mode == 'add':
                assert args.word_dim == args.embeddings.shape[1]
                Ew += Ew_pt
                word_dim = args.word_dim
        else:
            Ew = Embedding(args.no_words, args.word_dim,
                           regularizer=regularizers.l2(args.word_l2),
                           name='Ew')(x_word)
            word_dim = args.word_dim

        Et = Embedding(args.no_tags, args.tag_dim,
                       regularizer=regularizers.l2(args.tag_l2),
                       name='Et')(x_tag)
        Ew = Dropout(args.word_dropout, name='dropout', noise_shape=(None, None, 1))(Ew)
        Et = Dropout(args.tag_dropout, name='dropout', noise_shape=(None, None, 1))(Et)
        x = Concatenate(axis=-1)(Ew, Et)

        input_dim = word_dim + args.tag_dim
        h = MultiRNN(input_dim, args.hidden_dim,
                     cls=LSTM,
                     num_layers=args.num_lstm,
                     bi_directional=True,
                     weight_regularizer=regularizers.l2(args.l2),
                     dropout_rate=args.dropout,
                     recurrent_dropout_rate=args.recurrent_dropout,
                     input_dropout=True, output_dropout=False,
                     weight_initializer=initializers.orthogonal,
                     name='lstm')(x, input_mask)

        dep_mlp, head_mlp = SplitMLP(2 * args.hidden_dim, args.edge_mlp_dim + args.label_mlp_dim,
                                     no_splits=2, depth=args.mlp_depth, activation=leaky_relu,
                                     weight_regularizer=regularizers.l2(args.l2),
                                     dropout_rate=args.dropout,
                                     input_dropout=True, output_dropout=True,
                                     name='mlp')(h)
        dep_edge_mlp, dep_label_mlp = dep_mlp[:, :, :args.edge_mlp_dim], dep_mlp[:, :, args.edge_mlp_dim:]
        head_edge_mlp, head_label_mlp = head_mlp[:, :, :args.edge_mlp_dim], dep_mlp[:, :, args.edge_mlp_dim:]

        y_edge_dist, y_edge_pred = HeadClassifier(args.edge_mlp_dim,
                                                  weight_regularizer=regularizers.l2(args.l2),
                                                  name='head_clf')(dep_edge_mlp, head_edge_mlp, edge_mask)

        y_label_dist, y_label_pred = LabelClassifier(args.label_mlp_dim, args.no_labels,
                                                     weight_regularizer=regularizers.l2(args.l2),
                                                     name='label_clf')(dep_label_mlp, head_label_mlp, label_mask,
                                                                       y_head, y_edge_pred)

        no_tokens = CountTokens()(y_head)
        no_corect_head = CountCorrectTokens()(y_head, y_edge_pred)
        no_corect_label = CountCorrectLabels()(y_head, y_edge_pred, y_label, y_label_pred)

        if hasattr(args, 'embeddings'):
            inputs = [x_word, x_pretrain, x_tag, y_head, y_label]
        else:
            inputs = [x_word, x_tag, y_head, y_label]
        outputs = {
            'no_tokens': no_tokens,
            'no_correct_head': no_corect_head,
            'no_correct_label': no_corect_label,
            'edge_output': y_edge_pred,
            'label_output': y_label_pred,
            'edge_probs': y_edge_dist,
            'label_probs': y_label_dist,
        }
        self.loss = LossLayer()(y_head, y_edge_dist, y_label, y_label_dist)

        self.model = Model(input=inputs, output=outputs)

        self.best_score = 0.
        self.iter = -1

    def compile(self, learning_rate=0.001, decay_rate=.75, decay_step=5000, max_norm=None):
        learning_rate = StepDecay(learning_rate,
                                  decay_rate=decay_rate,
                                  decay_step=decay_step)
        optimizer = AdamOptimizer(learning_rate,
                                  beta2=0.9,
                                  clip_norm=max_norm)
        self.model.compile(self.loss, optimizer=optimizer)

    def train(self, train_data, dev_data, batch_size=10, shuffle=False,
              max_iter=10, save_path='model.pkl',
              eval_iter=5, final_path='best.pkl',
              continue_training=False):
        if len(train_data.data) == 4:
            train_words, train_tags, train_heads, train_labels = train_data.data
            train_x = [train_words, train_tags]
            train_y = [train_heads, train_labels]
        elif len(train_data.data) == 5:
            train_words, train_pretrained_words, train_tags, train_heads, train_labels = train_data.data
            train_x = [train_words, train_pretrained_words, train_tags]
            train_y = [train_heads, train_labels]
        else:
            raise ValueError

        if not continue_training:
            self.iter = 0
            self.best_score = 0.

        epoch = 0
        train_time = 0
        train_loss = 0
        n_train_iters = 0
        n_train_sents = 0
        n_train_tokens = 0
        n_train_correct_head = 0
        n_train_correct_label = 0

        while self.iter < max_iter:
            batch_idx = train_data.get_batches(batch_size, shuffle)
            epoch += 1
            for b, idx in enumerate(batch_idx):
                self.iter += 1

                vars = list()
                for train_xx in train_x:
                    b_x = [train_xx[i] for i in idx]
                    x = prepare_matrix(b_x, root_id=1, none_id=-1)
                    vars.append(x)
                for train_yy in train_y:
                    b_y = [train_yy[i] for i in idx]
                    y = prepare_matrix(b_y, root_id=-1, none_id=-1)
                    vars.append(y)

                start = time.time()
                output = self.model.train_batch(*vars)
                end = time.time()
                loss = output['loss']

                train_time += end - start
                train_loss += loss
                n_train_iters += 1
                n_train_sents += len(vars[0])
                n_train_tokens += output['no_tokens']
                n_train_correct_head += output['no_correct_head']
                n_train_correct_label += output['no_correct_label']

                if self.iter == 1 or self.iter % eval_iter == 0:
                    train_loss /= n_train_iters
                    train_uas = 100. * n_train_correct_head / n_train_tokens
                    train_las = 100. * n_train_correct_label / n_train_tokens
                    train_rate = n_train_sents / train_time
                    print 'Epoch %d, iteration %d:' % (epoch, self.iter)
                    print '   Train: Loss: %10.4f   UAS: %5.2f   LAS: %5.2f   Rate: %6.1f sents/s' % (
                        train_loss, train_uas, train_las, train_rate)
                    sys.stdout.flush()

                    train_time = 0
                    train_loss = 0
                    n_train_iters = 0
                    n_train_sents = 0
                    n_train_tokens = 0
                    n_train_correct_head = 0
                    n_train_correct_label = 0

                    uas, las, _, _ = self.evaluate(dev_data, batch_size=batch_size)
                    if uas > self.best_score:
                        self.best_score = uas
                        with open(final_path, 'wb') as f:
                            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
                            print 'Saved the best model file'
                            sys.stdout.flush()
                    with open(save_path, 'wb') as f:
                        cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)

        if self.iter % eval_iter != 0:
            uas, las, _, _ = self.evaluate(dev_data, batch_size=batch_size)
            if uas > self.best_score:
                self.best_score = uas
                with open(save_path, 'wb') as f:
                    cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
                    print 'Saved the best model file'
                    sys.stdout.flush()

    def evaluate(self, dev_data, batch_size=10):
        if len(dev_data.data) == 4:
            dev_words, dev_tags, dev_heads, dev_labels = dev_data.data
            dev_x = [dev_words, dev_tags]
            dev_y = [dev_heads, dev_labels]
        elif len(dev_data.data) == 5:
            dev_words, dev_pretrained_words, dev_tags, dev_heads, dev_labels = dev_data.data
            dev_x = [dev_words, dev_pretrained_words, dev_tags]
            dev_y = [dev_heads, dev_labels]
        else:
            raise ValueError

        dev_time = 0
        n_dev_sents = 0
        n_dev_tokens = 0
        n_dev_correct_head = 0
        n_dev_correct_label = 0
        result_edge = list()
        result_label = list()
        edge_prob = list()
        label_prob = list()

        batch_idx = dev_data.get_batches(batch_size, shuffle=False)
        for b, idx in enumerate(batch_idx):
            vars = list()
            for dev_xx in dev_x:
                b_x = [dev_xx[i] for i in idx]
                x = prepare_matrix(b_x, root_id=1, none_id=-1)
                vars.append(x)
            for dev_yy in dev_y:
                b_y = [dev_yy[i] for i in idx]
                y = prepare_matrix(b_y, root_id=-1, none_id=-1)
                vars.append(y)

            start = time.time()
            output = self.model.run_batch(*vars)
            pred_edge = output['edge_output']
            pred_label = output['label_output']
            pred_edge_prob = output['edge_probs']
            pred_label_prob = output['label_probs']
            result_edge.extend(np.split(pred_edge, len(pred_edge)))
            result_label.extend(np.split(pred_label, len(pred_label)))
            edge_prob.extend(np.split(pred_edge_prob, len(pred_edge_prob)))
            label_prob.extend(np.split(pred_label_prob, len(pred_label_prob)))
            end = time.time()

            dev_time += end - start
            n_dev_sents += len(vars[0])
            n_dev_tokens += output['no_tokens']
            n_dev_correct_head += output['no_correct_head']
            n_dev_correct_label += output['no_correct_label']

        dev_uas = 100. * n_dev_correct_head / n_dev_tokens
        dev_las = 100. * n_dev_correct_label / n_dev_tokens
        dev_rate = n_dev_sents / dev_time
        print '   Dev:   Loss:        N/A   UAS: %5.2f   LAS: %5.2f   Rate: %6.1f sents/s' % (
            dev_uas, dev_las, dev_rate)
        sys.stdout.flush()

        return dev_uas, dev_las, result_edge, result_label, edge_prob, label_prob

    def parse(self, test_data, batch_size=10):
        if len(test_data.data) == 4:
            test_words, test_tags, test_heads, test_labels = test_data.data
            test_x = [test_words, test_tags]
            test_y = [test_heads, test_labels]
        elif len(test_data.data) == 5:
            test_words, test_pretrained_words, test_tags, test_heads, test_labels = test_data.data
            test_x = [test_words, test_pretrained_words, test_tags]
            test_y = [test_heads, test_labels]
        else:
            raise ValueError

        test_time = 0
        n_test_sents = 0
        n_test_tokens = 0
        result_edge = list()
        result_label = list()
        edge_prob = list()
        label_prob = list()

        batch_idx = test_data.get_batches(batch_size, shuffle=False)
        for b, idx in enumerate(batch_idx):
            vars = list()
            for test_xx in test_x:
                b_x = [test_xx[i] for i in idx]
                x = prepare_matrix(b_x, root_id=1, none_id=-1)
                vars.append(x)
            for test_yy in test_y:
                b_y = [test_yy[i] for i in idx]
                y = prepare_matrix(b_y, root_id=-1, none_id=-1)
                vars.append(y)

            start = time.time()
            output = self.model.run_batch(*vars)
            pred_edge = output['edge_output']
            pred_label = output['label_output']
            pred_edge_prob = output['edge_probs']
            pred_label_prob = output['label_probs']
            result_edge.extend(np.split(pred_edge, len(pred_edge)))
            result_label.extend(np.split(pred_label, len(pred_label)))
            edge_prob.extend(np.split(pred_edge_prob, len(pred_edge_prob)))
            label_prob.extend(np.split(pred_label_prob, len(pred_label_prob)))
            end = time.time()

            test_time += end - start
            n_test_sents += len(vars[0])
            n_test_tokens += output['no_tokens']

        test_rate = n_test_sents / test_time
        print 'Parsed %d sentences, rate: %6.1f sents/s' % (n_test_sents, test_rate)
        sys.stdout.flush()

        return result_edge, result_label, edge_prob, label_prob
