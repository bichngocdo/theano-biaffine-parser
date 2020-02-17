import argparse
import cPickle
import os.path
import sys

import numpy as np
import theano
import theano.tensor as T

from batch_generators import SimpleBatch
from bucketing import Bucketing
from data_loader import DataLoader
from dataset import Dataset
from parser import DeNSeParser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args):
    sys.setrecursionlimit(10000)
    data_model = os.path.join(args.model_dir, args.data_model)
    parser_model = os.path.join(args.model_dir, args.classifier_model)
    best_parser_model = os.path.join(args.model_dir, args.best_model)

    if args.seed:
        np.random.seed(args.seed)

    if args.continue_training:
        if os.path.exists(parser_model):
            with open(parser_model, 'rb') as f:
                parser = cPickle.load(f)
        else:
            raise Exception('Parser model file not exists')
        if os.path.exists(data_model):
            with open(data_model, 'rb') as f:
                data_loader = cPickle.load(f)
        else:
            raise Exception('Data model file not exists')

        train_raw_data = data_loader.read_from_file(args.train_file)
        train_data = data_loader.load(train_raw_data)
        train_lengths = [len(item) for item in train_data[0]]
        train_dataset = Dataset(train_data, train_raw_data, Bucketing(args.num_train_buckets, train_lengths))
        print 'Load training data from file %s, no. sentences: %d' % (args.train_file, len(train_data[0]))

        dev_raw_data = data_loader.read_from_file(args.dev_file)
        dev_data = data_loader.load(dev_raw_data)
        dev_lengths = [len(item) for item in dev_data[0]]
        dev_dataset = Dataset(dev_data, dev_raw_data, Bucketing(args.num_dev_buckets, dev_lengths))
        print 'Load development data from file %s, no. sentences: %d' % (args.dev_file, len(dev_data[0]))

        args.no_words = len(data_loader.word_vocab)
        args.no_tags = len(data_loader.tag_vocab)
        args.no_labels = len(data_loader.label_vocab)
        print 'No. words:  %d' % args.no_words
        print 'No. tags:   %d' % args.no_tags
        print 'No. labels: %d' % args.no_labels

        print 'Current epoch:', parser.iter
        print 'Best score:', parser.best_score

        parser.train(train_data=train_dataset,
                     dev_data=dev_dataset,
                     batch_size=args.batch_size,
                     shuffle=args.shuffle,
                     max_iter=args.max_iter,
                     save_path=parser_model,
                     eval_iter=args.eval_iter,
                     final_path=best_parser_model,
                     continue_training=True)
    else:
        data_loader = DataLoader()
        data_loader.cutoff_threshold = args.cutoff
        data_loader.labeled = args.labeled
        data_loader.lowercase = args.lowercase

        train_raw_data = data_loader.read_from_file(args.train_file)
        if args.emb_file:
            # Call this before init_and_load
            args.embeddings = data_loader.read_pretrained_embeddings(args.emb_file)
        train_data = data_loader.init_and_load(train_raw_data)
        train_lengths = [len(item) for item in train_data[0]]
        train_dataset = Dataset(train_data, train_raw_data, Bucketing(args.num_train_buckets, train_lengths))
        print 'Load training data from file %s, no. sentences: %d' % (args.train_file, len(train_data[0]))

        dev_raw_data = data_loader.read_from_file(args.dev_file)
        dev_data = data_loader.load(dev_raw_data)
        dev_lengths = [len(item) for item in dev_data[0]]
        dev_dataset = Dataset(dev_data, dev_raw_data, Bucketing(args.num_dev_buckets, dev_lengths))
        print 'Load development data from file %s, no. sentences: %d' % (args.dev_file, len(dev_data[0]))

        args.no_words = len(data_loader.word_vocab)
        args.no_tags = len(data_loader.tag_vocab)
        args.no_labels = len(data_loader.label_vocab)
        print 'No. words:  %d' % args.no_words
        print 'No. tags:   %d' % args.no_tags
        print 'No. labels: %d' % args.no_labels

        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        print 'Save data model to file ' + data_model
        with open(data_model, 'wb') as f:
            cPickle.dump(data_loader, f)

        parser = DeNSeParser(args)
        parser.compile(learning_rate=args.learning_rate,
                       decay_rate=args.decay_rate,
                       decay_step=args.decay_step,
                       max_norm=args.max_norm)
        parser.train(train_data=train_dataset,
                     dev_data=dev_dataset,
                     batch_size=args.batch_size,
                     shuffle=args.shuffle,
                     max_iter=args.max_iter,
                     save_path=parser_model,
                     eval_iter=args.eval_iter,
                     final_path=best_parser_model,
                     continue_training=False)


def evaluate(args):
    data_model = os.path.join(args.model_dir, args.data_model)
    best_parser_model = os.path.join(args.model_dir, args.best_model)

    if os.path.exists(best_parser_model):
        with open(best_parser_model, 'rb') as f:
            parser = cPickle.load(f)
    else:
        raise Exception('Parser model file not exists')
    if os.path.exists(data_model):
        with open(data_model, 'rb') as f:
            data_loader = cPickle.load(f)
    else:
        raise Exception('Data model file not exists')

    dev_raw_data = data_loader.read_from_file(args.test_file)
    dev_data = data_loader.load(dev_raw_data)
    dev_dataset = Dataset(dev_data, dev_raw_data, SimpleBatch(len(dev_data[0])))
    print 'Load data from file %s, no. sentences: %d' % (args.test_file, len(dev_data[0]))

    _, _, result_heads, result_labels, edge_probs, label_probs = parser.evaluate(dev_data=dev_dataset, batch_size=args.batch_size)
    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            write_result(f, data_loader, dev_raw_data[0], dev_raw_data[1], result_heads, result_labels, edge_probs, label_probs, args.output_probs)


def parse(args):
    data_model = os.path.join(args.model_dir, args.data_model)
    best_parser_model = os.path.join(args.model_dir, args.best_model)

    if os.path.exists(best_parser_model):
        with open(best_parser_model, 'rb') as f:
            parser = cPickle.load(f)
    else:
        raise Exception('Parser model file not exists')
    if os.path.exists(data_model):
        with open(data_model, 'rb') as f:
            data_loader = cPickle.load(f)
    else:
        raise Exception('Data model file not exists')

    with open(args.test_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
        while True:
            raw_data = data_loader.read_from_buffer(f_in, args.buffer_size)
            if len(raw_data[0]) == 0:
                break
            data = data_loader.load(raw_data)
            dataset = Dataset(data, raw_data, SimpleBatch(len(data[0])))
            result_heads, result_labels, edge_probs, label_probs = parser.parse(test_data=dataset, batch_size=args.batch_size)
            write_result(f_out, data_loader, raw_data[0], raw_data[1], result_heads, result_labels,edge_probs,label_probs,args.output_probs)


def write_result(f, data_loader, words, tags, heads, label_ids, edge_probs, label_probs, output_probs):
    for word, tag, head, label_id, edge_prob, label_prob in zip(words, tags, heads, label_ids, edge_probs, label_probs):
        if not output_probs:
            for i in range(len(word)):
                label = data_loader.label_vocab.lookup(label_id[0][i + 1])
                f.write(
                '%d\t%s\t_\t_\t%s\t_\t%d\t%s\t_\t_\n' % ((i + 1), word[i], tag[i], head[0][i + 1], label))
        else:
            for i in range(len(word)):
                label = data_loader.label_vocab.lookup(label_id[0][i + 1])
                f.write(
                '%d\t%s\t_\t_\t%s\t_\t%d\t%s\t%3.3f\t%3.3f\n' % ((i + 1), word[i], tag[i], head[0][i + 1], label, np.max(edge_prob[0][i+1]),np.max(label_prob[0][i+1])))
        f.write('\n')