import argparse
import sys

from main import train, str2bool


def main(sys_args):
    parser = argparse.ArgumentParser(description='Training helper for scf frame classifier.')

    parser.add_argument('--train_file', type=str, required=True,
                        help='training data')
    parser.add_argument('--dev_file', type=str, required=True,
                        help='development data')
    parser.add_argument('--emb_file', type=str, default=None,
                        help='pre-trained embedding file')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='folder to save the model')
    parser.add_argument('--data_model', type=str, default='data.pkl',
                        help='data model file')
    parser.add_argument('--classifier_model', type=str, default='model.pkl',
                        help='classifier model file')
    parser.add_argument('--best_model', type=str, default='best.pkl',
                        help='best classifier model file')

    parser.add_argument('--cutoff', type=int, default=2,
                        help='word cutoff threshold')
    parser.add_argument('--labeled', type=str2bool, default=True,
                        help='labeled parsing')
    parser.add_argument('--lowercase', type=str2bool, default=True,
                        help='lowercase words')
    parser.add_argument('--num_train_buckets', type=int, default=40,
                        help='number of length buckets')
    parser.add_argument('--num_dev_buckets', type=int, default=10,
                        help='number of length buckets')
    parser.add_argument('--pretrained_mode', type=str, default='add',
                        help='pre-trained mode')

    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--continue_training', type=str2bool, default=False,
                        help='continue training')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.75,
                        help='decay rate')
    parser.add_argument('--decay_step', type=int, default=5000,
                        help='decay step')
    parser.add_argument('--shuffle', type=str2bool, default=True,
                        help='shuffle data while training')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='mini-batch size in number of tokens')
    parser.add_argument('--max_iter', type=int, default=50000,
                        help='number of training iterations')
    parser.add_argument('--eval_iter', type=int, default=100,
                        help='iteration interval to evaluate')

    parser.add_argument('--word_dim', type=int, default=100,
                        help='word embedding dimension')
    parser.add_argument('--tag_dim', type=int, default=100,
                        help='tag embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=400,
                        help='LSTM hidden state dimension')
    parser.add_argument('--num_lstm', type=int, default=3,
                        help='number of LSTM layers')
    parser.add_argument('--edge_mlp_dim', type=int, default=500,
                        help='edge MLP dimension')
    parser.add_argument('--label_mlp_dim', type=int, default=100,
                        help='label MLP dimension')
    parser.add_argument('--mlp_depth', type=int, default=1,
                        help='edge/label MLP depth')

    parser.add_argument('--word_dropout', type=float, default=0.33,
                        help='word embedding dropout prob')
    parser.add_argument('--tag_dropout', type=float, default=0.33,
                        help='tag embedding dropout prob')
    parser.add_argument('--dropout', type=float, default=0.33,
                        help='dropout prob')
    parser.add_argument('--recurrent_dropout', type=float, default=0.33,
                        help='recurrent dropout prob')
    parser.add_argument('--word_l2', type=float, default=0.,
                        help='L2 regularization')
    parser.add_argument('--tag_l2', type=float, default=0.,
                        help='L2 regularization')
    parser.add_argument('--l2', type=float, default=0.,
                        help='L2 regularization')
    parser.add_argument('--max_norm', type=float, default=5.,
                        help='max norm')

    args = parser.parse_args(sys_args)
    print args
    train(args)


if __name__ == '__main__':
    main(sys.argv[1:])
