import argparse
import sys

from main import evaluate


def main(sys_args):
    parser = argparse.ArgumentParser(description='Testing helper for DeNSe parser.')

    parser.add_argument('--test_file', type=str, required=True,
                        help='test data')
    parser.add_argument('--output_file', type=str, default=None,
                        help='output file')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='folder of the model')
    parser.add_argument('--data_model', type=str, default='data.pkl',
                        help='data model file')
    parser.add_argument('--classifier_model', type=str, default='model.pkl',
                        help='classifier model file')
    parser.add_argument('--best_model', type=str, default='best.pkl',
                        help='best classifier model file')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='mini-batch size')
    parser.add_argument('--output_probs', type=bool, default=False,
                        help='output probabilities')

    args = parser.parse_args(sys_args)
    evaluate(args)


if __name__ == '__main__':
    main(sys.argv[1:])
