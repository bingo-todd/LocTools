import numpy as np
import argparse
from BasicTools.parse_file import file2dict
from .add_log import add_log


def slice_log(log_path, result_log_path, var_i):
    if not isinstance(var_i, list):
        var_i = [var_i]

    log = file2dict(log_path, numeric=True)
    result_logger = open(result_log_path, 'x')
    for key in log.keys():
        value = log[key]
        part_value = np.concatenate([np.expand_dims(value[:, i], axis=1)
                                     for i in var_i], axis=1)
        add_log(result_logger, key, part_value)
    result_logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True, type=str,
                        help='path of the input file')
    parser.add_argument('--result-log', dest='result_log_path', required=True,
                        type=str, help='')
    parser.add_argument('--var-i', dest='var_i', required=True, nargs='+',
                        type=int, help='index of column or var')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    slice_log(log_path=args.log_path,
              result_log_path=args.result_log_path,
              var_i=args.var_i)


if __name__ == '__main__':
    main()
