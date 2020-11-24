import argparse
import numpy as np

from BasicTools.pase_file import file2dict, dict2file


def filter_log(log_path, filter_path, result_log_path, align='end'):

    x_log = file2dict(log_path, numeric=True)
    filter = file2dict(filter_path, numeric=True)
    y_log = {}

    for key in x_log.keys():
        x_value = x_log[key]
        filter_value = np.squeeze(filter[key]).astype(np.bool)
        n_sample = min((x_value.shape[0], filter_value.shape[0]))
        if align == 'end':
            x_value = x_value[-n_sample:]
            filter_value = filter_value[-n_sample:]
        elif align == 'start':
            x_value = x_value[:n_sample]
            filter_value = filter_value[:n_sample]
        y_value = x_value[filter_value]
        y_log[key] = y_value
    dict2file(y_log, result_log_path)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log-path', dest='log_path', required=True,
                        type=str, help='')
    parser.add_argument('--filter-path', dest='filter_path', required=True,
                        type=str, help='')
    parser.add_argument('--result-log-path', dest='result_log_path',
                        required=True, type=str, help='')
    parser.add_argument('--align', dest='align', type=str, default='end',
                        choices=['end', 'start', 'none'],  help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    filter_log(log_path=args.log_path,
               filter_path=args.filter_path,
               result_log_path=args.result_log_path,
               align=args.align)


if __name__ == '__main__':
    main()
