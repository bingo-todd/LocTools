import os
import numpy as np
import argparse
from BasicTools.parse_file import file2dict
from .add_log import add_log


def make_statistic_log(log_path, result_log_path=None):
    """
    """
    if result_log_path is None:
        log_dir = os.path.dirname(log_path)
        log_name = os.path.basename(log_path)[:-4]
        result_log_name = f'{log_name}_statistic'
        result_log_path = f'{log_dir}/{result_log_name}.txt'

    result_logger = open(result_log_path, 'x')
    result_logger.write(f'# {log_path}\n')
    result_logger.write('# key: mean std median mode\n')

    log = file2dict(log_path, numeric=True, repeat_processor='keep')
    for key in log.keys():
        value = log[key]  # list of values of the same key
        value = np.concatenate(value, axis=0)

        n_var = value.shape[1]
        statistic_result = []
        for var_i in range(n_var):
            # calculate the statistic parameters
            # mean, std, median, mode
            mean = np.mean(value[:, var_i], axis=0)
            std = np.std(value[:, var_i], axis=0)
            median = np.median(value[:, var_i], axis=0)
            unique_value, counts = np.unique(value[:, var_i],
                                             return_counts=True)
            mode = unique_value[np.argmax(counts)]
            statistic_result.append([mean, std, median, mode])
        add_log(result_logger, key, statistic_result)
    result_logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True, type=str,
                        help='path of the input file')
    parser.add_argument('--result-log', dest='result_log_path', type=str,
                        default=None, help='chunksize')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    make_statistic_log(log_path=args.log_path,
                       result_log_path=args.result_log_path)


if __name__ == '__main__':
    main()
