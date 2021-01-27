import os
import numpy as np
import argparse
from BasicTools import parse_file
from LocTools.add_log import add_log


def merge_logs(log0_path, log1_path, result_log_path, repeat_processor):
    """
    Args:
        log0_path, log1_path: the path of logs to be merged
        result_log_path: where the merged log is saved
        repeat_processor: how to deal with possible repeat keys in merged_log,
            'average': default, average the value of repeat keys
            'keep': keep all repeat keys
            'none': overwrite
    """
    if os.path.exists(result_log_path):
        raise Exception(f'{result_log_path} already exists')

    log0 = parse_file.file2dict(log0_path)
    log1 = parse_file.file2dict(log1_path)

    merged_log = {}
    for key in log0.keys():
        if key not in log1.keys():
            raise Exception(f'{key} not in {log1_path}')
        if log0[key] in merged_log.keys():  # repeat key
            merged_log[log0[key]].append(log1[key])
        else:
            merged_log[log0[key]] = [log1[key]]

    # deal with repeat keys in merged_log
    merged_logger = open(result_log_path, 'x')
    for key in merged_log.keys():
        if repeat_processor == 'average':  # numeric should be true
            value = np.mean(
                np.asarray([[[float(item) for item in row.split()]
                             for row in value_tmp.split(';')]
                            for value_tmp in merged_log[key]]),
                axis=0)
            add_log(merged_logger, key, value)
        elif repeat_processor == 'keep':
            for value in merged_log[key]:
                merged_logger.write(f'{key}: {value}\n')
        elif repeat_processor == 'none':
            value = merged_log[key][0]
            merged_logger.write(f'{key}: {value}\n')
        else:
            print('illegal argument for repeat_processor')
    merged_logger.close()


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', nargs='+',
                        required=True, type=str, help='')
    parser.add_argument('--result-log', dest='result_log_path',
                        required=True, type=str)
    parser.add_argument('--repeat-processor', dest='repeat_processor',
                        type=str, default='keep',
                        choices=['average', 'keep', 'none'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    merge_logs(log0_path=args.log_path[0],
               log1_path=args.log_path[1],
               result_log_path=args.result_log_path,
               repeat_processor=args.repeat_processor)


if __name__ == '__main__':
    main()
