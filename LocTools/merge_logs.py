import numpy as np
import argparse
from BasicTools import parse_file


def merge_logs(log0_path, log1_path, merged_log_path, average_repeat=False,
               dtype=float):
    log0 = parse_file.file2dict(log0_path)
    log1 = parse_file.file2dict(log1_path)

    merged_log = {}
    for key in log0.keys():
        if key not in log1.keys():
            raise Exception(f'{key} not in {log1_path}')
        if log0[key] in merged_log.keys():
            if average_repeat:
                if isinstance(merged_log[log0[key]], list):
                    merged_log[log0[key]].append(log1[key])
                else:
                    merged_log[log0[key]] = [merged_log[log0[key]], log1[key]]
            else:
                raise Exception(f'repeate {log0[key]} in {log0_path}')
        else:
            merged_log[log0[key]] = log1[key]

    if average_repeat:
        for key in merged_log.keys():
            if isinstance(merged_log[key], list):
                value_tmp = np.asarray(
                    [[list(map(dtype, row.strip().split(' ')))
                      for row in value_tmp.strip().split(';')]
                     for value_tmp in merged_log[key]])
                # print(f'key:{key} n_sample:{value_tmp.shape[0]}')
                value_averaged = np.mean(value_tmp, axis=0)
                merged_log[key] = '; '.join([' '.join(map(str, row))
                                             for row in value_averaged])
    return
    parse_file.dict2file(merged_log_path, merged_log)


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', nargs='+',
                        required=True, type=str, help='')
    parser.add_argument('--merged-log', dest='merged_log_path',
                        required=True, type=str)
    parser.add_argument('--average-repeat', dest='average_repeat',
                        type=str, default='true', choices=['true', 'false'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    merge_logs(log0_path=args.log_path[0],
               log1_path=args.log_path[1],
               merged_log_path=args.merged_log_path,
               average_repeat=args.average_repeat == 'true')


if __name__ == '__main__':
    main()
