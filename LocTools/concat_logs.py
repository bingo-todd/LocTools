import os
import argparse
from BasicTools import parse_file


def concat_logs(log0_path, log1_path, result_log_path):

    if os.path.exists(result_log_path):
        raise Exception(f'{result_log_path} already exists')

    log0 = parse_file.file2dict(log0_path)
    log1 = parse_file.file2dict(log1_path)

    concat_log = {}
    for key in log0.keys():
        if log0[key] not in log1.keys():
            raise Exception(f'{log0[key]} not in {log1_path}')
        concat_log[key] = log1[log0[key]]
    parse_file.dict2file(concat_log, result_log_path)


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', nargs='+',
                        required=True, type=str, help='')
    parser.add_argument('--result-log', dest='result_log_path',
                        required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    concat_logs(args.log_path[0], args.log_path[1], args.result_log_path)


if __name__ == '__main__':
    main()
