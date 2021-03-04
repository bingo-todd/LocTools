import os
import argparse
from BasicTools.parse_file import file2dict, dict2file


def combine_logs(log0_path, log1_path, result_log_path):

    if os.path.exists(result_log_path):
        raise Exception(f'{result_log_path} already exists')

    log0 = file2dict(log0_path, repeat_processor='except')
    log1 = file2dict(log1_path, repeat_processor='keep')

    result_log = {}
    for key in log0.keys():
        try:
            value0 = log0[key]
            value1 = log1[key]
            result_log[value0] = value1
        except Exception as e:
            print(f'{log0[key]} not in {log1_path}')
            raise Exception(e)

    dict2file(result_log, result_log_path)


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
    combine_logs(args.log_path[0], args.log_path[1], args.result_log_path)


if __name__ == '__main__':
    main()
