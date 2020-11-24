import argparse
from BasicTools.parse_file import file2dict, dict2file


def subtract_logs(log0_path, log1_path, result_log_path):
    """
    log0 - log1
    Args:
        log0_path, log1_path: the path of logs to be result
        result_log_path: where the result is saved
    """
    log0 = file2dict(log0_path, numeric=True)
    log1 = file2dict(log1_path, numeric=True)

    result_log = {}
    for key in log0.keys():
        result_log[key] = log0[key] - log1[key]
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
    subtract_logs(log0_path=args.log_path[0],
                  log1_path=args.log_path[1],
                  result_log_path=args.result_log_path)


if __name__ == '__main__':
    main()
