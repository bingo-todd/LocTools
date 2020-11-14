import argparse
from BasicTools.parse_file import file2dict, dict2file


def math_logs(log0_path, log1_path, result_log_path, operator):
    """
    do basic math  with log files
    Args:
        log0_path, log1_path: the path of logs to be result
        result_log_path: where the result is saved
        operator: minus or add
    """
    log0 = file2dict(log0_path, numeric=True)
    log1 = file2dict(log1_path, numeric=True)

    result_log = {}
    for key in log0.keys():
        if key not in log1.keys():  #
            raise Exception(f'{key} not in {log1_path}')
        if operator == 'sum':
            result_log[key] = log0[key] + log1[key]
        elif operator == 'minus':
            result_log[key] = log0[key] - log1[key]
        else:
            raise Exception('unsupported options for operator')
    dict2file(result_log, result_log_path)


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', nargs='+',
                        required=True, type=str, help='')
    parser.add_argument('--result-log', dest='result_log_path',
                        required=True, type=str)
    parser.add_argument('--operator', dest='operator', type=str,
                        choices=['sum', 'minus'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    math_logs(log0_path=args.log_path[0],
              log1_path=args.log_path[1],
              result_log_path=args.result_log_path,
              operator=args.operator)


if __name__ == '__main__':
    main()
