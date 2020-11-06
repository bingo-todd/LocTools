import argparse
from BasicTools.parse_file import file2dict


def sort_log(log_path):
    logger = file2dict(log_path)

    keys = list(logger.keys())
    keys.sort()

    with open(log_path, 'w') as log_file:
        for key in keys:
            log_file.write(f'{key}: {logger[key]}\n')
    log_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True,
                        type=str, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    sort_log(args.log_path)


if __name__ == '__main__':
    main()
