import logging
import argparse
import os

os.makedirs(os.path.expanduser('~/Exp_logs'), exist_ok=True)


def exp_logger(log_path, message):
    log_path = os.path.expanduser(log_path)

    logger = logging.getLogger('exp_logger')
    # open file for logging
    file_handler = logging.FileHandler(log_path, 'a')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.DEBUG)

    # write message to log file
    logger.info(message)


def parse_args():
    parser = argparse.ArgumentParser(
        description='log interface of experiments')
    parser.add_argument('--log-path', dest='log_path', type=str,
                        default='~/Exp_logs/exp.log', help='path of log file')
    parser.add_argument('--message', dest='message',
                        required=True, type=str, action='append',
                        help='message to be logged', nargs='+')
    args = parser.parse_args()
    return args


def main(log_path='~/Exp_logs/exp.log', message=None):
    if message is None:
        return
    else:
        exp_logger(log_path, message)


if __name__ == '__main__':
    args = parse_args()
    exp_logger(args.log_path, args.message)
