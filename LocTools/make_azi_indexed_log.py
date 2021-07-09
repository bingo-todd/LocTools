import os
import argparse


def make_azi_indexed_log(log_path, azi_pos=0):

    log_dir, log_name = os.path.dirname(log_path), os.path.basename(log_path)
    azi_indexed_log_path = f'{log_dir}/{log_name[:-4]}_azi_indexed.txt'
    azi_indexed_logger = open(azi_indexed_log_path, 'w')

    logger = open(log_path, 'r')
    while True:
        line = logger.readline()
        if line == '':
            break
        else:
            line = line.strip()
            if line.startswith('#'):
                continue
            file_path, value = line.split(':')
            file_name = os.path.basename(file_path)
            azi = int(file_name.split('_')[azi_pos])
            azi_indexed_logger.write(f'{azi}: {value}\n')
    logger.close()
    azi_indexed_logger.close()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True,
                        type=str, help='path of the input file')
    parser.add_argument('--azi-pos', dest='azi_pos', type=int, default=0,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    make_azi_indexed_log(
        log_path=args.log_path,
        azi_pos=args.azi_pos)
