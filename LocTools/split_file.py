import argparse
import os
import numpy as np


def split_file(file_path, n_part, dir_path=None):
    """ split src_file into n parts and stored in dir
    """

    if dir_path is None:
        dir_path = os.path.dirname(file_path)
        if len(dir_path) < 1:
            dir_path = '.'  # current directory

    src_file_name = os.path.basename(file_path)

    part_file_paths = []
    part_file_all = []
    for part_i in range(n_part):
        part_file_path = f'{dir_path}/{src_file_name}.part{part_i}'
        part_file_paths.append(part_file_path)
        part_file = open(part_file_path, 'x')
        part_file_all.append(part_file)

    src_file = open(file_path, 'r')
    for line_i, line in enumerate(src_file):
        part_i = np.int(np.mod(line_i, n_part))
        part_file_all[part_i].write(line)
        part_file_all[part_i].flush()

    src_file.close()
    for part_file in part_file_all:
        part_file.close()

    return part_file_paths


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--file', dest='file_path', required=True,
                        type=str, help='file be be divided')
    parser.add_argument('--dir', dest='dir_path', type=str, default=None,
                        help='where parts are saved')
    parser.add_argument('--n', dest='n_part', required=True,
                        type=int, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    split_file(file_path=args.file_path,
               dir_path=args.dir_path,
               n_part=args.n_part)


if __name__ == '__main__':
    main()
