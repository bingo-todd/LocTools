import argparse
import os
import numpy as np


def split_file(src_file_path, dest_dir_path, n_part):
    """ split src_file into n parts and stored in dest_dir
    """

    src_file_name = os.path.basename(src_file_path)
    src_file = open(src_file_path, 'r')
    n_line = 0
    for _, _ in enumerate(src_file):
        n_line = n_line + 1
    # move file pointer to the start
    src_file.seek(0)
    print(f'n_line: {n_line}')

    n_line_per_part = np.int(np.ceil(n_line/n_part))
    for part_i in range(n_part):
        part_file_path = f'{dest_dir_path}/part_{part_i}-{src_file_name}'
        part_file = open(part_file_path, 'x')
        for line_i in range(n_line_per_part):
            part_file.write(src_file.readline())
        part_file.close()

    src_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--file', dest='src_file_path', required=True,
                        type=str, help='file be be divided')
    parser.add_argument('--dir', dest='dest_dir_path', required=True,
                        type=str, help='where parts are saved')
    parser.add_argument('--n', dest='n_part', required=True,
                        type=int, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    split_file(src_file_path=args.src_file_path,
               dest_dir_path=args.dest_dir_path,
               n_part=args.n_part)


if __name__ == '__main__':
    main()
