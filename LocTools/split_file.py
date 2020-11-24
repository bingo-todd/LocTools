import argparse
import os
import numpy as np


def split_file(src_file_path, dest_dir_path, n_part):
    """ split src_file into n parts and stored in dest_dir
    """

    with open(src_file_path, 'r') as src_file:
        lines = src_file.readlines()

    n_line = len(lines)
    n_line_per_part = np.int(np.ceil(n_line/n_part))

    src_file_name = os.path.basename(src_file_path)
    for part_i in range(n_part):
        part_file_path = f'{dest_dir_path}/part_{part_i}-{src_file_name}'
        start_i = n_line_per_part*part_i
        end_i = start_i + n_line_per_part
        with open(part_file_path, 'w') as part_file:
            part_file.write(''.join(lines[start_i:end_i]))


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--src-file', dest='src_file_path', required=True,
                        type=str, help='file be be divided')
    parser.add_argument('--dest-dir', dest='dest_dir_path', required=True,
                        type=str, help='where parts are saved')
    parser.add_argument('--n-part', dest='n_part', required=True,
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
