import os
import re
import argparse
from BasicTools.get_file_path import get_file_path


def combine_file(dir_path, dest_file_path):
    file_paths = get_file_path(dir_path,
                               filter_func=lambda x: x.find('part_') >= 0,
                               is_absolute=True)
    n_part = len(file_paths)
    if n_part < 1:
        print(f'no part file found in {dir_path}')
        return

    *_, file_name = re.split("part_[\\d]*-", os.path.basename(file_paths[0]))

    with open(dest_file_path, 'x') as dest_file:
        for part_i in range(n_part):
            src_file_path = f'{dir_path}/part_{part_i}-{file_name}'
            with open(src_file_path, 'r') as src_file:
                lines = src_file.readlines()
            dest_file.write(''.join(lines))
            dest_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--dir', dest='dir_path', required=True,
                        type=str, help='')
    parser.add_argument('--file', dest='dest_file_path',
                        required=True, type=str, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    combine_file(dir_path=args.dir_path,
                 dest_file_path=args.dest_file_path)


if __name__ == '__main__':
    main()
