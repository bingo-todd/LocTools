import os
import re
import argparse
from BasicTools.get_file_path import get_file_path


def combine_files(dir_path, file_path=None, keep_part_file=True,
                  keep_comment=True):
    """ combine files into 1 file, correpsonding to split_file. Files to be
    combined should be name as [file_name].part[part_i]
    """
    part_file_paths = get_file_path(dir_path,
                                    filter_func=lambda x: x.find('.part') >= 0,
                                    is_absolute=True,
                                    max_depth=0)

    n_part = len(part_file_paths)
    if n_part < 1:
        print(f'no part file found in {dir_path}')
        return

    file_name, *_ = re.split(".part[\\d]*",
                             os.path.basename(part_file_paths[0]))
    if file_path is None:
        file_path = f'{dir_path}/{file_name}'

    with open(file_path, 'x') as file_obj:
        for part_i in range(n_part):
            part_file_path = f'{dir_path}/{file_name}.part{part_i}'
            with open(part_file_path, 'r') as src_file:
                lines = src_file.readlines()
                if not keep_comment:
                    lines = [line for line in lines
                             if not line.startswith('#')]
                file_obj.write(''.join(lines))
                file_obj.flush()

            if not keep_part_file:
                os.remove(part_file_path)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--dir', dest='dir_path', required=True,
                        type=str, help='')
    parser.add_argument('--file', dest='file_path', type=str,
                        default=None, help='')
    parser.add_argument('--keep-part-file', dest='keep_part_file', type=str,
                        default='true', choices=['true', 'false'], help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    combine_files(dir_path=args.dir_path,
                  file_path=args.file_path,
                  keep_part_file=args.keep_part_file == 'true')


if __name__ == '__main__':
    main()
