import os
import shutil
import argparse
from BasicTools.get_file_path import get_file_path, get_realpath
from BasicTools.parse_file import file2dict


def merge_folders(folder_paths, merged_folder_path, concat_text=True,
                  keep_source=True, modify_log=True, verbose=False,
                  root_dir='~/Work_Space'):

    os.makedirs(merged_folder_path, exist_ok=True)

    for folder_path in folder_paths:
        folder_realpath = get_realpath(folder_path, root_dir)
        src_file_paths = get_file_path(folder_path, is_absolute=True)
        for src_file_path in src_file_paths:
            file_name = os.path.basename(src_file_path)
            dest_file_path = f'{merged_folder_path}/{file_name}'

            if verbose:
                print(f'{src_file_path} >>> {dest_file_path}')

            if file_name.endswith('.txt'):
                if modify_log:  # change file path in log correspondingly
                    try:  # if txt file is a log
                        dict_obj = file2dict(src_file_path)
                        new_dict_obj = {}
                        for tmp_realpath in dict_obj.keys():
                            # if the file of file_path_tmp is in the dir,
                            # update its path
                            tmp_dir = os.path.dirname(tmp_realpath)
                            tmp_name = os.path.basename(tmp_realpath)
                            if tmp_dir == folder_realpath:
                                new_tmp_realpath = get_realpath(
                                    f'{merged_folder_path}/{tmp_name}',
                                    root_dir)
                            new_dict_obj[new_tmp_realpath] = \
                                dict_obj[tmp_realpath]
                        text = '\n'.join(
                            [f'{key}: {value}'
                             for key, value in new_dict_obj.items()])
                    except Exception:
                        src_file = open(src_file_path, 'r')
                        text = '\n'.join(src_file.readlines())
                        src_file.close()
                else:
                    src_file = open(src_file_path, 'r')
                    text = '\n'.join(src_file.readlines())
                    src_file.close()

                if concat_text:
                    dest_file = open(dest_file_path, 'a')
                else:
                    dest_file = open(dest_file_path, 'w')
                dest_file.write(text)
                dest_file.write('\n')
                dest_file.close()

            else:  # overwrite if not a txt file
                shutil.copy(src_file_path, dest_file_path)

        if not keep_source:
            shutil.rmtree(folder_path)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--folders', dest='folder_paths', required=True,
                        nargs='+', type=str, help='folder to be merged')
    parser.add_argument('--merged-folder', dest='merged_folder_path',
                        type=str, help='merged folder')
    parser.add_argument('--root-dir', dest='root_dir', type=str,
                        default='~/Work_Space', help='merged folder')
    parser.add_argument('--verbose', dest='verbose', type=str,
                        choices=['true', 'false'], default='false', help='')
    parser.add_argument('--concat-text', dest='concat_text', type=str,
                        choices=['true', 'false'], default='true', help='')
    parser.add_argument('--keep-source', dest='keep_source', type=str,
                        choices=['true', 'false'], default='true', help='')
    parser.add_argument('--modify-log', dest='modify_log', type=str,
                        choices=['true', 'false'], default='true', help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    merge_folders(folder_paths=args.folder_paths,
                  merged_folder_path=args.merged_folder_path,
                  concat_text=args.concat_text == 'true',
                  keep_source=args.keep_source == 'true',
                  modify_log=args.modify_log == 'true',
                  verbose=args.verbose == 'true',
                  root_dir=args.root_dir)


if __name__ == '__main__':
    main()
