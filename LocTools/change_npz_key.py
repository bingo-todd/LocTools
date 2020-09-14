import os
import numpy as np
import argparse


def find_item(item, array):
    for i, x in enumerate(array):
        if item == x:
            return i
    return None


def change_npz_key(npz_path, origin_keys, new_keys, is_print=False):

    origin_dict = np.load(npz_path)
    new_dict = {}
   
    if origin_keys[0] is not None:
        for key in origin_dict.keys():
            index_tmp = find_item(key, origin_keys)
            if index_tmp is None:
                key_new = key
            else:
                key_new = new_keys[index_tmp]

            new_dict[key_new] = origin_dict[key]

        os.system(f'mv {npz_path} {npz_path}.origin')
        np.savez(npz_path, **new_dict)

    if is_print: 
        print(f'origin_dict \n {list(origin_dict.items())}')
        print(f'new_dict \n {list(new_dict.items())}')



def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--npz-path', dest='npz_path',
                        required=True, type=str, help='path of npz file')
    parser.add_argument('--origin-key', dest='origin_key', action='append',
                        default=None, type=str, help='origin name of varible')
    parser.add_argument('--new-key', dest='new_key', action='append',
                        default=None, type=str, help='new name of varible')
    parser.add_argument('--is-print', dest='is_print', required=False, 
                        type=bool, default=False, help='new name of varible')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not isinstance(args.origin_key, list):
        origin_keys = [args.origin_key]
    else:
        origin_keys = args.origin_key
    if not isinstance(args.new_key, list):
        new_keys = [args.new_key]
    else:
        new_keys = args.new_key
    change_npz_key(args.npz_path, origin_keys, new_keys, args.is_print)


if __name__ == '__main__':
    main()
