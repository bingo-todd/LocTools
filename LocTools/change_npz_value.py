import os
import numpy as np
import argparse


def find_item(item, array):
    for i, x in enumerate(array):
        if item == x:
            return i
    return None


def set_dtype(x, dtype):
    dtype_dict = {'float':float, 'int':int, 'str':str}
    if dtype not in dtype_dict.keys():
        print('not supported dtype')
        return x
    else:
        return dtype_dict[dtype](x)


def change_npz_value(npz_path, key, value, dtype, is_print=False):

    if not isinstance(key, list):
        keys = [key]
        values = [value]
        dtypes = [dtype]
    else:
        keys = key
        values = value
        dtypes = dtype

    origin_dict = np.load(npz_path)
    new_dict = {}   
    if keys[0] is not None:
        for key in origin_dict.keys():
            index_tmp = find_item(key, keys)
            if index_tmp is None:
                value_new = origin_dict[key]
            else:
                value_new = set_dtype(values[index_tmp], dtypes[index_tmp])
            new_dict[key] = value_new

        os.system(f'mv {npz_path} {npz_path}.origin')
        np.savez(npz_path, **new_dict)

    if is_print: 
        print(f'origin_dict \n {list(origin_dict.items())}')
        print(f'new_dict \n {list(new_dict.items())}')



def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--npz-path', dest='npz_path',
            required=True, type=str, help='path of npz file')
    parser.add_argument('--key', dest='key', type=str, default=None, 
            help='key whose value to be update')
    parser.add_argument('--value', dest='value', type=str, 
            default=None, help='new name of varible')
    parser.add_argument('--dtype', dest='dtype', type=str, 
            default='str', help='data type of value')
    parser.add_argument('--is-print', dest='is_print', required=False, 
            type=bool, default=False, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    change_npz_value(args.npz_path, args.key, args.value, args.dtype, args.is_print)


if __name__ == '__main__':
    main()
