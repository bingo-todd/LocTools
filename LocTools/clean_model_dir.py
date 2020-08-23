import os
import argparse
from BasicTools import get_file_path


def get_best_eoch(model_dir):
    best_epoch_path = f'{model_dir}/best_epoch'
    if not os.path.exists(best_epoch_path):
        return 1e5

    with open(best_epoch_path, 'r') as file_tmp:
        best_epoch = int(file_tmp.readline())
    return best_epoch


def get_cur_epoch(model_dir):
    with open(f'{model_dir}/checkpoint', 'r') as file_tmp:
        lines = file_tmp.readlines()
        cur_epoch = int(lines[0].split(': ')[1].strip('"')[3:7])
    return cur_epoch


def clean_model_dir(model_dir, n_epoch_left=5):
    index_path_all = get_file_path(model_dir, pattern='ckpt.index')
    epochs = [int(os.path.basename(file_path)[3:7]) 
        for file_path in index_path_all]
    
    cur_epoch = get_cur_epoch(model_dir)
    best_epoch = get_best_eoch(model_dir)
    # minimal epoch to be preserve
    min_epoch = min(cur_epoch, best_epoch) - n_epoch_left
    for epoch in epochs:
        if epoch < min_epoch:
            print(f'delete cp-{epoch:0>4d}.*')
            os.remove(f'{model_dir}/cp-{epoch:0>4d}.ckpt.index')
            os.remove(f'{model_dir}/cp-{epoch:0>4d}.ckpt.data-00000-of-00002')
            os.remove(f'{model_dir}/cp-{epoch:0>4d}.ckpt.data-00001-of-00002')
    
    file_path_all = get_file_path(model_dir, pattern='ckpt')
    print('model files left')
    for file_path in file_path_all:
        print(file_path)
   

def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model-dir', dest='model_dir',
                        required=True, type=str, help='directory of model')
    parser.add_argument('--n-epoch-left', dest='n_epoch_left', type=str, 
                        default=5, help='the number of last epoches to be preserved')
    args = parser.parse_args()
    return args


def main(args):
    clean_model_dir(args.model_dir, args.n_epoch_left)


if __name__ == '__main__':
    args = parse_arg()
    main(args)
