import os
import numpy as np
import argparse
from BasicTools import get_file_path


def get_best_eoch(model_dir, loss_name='loss_record'):
    best_epoch_path = f'{model_dir}/best_epoch'
    if not os.path.exists(best_epoch_path):
        train_record = np.load(f'{model_dir}/train_record.npz')
        loss_record = train_record[loss_name]
        if 'cur_epoch' in train_record.keys():
            cur_epoch = train_record['cur_epoch']
        else:
            cur_epoch = np.nonzero(loss_record)[0][-1]
        
        best_epoch = np.argmin(loss_record[:cur_epoch+1])
        
        os.system(f'echo {best_epoch} > {model_dir}/best_epoch')
        return best_epoch

    with open(best_epoch_path, 'r') as file_tmp:
        best_epoch = int(file_tmp.readline())
    return best_epoch


def get_cur_epoch(model_dir):
    with open(f'{model_dir}/checkpoint', 'r') as file_tmp:
        lines = file_tmp.readlines()
        cur_epoch = int(lines[0].split(': ')[1].strip('"')[3:7])
    return cur_epoch


def clean_model_dir(model_dir, n_epoch_left=8, only_best=False):
    index_path_all = get_file_path(model_dir, pattern='ckpt.index')
    epochs = [int(os.path.basename(file_path)[3:7]) 
        for file_path in index_path_all]
    
    cur_epoch = get_cur_epoch(model_dir)
    best_epoch = get_best_eoch(model_dir)
    # minimal epoch to be preserve

    print(f'cur_epoch: {cur_epoch}  best_epoch: {best_epoch}')

    epoch_all = [i for i in range(cur_epoch+1)  
            if os.path.exists(f'{model_dir}/cp-{i:0>4d}.ckpt.index')]

    if only_best:
        if best_epoch is None:
            return
        epoch_deleted_all = [epoch for epoch in epoch_all 
                if epoch != best_epoch]
    else:
        if best_epoch is None:
            best_epoch = 0
        epoch_deleted_all = [epoch for epoch in epoch_all 
                if epoch < best_epoch-n_epoch_left]
     
    print('remove {}'.format('; '.join(map(str, epoch_deleted_all))))
    if input(f'continue (y/n) ?') == 'y':
        for epoch in epoch_deleted_all:
            os.system(f'rm {model_dir}/cp-{epoch:0>4d}.ckpt*')
        
        file_paths = get_file_path(model_dir, pattern='ckpt')
        print('models left {}'.format('; '.join(file_paths)))
       

def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model-dir', dest='model_dir',
                        required=True, type=str, help='directory of model')
    parser.add_argument('--n-epoch-left', dest='n_epoch_left', type=str, 
                        default=8, help='the number of last epoches to be preserved')
    parser.add_argument('--only-best', dest='only_best', type=bool, 
                        default=False, help='delete all preserved model excep the best')
    args = parser.parse_args()
    return args


def main(args):
    clean_model_dir(args.model_dir, args.n_epoch_left, args.only_best)


if __name__ == '__main__':
    args = parse_arg()
    main(args)
