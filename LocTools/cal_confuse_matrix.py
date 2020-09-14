import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from BasicTools import get_file_path
from BasicTools import plot_tools


def cal_confuse_matrix(log_path, result_path, fig_path):

    if len(log_path) == 1 and os.path.isdir(log_path[0]):
        log_path_all = get_file_path(log_path[0], suffix='.txt', is_absolute=True)
    else:
        log_path_all = log_path

    max_index = 1 
    confuse_matrix = np.zeros((max_index+1, max_index+1), dtype=np.float32)
    for log_path_tmp in log_path_all:
        with open(log_path_tmp, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if len(line)<1:
                    continue

                label_info, estimation_info = line.split(';')
                label = int(label_info.split()[0])
                estimation = list( 
                        map(lambda x: int(float(x)), 
                            estimation_info.split()))
                # count the present frequency of each estimation 
                indexs, counts = np.unique(estimation, return_counts=True)
                max_index_tmp = np.max(indexs)
                if max_index_tmp > max_index:
                    padd_len = max_index_tmp-max_index
                    max_index = max_index_tmp
                    confuse_matrix = np.pad(
                            confuse_matrix, 
                            ((0, padd_len), (0, padd_len)), 
                            mode='constant', 
                            constant_values=0)
                else:
                    for index, count in zip(indexs, counts):
                        confuse_matrix[label, index] = confuse_matrix[label, index] + count

    # normalize
    sum_tmp = np.sum(confuse_matrix, axis=1)
    sum_tmp[sum_tmp==0] = 1e-20
    confuse_matrix = confuse_matrix/sum_tmp[:, np.newaxis]
    
    print('confuse matrix')
    print(confuse_matrix)
    if len(os.path.dirname(result_path)) > 0:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as log_file:
        for i in range(max_index):
            log_file.write(' '.join(map(lambda x:f'{x:>5.2f}', confuse_matrix[i])))
            log_file.write('\n')

    if fig_path is not None:
        fig, ax = plot_tools.plot_confuse_matrix(confuse_matrix)
        fig.savefig(fig_path)
        plt.close(fig)
            

def parse_args():
    parser = argparse.ArgumentParser(description='parse arguments')
    parser.add_argument('--log-path', dest='log_path', type=str, 
            nargs='+', required=True)
    parser.add_argument('--result-path', dest='result_path', type=str,
            required=True)
    parser.add_argument('--fig-path', dest='fig_path', type=str, 
            default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()  
    cal_confuse_matrix(args.log_path, args.result_path, args.fig_path)


if __name__ == '__main__':
    main()
