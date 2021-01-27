import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from BasicTools import plot_tools
from BasicTools.parse_file import file2dict


def cal_confuse_matrix(log_path):

    max_label = 1  # the confuse matrix is initialized as 1x1
    CM = np.zeros((max_label+1, max_label+1), dtype=np.float32)

    log = file2dict(log_path, numeric=True, repeat_processor='keep')
    for grandtruth, estimations in log.items():
        grandtruth = int(grandtruth)
        # merge all estimations coreesponding to the same label
        estimations = np.concatenate(estimations).astype(np.int)
        # count the present frequency of each estimation
        estimations, counts = np.unique(estimations, return_counts=True)
        max_label_tmp = max((np.max(estimations), grandtruth))
        if max_label_tmp > max_label:  # expand confuse matrix
            padd_len = int(max_label_tmp-max_label)
            max_label = max_label_tmp
            CM = np.pad(CM, ((0, padd_len), (0, padd_len)), mode='constant',
                        constant_values=0)
        for estimation, count in zip(estimations, counts):
            CM[grandtruth, estimation] = \
                CM[grandtruth, estimation]+count

    # normalize
    sum_tmp = np.sum(CM, axis=1)
    sum_tmp[sum_tmp == 0] = 1e-20
    CM = CM/sum_tmp[:, np.newaxis]
    return CM


def parse_args():
    parser = argparse.ArgumentParser(description='calculate confuse matrix \
                                     from log file, confuse matrix can be \
                                     plot or saved by specifying fig-path or \
                                     result-path')
    parser.add_argument('--log', dest='log_path', type=str, required=True,
                        help='')
    parser.add_argument('--npy-path', dest='npy_path', type=str,
                        default=None, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--print', dest='print', type=str,
                        choices=['true', 'false'], default='false',
                        help='whether to print confuse matrix directly in \
                        terminal')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.npy_path is None and args.fig_path is None:
        return None

    if args.npy_path is not None and os.path.exists(args.npy_path):
        raise Exception(f'{args.npy_path} already exists')

    if args.fig_path is not None and os.path.exists(args.fig_path):
        raise Exception(f'{args.fig_path} already exists')

    CM = cal_confuse_matrix(log_path=args.log_path)

    if args.print == 'true':
        print('confuse matrix')
        with np.printoptions(precision=2, suppress=True, floatmode='fixed'):
            print(CM)

    if args.npy_path is not None:
        np.save(args.npy_path, CM)

    if args.fig_path is not None:
        n_label = CM.shape[0]
        if n_label < 50:
            fig, ax = plot_tools.plot_confuse_matrix(CM)
        else:
            fig, ax = plt.subplots(1, 1)
            plt.imshow(CM, cmap='Blues', aspect='auto')
            plt.colorbar()
            ax.set_xlabel('Estimation')
            ax.set_ylabel('Grandtruth')
        fig.savefig(args.fig_path)
        print(f'figure is saved to {args.fig_path}')


if __name__ == '__main__':
    main()
