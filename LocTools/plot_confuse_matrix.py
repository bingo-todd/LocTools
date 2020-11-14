import matplotlib.pyplot as plt
import numpy as np
import argparse
from BasicTools import plot_tools
from BasicTools.parse_file import file2dict


def plot_confuse_matrix(log_path, fig_path=None):

    max_label = 1  # the confuse matrix is initialized as 1x1
    confuse_matrix = np.zeros((max_label+1, max_label+1), dtype=np.float32)

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
            confuse_matrix = np.pad(confuse_matrix,
                                    ((0, padd_len), (0, padd_len)),
                                    mode='constant',
                                    constant_values=0)
        for estimation, count in zip(estimations, counts):
            confuse_matrix[grandtruth, estimation] = \
                confuse_matrix[grandtruth, estimation]+count

    # normalize
    sum_tmp = np.sum(confuse_matrix, axis=1)
    sum_tmp[sum_tmp == 0] = 1e-20
    confuse_matrix = confuse_matrix/sum_tmp[:, np.newaxis]

    print('confuse matrix')
    print(confuse_matrix)

    if fig_path is not None:
        fig, ax = plot_tools.plot_confuse_matrix(confuse_matrix)
        fig.savefig(fig_path)
        plt.close(fig)
        print(f'figure is saved to {fig_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='parse arguments')
    parser.add_argument('--log-path', dest='log_path', type=str,
                        required=True, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str,
                        default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    plot_confuse_matrix(log_path=args.log_path,
                        fig_path=args.fig_path)


if __name__ == '__main__':
    main()
