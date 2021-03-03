import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


def plot_result_eg(log_paths, file_name_required, fig_path, labels=None,
                   interactive=False, dpi=100, transparent=False,
                   x_starts=None):

    if labels is None:
        labels = [os.path.basename(log_path) for log_path in log_paths]
    if x_starts is None:
        x_starts = [0 for i in range(len(log_paths))]

    y_all = []
    for log_path in log_paths:
        y = None
        log_file = open(log_path, 'r')
        for line in log_file:
            file_path, prob_frame_str = line.strip().split(':')
            file_name = os.path.basename(file_path).split('.')[0]
            if file_name == file_name_required:
                y = np.squeeze(
                    np.asarray(
                        [list(map(float, row.split()))
                         for row in prob_frame_str.split(';')]))
                break
        if y is None:
            raise Exception(f'{file_name} not found')
        y_all.append(y)

    if len(y_all[0].shape) == 1:
        fig, ax = plt.subplots(1, 1)
        for i, y in enumerate(y_all):
            ax.plot(np.arange(y.shape[0])+x_starts[i], y, label=labels[i])
            ax.set_xlim([0, y.shape[0]+x_starts[i]-1])
        ax.legend()
    else:
        fig, ax = plt.subplots(1, len(y_all))
        for i, y in enumerate(y_all):
            ax.imshow(y.T, aspect='auto', cmap='jet', origin='lower')
            ax.set_title(labels[i])
    ax.set_xlabel('frame')

    if interactive:
        plt.show(block=True)

    if fig_path is not None:
        fig.savefig(fig_path, transparent=transparent)
        print(f'fig is saved to {fig_path}')

    return True


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_paths', required=True, nargs='+',
                        type=str, help='log file path')
    parser.add_argument('--file-name', dest='file_name_required',
                        required=True, type=str, help='which file to be plot')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--label', dest='labels', type=str, nargs='+',
                        help='where figure will be saved')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        choices=['true', 'false'], help='')
    parser.add_argument('--x-start', dest='x_starts', type=int, nargs='+',
                        help='where figure will be saved')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='')
    parser.add_argument('--transparent', dest='transparent', type=str,
                        default='false', choices=['true', 'false'], help='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    plot_result_eg(log_paths=args.log_paths,
                   file_name_required=args.file_name_required,
                   fig_path=args.fig_path,
                   labels=args.labels,
                   interactive=args.interactive == 'true',
                   dpi=args.dpi,
                   transparent=args.transparent == 'true',
                   x_starts=args.x_starts)
