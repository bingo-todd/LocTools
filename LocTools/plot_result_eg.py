import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from BasicTools import plot_tools


def plot_result_eg(log_paths, file_name, titles=None, ax_labels=None,
                   fig_type='image', view=[-60, 40], interactive=False,
                   dpi=100, transparent=False, x_starts=None, fig_path=None):

    if titles is None:
        titles = [os.path.basename(log_path) for log_path in log_paths]
    if ax_labels is None:
        ax_labels = [None, None, None]
    if x_starts is None:
        x_starts = [0 for i in range(len(log_paths))]

    y_all = []
    file_name = file_name.split('.')[0]
    for log_path in log_paths:
        y = None
        log_file = open(log_path, 'r')
        for line in log_file:
            file_path_tmp, prob_frame_str = line.strip().split(':')
            file_name_tmp = os.path.basename(file_path_tmp).split('.')[0]
            if file_name_tmp == file_name:
                y = np.squeeze(
                    np.asarray(
                        [list(map(float, row.split()))
                         for row in prob_frame_str.split(';')]))
                break
        if y is None:
            raise Exception(f'{file_name} not found in {log_path}')
        y_all.append(y)

    if len(y_all[0].shape) == 1:
        fig, ax = plt.subplots(1, 1)
        for i, y in enumerate(y_all):
            ax.plot(np.arange(y.shape[0])+x_starts[i], y, label=titles[i])
            ax.set_xlim([0, y.shape[0]+x_starts[i]-1])
        ax.legend()
    else:
        n_log = len(y_all)
        fig = plt.figure(figsize=(6.4+(n_log-1)*2.5, 4.8))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
        ax_tmp = None
        ax = []
        for i, y in enumerate(y_all):
            if fig_type == 'image':
                ax_tmp = fig.add_subplot(1, n_log, i+1, sharex=ax_tmp,
                                         sharey=ax_tmp)
                plot_tools.plot_matrix(y.T, ax=ax_tmp)
                ax_tmp.set_xlabel(ax_labels[0])
                if i == 0:
                    ax_tmp.set_ylabel(ax_labels[1])
            elif fig_type == 'surf':
                ax_tmp = fig.add_subplot(1, n_log, i+1, projection='3d',
                                         sharex=ax_tmp, sharey=ax_tmp,
                                         sharez=ax_tmp)
                plot_tools.plot_surf(y, ax=ax_tmp)
                ax_tmp.view_init(azim=view[0], elev=view[1])
                ax_tmp.set_xlabel(ax_labels[0])
                ax_tmp.set_ylabel(ax_labels[1])
                if i == 0:
                    ax_tmp.set_zlabel(ax_labels[2])
            else:
                raise Exception()
            ax_tmp.set_title(titles[i])
            ax.append(ax_tmp)

    if fig_path is not None:
        fig.savefig(fig_path, transparent=transparent)
        print(f'fig is saved to {fig_path}')

    if interactive:
        plt.show(block=True)

    return fig, ax


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log-paths', dest='log_paths', required=True,
                        nargs='+', type=str, help='log file path')
    parser.add_argument('--file-name', dest='file_name',
                        required=True, type=str, help='which file to be plot')
    parser.add_argument('--titles', dest='titles', type=str, nargs='+',
                        help='')
    parser.add_argument('--ax-labels', dest='ax_labels', type=str, nargs='+',
                        default=None, help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        choices=['true', 'false'], help='')
    parser.add_argument('--x-start', dest='x_starts', type=int, nargs='+',
                        help='where figure will be saved')
    parser.add_argument('--fig-type', dest='fig_type', type=str,
                        default='image', choices=['image', 'surf'], help='')
    parser.add_argument('--view', dest='view', type=int, nargs='+',
                        default=[-60, 30], help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='')
    parser.add_argument('--transparent', dest='transparent', type=str,
                        default='false', choices=['true', 'false'], help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    plot_result_eg(log_paths=args.log_paths,
                   file_name=args.file_name,
                   fig_path=args.fig_path,
                   titles=args.titles,
                   ax_labels=args.ax_labels,
                   fig_type=args.fig_type,
                   view=args.view,
                   interactive=args.interactive == 'true',
                   dpi=args.dpi,
                   transparent=args.transparent == 'true',
                   x_starts=args.x_starts)
