import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from BasicTools import plot_tools


def find_in_log(log_path, file_name):
    y = None
    log_file = open(log_path, 'r')
    for line in log_file:
        file_path_tmp, prob_frame_str = line.strip().split(':')
        file_name_tmp = os.path.basename(file_path_tmp).split('.')[0]
        if file_name_tmp == file_name:
            y = np.squeeze(
                np.asarray(
                    [[float(item) for item in row.split()]
                     for row in prob_frame_str.split(';')]))
            break
    if y is None:
        raise Exception(f'{file_name} not found in {log_path}')
    return y


def plot_result_eg(log_paths, file_name, titles=None, ax_labels=None,
                   fig_type='image', view=[-60, 40], interactive=False,
                   dpi=100, transparent=False, x_starts=None, fig_path=None):

    if titles is None:
        titles = [None for log_path in log_paths]
    if ax_labels is None:
        ax_labels = [None, None, None]
    if x_starts is None:
        x_starts = [0 for i in range(len(log_paths))]

    file_name = file_name.split('.')[0]
    y_all = [find_in_log(log_path, file_name) for log_path in log_paths]

    n_log = len(y_all)

    if fig_type == 'image':
        fig, ax = plot_tools.subplots(1, n_log, sharex=True, sharey=True)
        if n_log == 1:
            ax = [ax]
        max_amp = np.max([np.max(y) for y in y_all])
        min_amp = np.min([np.min(y) for y in y_all])
        for i in range(n_log):
            if i == n_log-1:
                plot_tools.plot_matrix(
                    y_all[i].T, ax=ax[i], vmax=max_amp, vmin=min_amp, fig=fig)
            else:
                plot_tools.plot_matrix(
                    y_all[i].T, ax=ax[i], vmax=max_amp, vmin=min_amp)
            if i == 0:
                ax[i].set_ylabel(ax_labels[1])
            ax[i].set_xlabel(ax_labels[0])
            ax[i].set_title(titles[i])

    elif fig_type == 'surf':
        fig = plt.figure(figsize=plot_tools.get_figsize(1, n_log))
        ax_tmp = None
        for y_i, y in enumerate(y_all):
            ax_tmp = fig.add_subplot(
                n_log, 1, y_i+1, projection='3d',
                sharex=ax_tmp, sharey=ax_tmp, sharez=ax_tmp)
            plot_tools.plot_surf(y, ax=ax_tmp)
            ax_tmp.view_init(azim=view[0], elev=view[1])
            ax_tmp.set_xlabel(ax_labels[0])
            ax_tmp.set_ylabel(ax_labels[1])
            if y_i == 0:
                ax_tmp.set_zlabel(ax_labels[2])
            ax_tmp.set_title(titles[y_i])
            ax.append(ax_tmp)
    else:
        raise Exception()

    if fig_path is not None:
        fig.savefig(fig_path, transparent=transparent)
        print(f'fig is saved to {fig_path}')

    if interactive:
        plt.show(block=True)

    return fig, ax


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True,
                        nargs='+', type=str, help='log file path')
    parser.add_argument('--file-name', dest='file_name',
                        required=True, type=str, help='which file to be plot')
    parser.add_argument('--title', dest='title', type=str, nargs='+',
                        help='')
    parser.add_argument('--ax-label', dest='ax_label', type=str, nargs='+',
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
    plot_result_eg(log_paths=args.log_path,
                   file_name=args.file_name,
                   fig_path=args.fig_path,
                   titles=args.title,
                   ax_labels=args.ax_label,
                   fig_type=args.fig_type,
                   view=args.view,
                   interactive=args.interactive == 'true',
                   dpi=args.dpi,
                   transparent=args.transparent == 'true',
                   x_starts=args.x_starts)
