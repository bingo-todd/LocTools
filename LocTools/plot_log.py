import numpy as np
import seaborn
import argparse
import matplotlib.pyplot as plt
from BasicTools.parse_file import file2dict


def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    return True


def plot_log(log_path, key=None, n_bin=-1, fig_path=None, ax=None,
             var_name=None, plot_settings=None, plot_bar=True):

    log = file2dict(log_path, numeric=True, repeat_processor='keep')
    keys = list(log.keys())
    n_field = log[keys[0]][0].shape[1]  # n_repeat * n_row * n_col
    if var_name is None:
        var_name = [f'value_{i}' for i in range(n_field)]

    # split keys into bins
    keys_value = np.asarray([float(key) for key in keys])
    sort_order = np.argsort(keys_value)
    keys_value_sorted = keys_value[sort_order]
    keys_sorted = [keys[i] for i in sort_order]
    if n_bin > 1:
        max_key_value = keys_value_sorted[-1]
        min_key_value = keys_value_sorted[0]
        bin_width = (max_key_value-min_key_value + 1e-10)/n_bin
        keys_in_bins = [[] for i in range(n_bin)]
        x = np.zeros(n_bin)
        for bin_i in range(n_bin):
            left_edge = bin_i*bin_width+min_key_value
            right_edge = left_edge + bin_width
            x[bin_i] = (left_edge+right_edge)/2.
            for key in log.keys():
                key_float = float(key)
                if key_float >= left_edge and key_float < right_edge:
                    keys_in_bins[bin_i].append(key)
    elif n_bin == -1:
        n_bin = len(keys_sorted)
        x = keys_value_sorted
        bin_width = np.min(keys_value_sorted[1:] - keys_value_sorted[:-1])
        keys_in_bins = [[key] for key in keys_sorted]

    if ax is None:
        fig, ax = plt.subplots(1, n_field, tight_layout=True,
                               figsize=[4+2*n_field, 4])
    else:
        fig = None
    if not iterable(ax):
        ax = [ax]

    for field_i in range(n_field):
        y_mean, y_std = np.zeros(n_bin), np.zeros(n_bin)
        for bin_i in range(n_bin):
            y_tmp = np.asarray(
                [log[key][0][field_i] for key in keys_in_bins[bin_i]])
            y_mean[bin_i] = np.mean(y_tmp)
            y_std[bin_i] = np.std(y_tmp)

        ax[field_i].errorbar(x, y_mean, yerr=y_std/2, **plot_settings)
        ax[field_i].set_title(var_name[field_i])

    if fig_path is not None:
        print(f'fig is saved to {fig_path}')
        fig.savefig(fig_path)

    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True, nargs='+',
                        type=str, help='path of the input file')
    parser.add_argument('--label', dest='label', nargs='+', type=str,
                        default=None, help='label for each log')
    parser.add_argument('--bins', dest='n_bin', type=int,
                        default=-1, help='')
    parser.add_argument('--fig-path', dest='fig_path', required=True,
                        type=str, default=None, help='figure path')
    parser.add_argument('--var-name', dest='var_name', nargs='+',
                        type=str, default=None,
                        help='var_name for each value field')
    parser.add_argument('--xlabel', dest='xlabel', type=str, default=None,
                        help='x-axis label')
    parser.add_argument('--ylim', dest='ylim', nargs='+',
                        type=float, default=None,
                        help='range of y-axis')
    parser.add_argument('--linewidth', dest='linewidth', type=int, default=2,
                        help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='dpi if figure')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    n_log = len(args.log_path)
    colors = seaborn.color_palette('tab10', n_colors=n_log)
    if args.label is None:
        label = [None for i in range(n_log)]
    else:
        label = args.label

    fig, ax = plot_log(log_path=args.log_path[0],
                       n_bin=args.n_bin,
                       var_name=args.var_name,
                       plot_settings={'label': label[0],
                                      'color': colors[0],
                                      'linewidth': args.linewidth})
    for log_i in range(1, n_log):
        plot_log(log_path=args.log_path[log_i],
                 plot_settings={'label': args.label[log_i],
                                'color': colors[log_i],
                                'linewidth': args.linewidth},
                 n_bin=args.n_bin,
                 var_name=args.var_name,
                 ax=ax,
                 plot_bar=False)

    for ax_i in range(len(ax)):
        ax[ax_i].set_xlabel(args.xlabel)

        if args.ylim is not None:
            ax[ax_i].set_ylim([args.ylim[ax_i*2], args.ylim[ax_i*2+1]])

    ax[-1].legend()

    print(f'fig is saved to {args.fig_path}')
    fig.savefig(args.fig_path, dpi=args.dpi)


if __name__ == '__main__':
    main()
