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


def divide_into_bins(x_str_all, n_bin):
    x_value_all = [np.float(item) for item in x_str_all]
    sort_index = np.argsort(x_value_all)
    if n_bin > 1:
        min_value = x_value_all[sort_index[0]]
        max_value = x_value_all[sort_index[-1]]
        bin_width = (max_value-min_value + 1e-10)/n_bin
        bin_edges = np.zeros((n_bin, 2))
        x_str_in_bins = [[] for i in range(n_bin)]
        for bin_i in range(n_bin):
            left_edge = bin_i*bin_width+min_value
            right_edge = left_edge + bin_width
            bin_edges[bin_i] = [left_edge, right_edge]
            for x_value, x_str in zip(x_value_all, x_str_all):
                if x_value >= left_edge and x_value < right_edge:
                    x_str_in_bins[bin_i].append(x_str)
    elif n_bin == -1:
        n_bin = len(x_str_all)
        bin_edges = np.zeros((n_bin, 2))
        bin_edges[:, 0] = [x_value_all[item] for item in sort_index]
        bin_edges[:, 1] = bin_edges[:, 0]
        x_str_in_bins = [[x_str_all[i]] for i in sort_index]
    return bin_edges, x_str_in_bins


def plot_log(log_path, key=None, n_bin=-1, fig_path=None, ax=None,
             var_name=None, plot_settings=None, plot_bar=True, smooth=False,
             log_i=0):
    log = file2dict(log_path, numeric=True, repeat_processor='keep')
    keys = list(log.keys())

    n_field = log[keys[0]][0].shape[1]  # n_repeat * n_row * n_col
    if var_name is None:
        var_name = [f'value_{i}' for i in range(n_field)]

    #
    bin_edges, keys_in_bins = divide_into_bins(keys, n_bin)
    bin_width = np.min(bin_edges[1:, 0] - bin_edges[:-1, 0])
    n_bin = bin_edges.shape[0]

    if ax is None:
        fig, ax = plt.subplots(1, n_field, tight_layout=True,
                               figsize=[4+2*n_field, 4])
    else:
        fig = None
    if not iterable(ax):
        ax = [ax]

    x_shift = bin_width/2/5*log_i
    for field_i in range(n_field):
        x, y_mean, y_std = [], [], []
        for bin_i in range(n_bin):
            y_tmp = []
            for key in keys_in_bins[bin_i]:
                for item in log[key]:
                    y_tmp.append(item[0, field_i])
            if len(y_tmp) > 0:
                x.append(np.mean(bin_edges[bin_i]))
                y_mean.append(np.mean(y_tmp))
                y_std.append(np.std(y_tmp))

        if smooth:
            ax[field_i].errorbar(np.asarray(x)+x_shift, y_mean, yerr=y_std,
                                 alpha=0.6, **plot_settings)
            #
            coefs_len = np.max((1, int(n_bin/10)))
            coefs = np.ones(coefs_len)/coefs_len
            y_mean_smooth = np.convolve(y_mean, coefs, mode='same')
            ax[field_i].plot(np.asarray(x)+x_shift, y_mean_smooth, linewidth=2)
        else:
            ax[field_i].errorbar(np.asarray(x)+x_shift, y_mean, yerr=y_std,
                                 **plot_settings)

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
    parser.add_argument('--bins', dest='n_bin', type=int, default=-1, help='')
    parser.add_argument('--var-name', dest='var_name', nargs='+', type=str,
                        default=None, help='var_name for each value field')
    parser.add_argument('--xlabel', dest='xlabel', type=str, default=None,
                        help='x-axis label')
    parser.add_argument('--ylim', dest='ylim', nargs='+', type=float,
                        default=None,
                        help='range of y-axis')
    parser.add_argument('--linewidth', dest='linewidth', type=int, default=2,
                        help='')
    parser.add_argument('--smooth', dest='smooth', type=str, default='false',
                        choices=['true', 'false'], help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='dpi if figure')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        default='false', choices=['true', 'false'],
                        help='figure path')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='figure path')
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
                       plot_settings={'label': label[0],
                                      'color': colors[0],
                                      'linewidth': args.linewidth},
                       n_bin=args.n_bin,
                       var_name=args.var_name,
                       smooth=args.smooth == 'true')

    for log_i in range(1, n_log):
        plot_log(log_path=args.log_path[log_i],
                 plot_settings={'label': args.label[log_i],
                                'color': colors[log_i],
                                'linewidth': args.linewidth},
                 n_bin=args.n_bin,
                 var_name=args.var_name,
                 ax=ax,
                 plot_bar=False,
                 smooth=args.smooth == 'true',
                 log_i=log_i)

    for ax_i in range(len(ax)):
        ax[ax_i].set_xlabel(args.xlabel)

        if args.ylim is not None:
            ax[ax_i].set_ylim([args.ylim[ax_i*2], args.ylim[ax_i*2+1]])

    ax[-1].legend()

    if args.fig_path is not None:
        print(f'fig is saved to {args.fig_path}')
        fig.savefig(args.fig_path, dpi=args.dpi)

    if args.interactive == 'true':
        plt.show()


if __name__ == '__main__':
    main()
