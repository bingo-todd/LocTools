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


def plot_log(log_path, n_bin=-1, fig_path=None, ax=None, var_name=None,
             plot_settings=None, plot_bar=True):

    log = file2dict(log_path, dtype=float)

    keys = list(log.keys())
    keys_float = list(map(float, keys))
    keys_float.sort()
    n_field = len(log[keys[0]])

    if n_bin > 1:
        key_max_float = max(keys_float)
        key_min_float = min(keys_float)
        bin_width = (key_max_float-key_min_float)/n_bin

    if ax is None:
        fig, ax = plt.subplots(1, n_field, tight_layout=True,
                               figsize=[4+2*n_field, 4])
    else:
        fig = None
    if not iterable(ax):
        ax = [ax]

    if var_name is None:
        var_name = [f'value_{i}' for i in range(n_field)]
    for field_i in range(n_field):
        if n_bin > 1:
            x = np.zeros(n_bin)
            y_mean, y_std = np.zeros(n_bin), np.zeros(n_bin)
            n_sample_bin_all = np.zeros(n_bin)
            for bin_i in range(n_bin):
                if bin_i == 0:
                    bin_left = key_min_float
                else:
                    bin_left = bin_left + bin_width
                bin_right = bin_left + bin_width
                x[bin_i] = (bin_left+bin_right)/2.
                y_tmp = [log[key][field_i]
                         for key in keys
                         if float(key) >= bin_left and float(key) < bin_right]
                n_sample_bin_all[bin_i] = len(y_tmp)
                y_mean[bin_i] = np.mean(y_tmp)
                y_std[bin_i] = np.std(y_tmp)

            ax[field_i].errorbar(x, y_mean, yerr=y_std/2, **plot_settings)
            ax[field_i].set_title(var_name[field_i])

            if plot_bar:
                ax_n_sample = ax[field_i].twinx()
                ax_n_sample.bar(x, n_sample_bin_all, alpha=0.3,
                                width=bin_width*0.8)
                ax_n_sample.set_ylabel('n_sample')
        else:
            ax[field_i].plot(keys_float,
                             [log[key][field_i] for key in keys],
                             **plot_settings)
            ax[field_i].set_title(var_name[field_i])

    if fig_path is not None:
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
    parser.add_argument('--fig-path', dest='fig_path',
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
    fig.savefig(args.fig_path, dpi=args.dpi)


if __name__ == '__main__':
    main()
