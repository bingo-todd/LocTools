import numpy as np
import argparse
import matplotlib.pyplot as plt
from BasicTools.parse_file import file2dict


def plot_hist(log_path, fig_path, n_bin):
    log = file2dict(log_path, numeric=True)
    values = np.squeeze(
        np.concatenate(
            list(log.values()),
            axis=1))
    print(values)
    freqs, bin_edges = np.histogram(values, bins=n_bin, density=True)
    centers = (bin_edges[1:] + bin_edges[:-1])/2

    fig, ax = plt.subplots(1, 1)
    ax.bar(centers, freqs)
    ax.set_ylabel('freq')
    fig.savefig(fig_path)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log', dest='log_path', required=True,
                        type=str, help='path of the input file')
    parser.add_argument('--fig-path', dest='fig_path', required=True,
                        type=str, help='')
    parser.add_argument('--n-bin', dest='n_bin', required=True,
                        type=int, help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    plot_hist(log_path=args.log_path,
              fig_path=args.fig_path,
              n_bin=args.n_bin)


if __name__ == '__main__':
    main()
