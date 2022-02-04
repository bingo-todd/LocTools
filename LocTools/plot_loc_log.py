import os
import argparse

from BasicTools.parse_file import file2dict
from BasicTools import plot_tools


def plot_loc_log(log_path, file_name, fig_path):

    logger = file2dict(log_path, numeric=True)

    file_paths = list(logger.keys())
    value = None
    for file_path in file_paths:
        if os.path.basename(file_path).find(file_name) > -1:
            value = logger[file_path]

    if value is not None:
        fig, ax = plot_tools.subplots(1, 1)
        plot_tools.plot_matrix(value.T, ax=ax, fig=fig)
        fig.savefig(fig_path)
    else:
        print(f'{file_name} not found')


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--fig_path', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    plot_loc_log(args.log_path, args.file_name, args.fig_path)
