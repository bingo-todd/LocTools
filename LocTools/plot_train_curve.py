import numpy as np
import matplotlib.pyplot as plt
import argparse


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 200


def tolist(x):
    if not isinstance(x, list):
        return [x]
    elif x is None:
        return [None]
    else:
        return x


def plot_train_curve(train_record_path, label, fig_path=None,
                     var_name='loss_record', dpi=100, interactive=False,
                     linewidth=2):

    train_record_paths = tolist(train_record_path)
    labels = tolist(label)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    for train_record_path, label in zip(train_record_paths, labels):
        record_info = np.load(train_record_path)
        loss_record = record_info[var_name]
        n_epoch = np.max(np.nonzero(loss_record)[0])
        loss_record = loss_record[:n_epoch+1]
        ax.plot(loss_record, label=label, linewidth=linewidth)

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()

    if fig_path is not None:
        fig.savefig(fig_path, dpi=dpi)
        print(f'fig is saved to {fig_path}')

    if interactive is True:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='plot learning curve')
    parser.add_argument('--train-record', dest='train_record_path', type=str,
                        required=True, nargs='+',
                        help='train record path')
    parser.add_argument('--label', dest='label', type=str, default=None,
                        nargs='+', help='label for each train record')
    parser.add_argument('--var-name', dest='var_name', type=str,
                        default='loss_record', help='')
    parser.add_argument('--linewidth', dest='linewidth', type=int,
                        default=2, help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, default=None,
                        help='')
    parser.add_argument('--interactive', dest='interactive', type=str,
                        default='false', choices=['true', 'false'], help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='dpi if figure')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot_train_curve(train_record_path=args.train_record_path,
                     label=args.label,
                     var_name=args.var_name,
                     linewidth=args.linewidth,
                     fig_path=args.fig_path,
                     dpi=args.dpi,
                     interactive=args.interactive == 'true')


if __name__ == '__main__':
    main()
