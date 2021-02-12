import numpy as np
import matplotlib.pyplot as plt
import argparse


def tolist(x):
    if not isinstance(x, list):
        return [x]
    elif x is None:
        return [None]
    else:
        return x


def plot_train_curve(train_record_path, label, fig_path,
                     var_name='loss_record', dpi=100):

    train_record_paths = tolist(train_record_path)
    labels = tolist(label)

    fig, ax = plt.subplots(1, 1)
    for train_record_path, label in zip(train_record_paths, labels):
        record_info = np.load(train_record_path)
        loss_record = record_info[var_name]
        n_epoch = np.max(np.nonzero(loss_record)[0])
        loss_record = loss_record[:n_epoch+1]
        ax.plot(loss_record, label=label)

    ax.set_xlabel('epoch')
    ax.set_ylabel(var_name)
    ax.legend()
    fig.savefig(fig_path, dpi=dpi)
    print(f'fig is saved to {fig_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='plot learning curve')
    parser.add_argument('--train-record', dest='train_record_path', type=str,
                        required=True, nargs='+',
                        help='train record path')
    parser.add_argument('--label', dest='label', type=str, default=None,
                        nargs='+', help='label for each train record')
    parser.add_argument('--var-name', dest='var_name', type=str,
                        default='loss_record', help='')
    parser.add_argument('--fig-path', dest='fig_path', type=str, required=True,
                        help='')
    parser.add_argument('--dpi', dest='dpi', type=int, default=100,
                        help='dpi if figure')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot_train_curve(train_record_path=args.train_record_path,
                     label=args.label,
                     fig_path=args.fig_path,
                     var_name=args.var_name,
                     dpi=args.dpi)


if __name__ == '__main__':
    main()
