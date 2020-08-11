import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
# from BasicTools import plot_tools


def plot_train_process(record_path, ax=None, room=None, label=None):

    plt.rcParams.update({"font.size": "10"})
    # linestyle_all = ('solid', 'dashed', 'dotted')
    plot_settings = {'linewidth': 3}   # , 'color':'black'}

    record_info = np.load(record_path)
    key_all = list(record_info.keys())
    if 'cost_loc_record_valid' in key_all:
        cost_record = record_info['cost_loc_record_valid']
    elif 'cost_loc_record' in key_all:
        cost_record = record_info['cost_loc_record']
    elif 'loss_record' in key_all:
        cost_record = record_info['loss_record']
    elif 'loss_loc_record' in key_all:
        cost_record = record_info['loss_loc_record']
    else:
        print(key_all)
        raise Exception()

    n_epoch = np.nonzero(cost_record)[0][-1] + 1
    ax.plot(cost_record[:n_epoch], label=label, **plot_settings)


def plot_train_curve(args):
    n_room = len(args.rooms)
    fig, ax = plt.subplots(1, n_room,  figsize=(10, 4),
                           tight_layout=True, sharex=True, sharey=True)

    if not isinstance(args.model_dir, list):
        model_dir_all = [args.model_dir]
        model_label_all = [args.model_label]
    else:
        model_dir_all = args.model_dir
        model_label_all = args.model_label

    for model_dir, model_label in zip(model_dir_all, model_label_all):
        for room_i, room in enumerate(args.rooms):
            record_path = f'{model_dir}/{room}/train_record.npz'
            print(record_path)
            if os.path.exists(record_path):
                plot_train_process(record_path, ax[room_i], room, model_label)
            else:
                print('not exists')

    for room_i, room in enumerate(args.rooms):
        ax[room_i].set_xlabel('Epoch')
        ax[room_i].set_title(room)

    ax[0].set_ylabel('cost')
    ax[-1].legend()

    fig.savefig(args.fig_path)


def parse_args():
    parser = argparse.ArgumentParser(description='plot learning curve')
    parser.add_argument('--model-dir', type=str, dest='model_dir',
                        action='append', help='base dir of mct model')
    parser.add_argument('--model-label', type=str, dest='model_label',
                        action='append', help='model type')
    parser.add_argument('--fig-path', type=str, dest='fig_path',
                        default='train_process.png', help='figure path')
    parser.add_argument('--rooms', type=list, dest='rooms',
                        default=['Room_A', 'Room_B', 'Room_C', 'Room_D'],
                        help='room name, subfolder name of model-dir')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    plot_train_curve(args)
