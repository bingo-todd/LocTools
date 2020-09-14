import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
# from BasicTools import plot_tools


def plot_train_curve(model_dir, model_label, fig_path, room=None):

    plt.rcParams.update({"font.size": "10"})
    # linestyle_all = ('solid', 'dashed', 'dotted')
    plot_settings = {'linewidth': 3}   # , 'color':'black'}

    if not isinstance(model_dir, list):
        model_dir_all = [model_dir]
        model_labels = [model_labels]
    else:
        model_dir_all = model_dir
        model_labels = model_label

    if room is None:
        rooms = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
    elif not isinstance(room, list):
        rooms = [room]
    else:
        rooms = room
    n_room = len(rooms)

    fig, ax = plt.subplots(1, n_room, tight_layout=True, sharex=True, sharey=True)
    if n_room == 1:
        ax = [ax]
    for model_dir, model_label in zip(model_dir_all, model_labels):
        for room_i, room in enumerate(rooms):
            record_path = f'{model_dir}/{room}/train_record.npz'
            if os.path.exists(record_path):
                record_info = np.load(record_path)
                try:
                    loss_record = record_info['loss_record']
                except Exception:
                    loss_record = record_info['loss_loc_record']
                n_epoch = np.nonzero(loss_record)[0][-1] + 1
                ax[room_i].plot(loss_record[:n_epoch], label=model_label, **plot_settings)
            else:
                print(f'{record_path} not exists')

    for room_i, room in enumerate(rooms):
        ax[room_i].set_xlabel('Epoch')
        ax[room_i].set_title(room)

    ax[0].set_ylabel('loss')
    ax[-1].legend()
    fig.savefig(fig_path)


def parse_args():
    parser = argparse.ArgumentParser(description='plot learning curve')
    parser.add_argument('--model-dir', type=str, dest='model_dir',
            required=True, action='append', help='base dir of mct model')
    parser.add_argument('--model-label', type=str, dest='model_label', 
            required=True, action='append', help='model type')
    parser.add_argument('--fig-path', type=str, dest='fig_path', 
            default='train_process.png', help='figure path')
    parser.add_argument('--rooms', type=str, dest='rooms', default=None, 
            action='append', help='room name, subfolder name of model-dir')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot_train_curve(args.model_dir, args.model_label, args.fig_path, args.rooms)


if __name__ == '__main__':
    main()
