import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# from BasicTools import plot_tools

plt.rcParams.update({"font.size": "10"})
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']

linestyle_all = ('solid', 'dashed', 'dotted')
plot_settings = {'linewidth': 3}   # , 'color':'black'}


def plot_train_process(record_fpath, ax=None, room=None, label=None):
    record_info = np.load(record_fpath)
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


    # ax.set_ylim([0.5, 1.])


def main(model_info, fig_fpath='train_process.png'):
    fig, ax = plt.subplots(1, 4,  figsize=(10, 4),
                           tight_layout=True, sharex=True, sharey=True)

    n_model =np.int32(len(model_info) / 2)
    for model_i in range(n_model):
        for room_i, room in enumerate(reverb_room_all):
            model_dir = model_info[model_i*2]
            label = model_info[model_i*2+1]
            record_fpath = f'{model_dir}/{room}/train_record.npz'
            print(record_fpath)
            if os.path.exists(record_fpath):
                plot_train_process(record_fpath, ax[room_i], room, label)
            else:
                print('not exists')

    for room_i, room in enumerate(reverb_room_all):
        ax[room_i].set_xlabel('Epoch')
        ax[room_i].set_title(room)

    ax[0].set_ylabel('cost')
    ax[-1].legend()

    fig.savefig(fig_fpath)


if __name__ == '__main__':
    main(sys.argv[1:-1], fig_fpath=sys.argv[-1])
