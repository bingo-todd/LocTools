import numpy as np
import os
import matplotlib.pyplot as plt
azi_diff_all = [-8, -6, -4, -2, 1, 1, 2, 4, 6, 8]


def plot_loc_result_eg(log_path, fig_path):
    with open(log_path) as log_file:
        line_all = log_file.readlines()
    
    for line_i, line in enumerate(line_all):
        file_i_str, file_path, *prob_frame_str_all = line.strip().split(';')
        tar_azi, wav_i = map(int, os.path.basename(file_path)[:-4].split('_'))

        if tar_azi == 17 and wav_i == 0: 
            prob_frame_all = np.asarray([list(map(float, item.split())) for item in prob_frame_str_all])
            fig, ax = plt.subplots(1, 1)
            ax.imshow(prob_frame_all.T, aspect='auto', cmap='jet')
            ax.set_title(f'tar:{tar_azi}')
            fig.savefig(fig_path)
            plt.close(fig)


if __name__ == "__main__":
    room = 'Room_A'
    for snr in [0, 10, 20]:
        log_path = f'models_1task/mct/local_log/{room}_{snr}_1_0.txt'
        fig_path = f'img/loc_result_eg/1task_{room}_{snr}.png'
        plot_loc_result_eg(log_path, fig_path)

        log_path = f'models_2task/mct_0.8/local_log/{room}_{snr}_1_0.txt'
        fig_path = f'img/loc_result_eg/2task_{room}_{snr}.png'
        plot_loc_result_eg(log_path, fig_path)