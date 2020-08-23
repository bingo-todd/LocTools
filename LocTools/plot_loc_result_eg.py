import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
azi_diff_all = [-8, -6, -4, -2, 1, 1, 2, 4, 6, 8]


def plot_loc_result_eg(log_path, fig_path, file_name_required):
    with open(log_path) as log_file:
        line_all = log_file.readlines()
    
    for line_i, line in enumerate(line_all):
        file_i_str, file_path, *prob_frame_str_all = line.strip().split(';')
        file_name = os.path.basename(file_path)
        if file_name[:-4] == file_name_required:
            prob_frame_all = np.asarray([list(map(float, item.split())) 
                                         for item in prob_frame_str_all])
            fig, ax = plt.subplots(1, 1)
            ax.imshow(prob_frame_all.T, aspect='auto', cmap='jet')
            ax.set_xlabel('frame')
            ax.set_ylabel('azimuth')
            ax.set_title(f'{os.path.basename(file_path)}')
            fig.savefig(fig_path)
            plt.close(fig)
            print(f'fig is saved to {fig_path}')
            return True
        
    # file_name_required is not found in log
    print(f'{file_name_required} is not found in log')
    return False


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log-path', dest='log_path',
                        required=True, type=str, help='log file path')
    parser.add_argument('--fig-path', dest='fig_path', required=True,
                        type=str, help='where figure will be saved')
    parser.add_argument('--file-name', dest='file_name_required', 
                        type=str, help='where figure will be saved')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    plot_loc_result_eg(args.log_path, args.fig_path, args.file_name_required)
