import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import MaxNLocator
from BasicTools.normalize import normalize


def plot_loc_result_eg(log_path, file_name_required, fig_path, dpi=100, is_normalize=False):
    with open(log_path) as log_file:
        line_all = log_file.readlines()
    
    for line_i, line in enumerate(line_all):
        file_path, *prob_frame_str_all = line.strip().split(';')
        file_name = os.path.basename(file_path).split('.')[0]
        if file_name == file_name_required:
            prob_frame_all = np.asarray([list(map(float, item.split())) 
                                         for item in prob_frame_str_all])
            if is_normalize:
                prob_frame_all = normalize(prob_frame_all, axis=1)

            fig = plt.figure()
            space = 0.01
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.8
            rect_image = [left, bottom, width, height]
            rect_line = [left+width+space, bottom, 0.15, height]
            ax_image = plt.axes(rect_image)
            ax_image.imshow(prob_frame_all.T, aspect='auto', cmap='jet', origin='lower')
            ax_image.set_xlabel('frame')
            ax_image.yaxis.set_major_locator(MaxNLocator(integer=True))

            # ax_image.set_ylabel('azimuth')
            ax_image.set_title(f'{os.path.basename(file_path)}')

            ax_line = plt.axes(rect_line, sharey=ax_image)
            view_base = plt.gca().transData
            view_rot = transforms.Affine2D().rotate_deg(90)
            result_len = prob_frame_all.shape[1]
            ax_line.plot(prob_frame_all.T, transform=view_rot+view_base)
            plt.setp(ax_line.get_yticklabels(), visible=False)
            ax_line.spines['top'].set_visible(False)
            ax_line.spines['right'].set_visible(False)

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
    parser.add_argument('--fig-path', dest='fig_path', 
            required=True, type=str, help='where figure will be saved')
    parser.add_argument('--dpi', dest='dpi', type=int, 
            default=100, help='')
    parser.add_argument('--file-name', dest='file_name_required', 
            required=True, type=str, help='which file to be plot')
    parser.add_argument('--is-normalize', dest='is_normalize', 
                        type=bool, default=False, help='normalize along the 2th dimentions')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    plot_loc_result_eg(args.log_path, args.file_name_required, 
            args.fig_path, args.dpi, args.is_normalize)
