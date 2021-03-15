import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


from BasicTools.parse_file import file2dict
from BasicTools.wav_tools import frame_data


def list2str(x):
    if x is None:
        return None
    else:
        return '_'.join(map(str, x))


def cal_statistic(azi_gt, azi_est, theta=1):
    """ calculate CP and RMSE for a pair of azi_gt and azi_est
    both azi_gt and azi_est can be ndarray
    """
    n_sample, n_src = azi_est.shape
    cp = 0
    rmse = 0

    for src_i in range(n_src):
        diff_tmp = np.abs(
            azi_est[:, src_i, np.newaxis]-azi_gt[np.newaxis, :])
        min_diff = np.min(diff_tmp, axis=1)
        equality = min_diff < theta
        cp = cp + np.sum(equality.astype(np.int))
        rmse = rmse + np.sum(min_diff**2)

    n_est = n_sample * n_src  # number of location estimations, 1 for 1 source
    cp = cp/n_est
    rmse = np.sqrt(rmse/n_est)
    return cp, rmse


def get_n_max_pos(x, n):
    """ return the positions of n-max values in each column of x
    currently, no post process
    Args:
        x: [n_sample, n_ouput]
        n
    Return
    """
    n_max_pos = np.argsort(x, axis=1)[:, -n:]
    return n_max_pos
    # n_sample = x.shape[0]
    # n_max_pos = np.zeros((n_sample, n), dtype=np.int)
    # for sample_i in range(n_sample):
    #     tmp = x[sample_i, :]
    #     # maybe smooth
    #     n_max_pos[sample_i] = np.argsort(tmp,)
    # sort_index = np.argsort(x)
    # return sort_index[x]


def get_azi_gt_from_name(loc_log_path, azi_pos_all):
    azi_gt_log = {}
    loc_logger = open(loc_log_path, 'r')
    for line_i, line in enumerate(loc_logger):
        feat_path, _ = line.split(':')
        feat_name = os.path.basename(feat_path).split('.')[0]
        attrs = [float(item) for item in feat_name.split('_')]
        azi_gt_log[feat_path] = np.asarray([attrs[i] for i in azi_pos_all])
    loc_logger.close()
    return azi_gt_log


def load_loc_log(loc_log_path, file_name, result_dir, vad_log_path, chunksize,
                 n_src, reliability_log_path=None, azi_pos_all=None,
                 azi_gt_log_path=None, none_label=None):
    """load loc log
    """

    # prepare the directory
    log_name = os.path.basename(loc_log_path)
    if result_dir is None:
        result_dir = os.path.dirname(os.path.dirname(loc_log_path))
    result_dir = (f'{result_dir}/'
                  + '-'.join((f'chunksize_{chunksize}',
                              f'azipos_{list2str(azi_pos_all)}',
                              f'nsrc_{n_src}')))
    if vad_log_path is not None:
        result_dir = result_dir + '-vad'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f'{result_dir}/log', exist_ok=True)

    # log for CP and RMSE
    performace_log_path = f'{result_dir}/{log_name}'
    if os.path.exists(performace_log_path):
        raise FileExistsError(performace_log_path)
    # log for estimation result
    azi_est_log_path = f'{result_dir}/log/{log_name}'
    if os.path.exists(azi_est_log_path):
        raise FileExistsError(azi_est_log_path)

    # load vad log if vad_log_path is specified
    if vad_log_path is None:
        vad_log = None
    else:
        vad_log = file2dict(vad_log_path, numeric=True, squeeze=True)

    if reliability_log_path is not None:
        reliability_log = file2dict(reliability_log_path, numeric=True,
                                    squeeze=True)
    else:
        reliability_log = None

    loc_logger = open(loc_log_path, 'r')
    for line_i, line in enumerate(loc_logger):
        feat_path, y = line.split(':')
        if os.path.basename(feat_path).split('.')[0] != file_name:
            continue

        y = np.asarray([[np.float32(item) for item in row.split()]
                        for row in y.split(';')],
                       dtype=np.float32)
        n_sample = y.shape[0]

        if reliability_log is not None:
            reliability = reliability_log[feat_path]
        else:
            reliability = np.ones(n_sample)

        invalid_flags = np.zeros(n_sample, dtype=np.bool)
        if vad_log is not None:
            vad = np.bool(vad_log[feat_path])
            n_sample = np.min([y.shape[0], vad.shape[0]])
            y = y[:n_sample]
            reliability = reliability[:n_sample]
            invalid_flags = invalid_flags[:n_sample]
            vad = vad[:n_sample]
            invalid_flags[vad == 0] = True
        if none_label is not None:
            tmp = np.argmax(y, axis=1)
            invalid_flags[tmp == none_label] = True
            y[:, none_label] = 0

        y[invalid_flags, :] = 0

        y_chunk = frame_data(y, frame_len=chunksize, frame_shift=1)
        y_chunk_combined = np.mean(y_chunk, axis=1)

        reliability_chunk = frame_data(reliability, frame_len=chunksize,
                                       frame_shift=1)

        n_chunk = y_chunk.shape[0]
        for chunk_i in range(n_chunk):
            fig, ax = plt.subplots(2, 2, tight_layout=True)
            ax[0, 0].imshow(y_chunk[chunk_i], aspect='auto')
            ax[0, 1].plot(reliability_chunk[chunk_i])
            ax[0, 1].set_title(np.mean(reliability_chunk[chunk_i]))
            ax[1, 0].plot(y_chunk_combined[chunk_i])
            ax[1, 0].set_title(f'{np.argmax(y_chunk_combined[chunk_i])}')
            fig.savefig(f'{result_dir}/{chunk_i}.png')
            plt.close()

        azi_est = get_n_max_pos(y_chunk_combined, n_src)

        invalid_frame_num = np.sum(
            frame_data(invalid_flags, frame_len=chunksize, frame_shift=1),
            axis=1)
        invalid_chunk_flag = invalid_frame_num >= chunksize  # TODO
        azi_est[invalid_chunk_flag, :] = -1*np.ones(n_src)

        fig, ax = plt.subplots(1, 1)
        ax.plot(azi_est)
        fig.savefig(f'{result_dir}/azi_est.png')
        plt.close()


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--loc-log', dest='loc_log_path', required=True,
                        type=str, help='localization log path')
    parser.add_argument('--reliability-log', dest='reliability_log_path',
                        default=None, type=str, help='localization log path')
    parser.add_argument('--file-name', dest='file_name', type=str,
                        help='vad log path')
    parser.add_argument('--vad-log', dest='vad_log_path', type=str,
                        default=None, help='vad log path')
    parser.add_argument('--result-dir', dest='result_dir', type=str,
                        help='where to save result files')
    parser.add_argument('--azi-pos', dest='azi_pos', type=int, nargs='+',
                        default=None)
    parser.add_argument('--none-label', dest='none_label', type=int,
                        default=None)
    parser.add_argument('--azi-gt-log', dest='azi_gt_log_path', type=str,
                        default=None)
    parser.add_argument('--chunksize', dest='chunksize', required=True,
                        type=int, help='sample number of a chunk')
    parser.add_argument('--n-src', dest='n_src', required=True, type=int,
                        default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    load_loc_log(loc_log_path=args.loc_log_path,
                 reliability_log_path=args.reliability_log_path,
                 file_name=args.file_name,
                 vad_log_path=args.vad_log_path,
                 chunksize=args.chunksize,
                 n_src=args.n_src,
                 none_label=args.none_label,
                 azi_pos_all=args.azi_pos,
                 azi_gt_log_path=args.azi_gt_log_path,
                 result_dir=args.result_dir)


if __name__ == '__main__':
    main()
