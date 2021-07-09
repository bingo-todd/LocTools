import numpy as np
import os
import argparse
from BasicTools.parse_file import file2dict, dict2file
from BasicTools.wav_tools import frame_data
from BasicTools.easy_parallel import easy_parallel

from .split_file import split_file
from .combine_files import combine_files


def list2str(x):
    if x is None:
        return None
    else:
        return '_'.join(map(str, x))


def cal_statistic(azi_gt, azi_est, circule, n_azi, theta=1):
    """ calculate CP and RMSE for a pair of azi_gt and azi_est
    both azi_gt and azi_est can be ndarray
    """
    n_sample, n_src = azi_est.shape
    cp = 0
    rmse = 0

    for src_i in range(n_src):
        diff = np.abs(azi_est[:, src_i, np.newaxis]-azi_gt[np.newaxis, :])
        if circule:
            diff = np.expand_dims(diff, axis=-1)
            diff = np.min(
                np.concatenate((diff, n_azi-diff), axis=-1),
                axis=-1)
        min_diff = np.min(diff, axis=1)
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


def make_azi_decision(y, vad, none_label, chunksize, n_src):
    n_frame = y.shape[0]
    invalid_flags = np.zeros(n_frame, dtype=np.bool)
    if vad is not None:
        invalid_flags[vad == 0] = True
    if none_label is not None:
        tmp = np.argmax(y, axis=1)
        invalid_flags[tmp == none_label] = True
        y[:, none_label] = 0

    y[invalid_flags, :] = 0
    y_chunk = np.mean(
        frame_data(y, frame_len=chunksize, frame_shift=1),
        axis=1)
    azi_est = get_n_max_pos(y_chunk, n_src)

    return azi_est, invalid_flags


def load_loc_log(loc_log_path, chunksize, n_src, vad_log_path=None,
                 azi_pos_all=None, azi_gt_log_path=None,
                 none_label=None, circule=False, n_azi=None, result_dir=None,
                 keep_sample_num=False, print_result=False):
    """load loc log
    Args:
        loc_log_path:
        chunksize:
        n_src: specifiy how many source position should be estimated from each
            frame
        vad_log_path:
        azi_pos_all: specify which pars in file names are azimuth labels
        azi_gt_log_path: log file containing azimuth labels of file in loc_log
        none_label: label for unvoiced frames
        circule: whether azi_pos_all form a circule
        n_azi: number of sound azimuth
        result_dir: where to store result files
        keep_sample_num: where to padd in the front of model output to ensure
            there are azimuth estimations for each frame no matter how much the
            chunksize is specified
        print_result: print result to the terminal
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
    statistic_log_path = f'{result_dir}/{log_name}'
    if os.path.exists(statistic_log_path):
        raise FileExistsError(statistic_log_path)
    # log for estimation result
    azi_est_log_path = f'{result_dir}/log/{log_name}'
    if os.path.exists(azi_est_log_path):
        raise FileExistsError(azi_est_log_path)

    # load vad log if vad_log_path is specified
    if vad_log_path is None:
        vad_log = None
    else:
        vad_log = file2dict(vad_log_path, numeric=True, squeeze=True)

    # get azi_gt_log either from feat_path or azi_gt_log_path
    if azi_pos_all is not None:
        azi_gt_log = get_azi_gt_from_name(loc_log_path, azi_pos_all)
    elif azi_gt_log_path is not None:
        azi_gt_log = file2dict(azi_gt_log_path, numeric=True, squeeze=True)
    else:
        raise Exception('either azi_pos_all or azi_gt_log_path is specified')

    azi_est_log = {}
    statistic_log = {}
    sample_num_log = {}
    loc_logger = open(loc_log_path, 'r')
    for line_i, line in enumerate(loc_logger):
        feat_path, output = line.split(':')
        y = np.asarray([[np.float32(item) for item in row.split()]
                        for row in output.split(';')],
                       dtype=np.float32)
        n_frame = y.shape[0]
        # get vad
        if vad_log is not None:
            try:
                vad = vad_log[feat_path][-n_frame]
            except Exception as e:
                print(feat_path)
                raise Exception(e)
        else:
            vad = None
        # padd to ensure chunk number is the same with frames
        if chunksize > 1 and keep_sample_num:
            y = np.pad(y, ((chunksize-1, 0), (0, 0)))
        # make azimuth decision
        azi_est, invalid_flags = make_azi_decision(y, vad, none_label,
                                                   chunksize, n_src)
        # set azimuth estimation to -1 if it has too many invalid frames
        invalid_frame_num = np.sum(
            frame_data(invalid_flags, frame_len=chunksize, frame_shift=1),
            axis=1)
        invalid_chunk_flags = invalid_frame_num >= chunksize  # TODO
        azi_est[invalid_chunk_flags, :] = -1*np.ones(n_src)
        azi_est_log[feat_path] = azi_est
        #
        azi_est = azi_est[np.logical_not(invalid_chunk_flags), :]
        cp, rmse = cal_statistic(
            azi_gt_log[feat_path], azi_est, circule, n_azi)
        statistic_log[feat_path] = [[cp, rmse]]
        sample_num_log[feat_path] = azi_est.shape[0]
    loc_logger.close()

    # write to file
    dict2file(statistic_log, statistic_log_path, item_format='.4f')
    dict2file(azi_est_log, azi_est_log_path, item_format='2d')

    # average over all feat files
    cp_mean, rmse_mean, sample_num = 0, 0, 0
    for feat_path in statistic_log.keys():
        sample_num_tmp = sample_num_log[feat_path]
        cp_tmp, rmse_tmp = statistic_log[feat_path][0]
        cp_mean = cp_mean + sample_num_tmp*cp_tmp
        rmse_mean = rmse_mean + sample_num_tmp*rmse_tmp**2
        sample_num = sample_num + sample_num_tmp
    cp_mean, rmse_mean = cp_mean/sample_num, np.sqrt(rmse_mean/sample_num)

    with open(statistic_log_path, 'a') as statistic_logger:
        statistic_logger.write('# average result\n')
        statistic_logger.write(f'# cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}\n')
    return result_dir


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--loc-log', dest='loc_log_path', required=True,
                        type=str, help='localization log path')
    parser.add_argument('--vad-log', dest='vad_log_path', type=str,
                        default=None, help='vad log path')
    parser.add_argument('--result-dir', dest='result_dir', type=str,
                        default=None, help='where to save result files')
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
    parser.add_argument('--circule', dest='circule', type=str, default='false',
                        choices=['true', 'false'])
    parser.add_argument('--n-azi', dest='n_azi', type=int, default=None)
    parser.add_argument('--keep-sample-num', dest='keep_sample_num', type=str,
                        default='false', choices=['true', 'false'])
    parser.add_argument('--print-result', dest='print_result', type=str,
                        default='true', choices=['true', 'false'])
    parser.add_argument('--n-part', dest='n_part', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    if args.n_part > 0:
        part_loc_log_paths = split_file(args.loc_log_path, n_part=args.n_part,
                                        exist_ok=True)
        log_name = os.path.basename(args.loc_log_path)
        tasks = []
        for part_loc_log_path in part_loc_log_paths:
            tasks.append([part_loc_log_path, args.vad_log_path, args.chunksize,
                          args.n_src, args.azi_pos, args.azi_gt_log_path,
                          args.none_label, args.circule == 'true', args.n_azi,
                          args.result_dir, args.keep_sample_num == 'true',
                          args.print_result == 'true'])
        statistic_dir_paths = easy_parallel(load_loc_log, tasks, len(tasks))
        # statistic_dir_paths = []
        # for task in tasks:
        #     statistic_dir_path = load_log(*task)
        #     statistic_dir_paths.append(statistic_dir_path)

        combine_files(statistic_dir_paths[0], keep_part_file=False,
                      keep_comment=False)
        # calculate overall mean
        statistic_log_path = f'{statistic_dir_paths[0]}/{log_name}'
        statistic_log = file2dict(statistic_log_path, numeric=True)
        cp_mean, rmse_mean = np.mean(
            np.concatenate(
                list(statistic_log.values()),
                axis=0),
            axis=0)
        with open(statistic_log_path, 'a') as statistic_logger:
            statistic_logger.write('# average result\n')
            statistic_logger.write(
                f'# cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}\n')

        combine_files(f'{statistic_dir_paths[0]}/log', keep_part_file=False,
                      keep_comment=False)
        # if loc_log is very large, spliting can be very time-consuming, so
        # there file parts will not be deleted automatically. if you are
        # certain these file parts are no longer need, you can deleted them
        # mannually
        # remote part file of loc_log
        # for part_loc_log_path in part_loc_log_paths:
        #     os.remove(part_loc_log_path)
    else:
        load_loc_log(loc_log_path=args.loc_log_path,
                     vad_log_path=args.vad_log_path,
                     chunksize=args.chunksize,
                     n_src=args.n_src,
                     none_label=args.none_label,
                     azi_pos_all=args.azi_pos,
                     azi_gt_log_path=args.azi_gt_log_path,
                     circule=args.circule == 'true',
                     n_azi=args.n_azi,
                     result_dir=args.result_dir,
                     keep_sample_num=args.keep_sample_num == 'true',
                     print_result=args.print_result == 'true')


if __name__ == '__main__':
    main()
