import numpy as np
import os
import argparse
from BasicTools.parse_file import file2dict
from BasicTools.wav_tools import frame_data
from BasicTools.ProcessBar import ProcessBar
from .add_log import add_log


def list2str(x):
    return ' '.join(map(str, x))


def convert2list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x


def load_log(est_log_path, gt_log_path, result_dir=None, chunksize=1,
             print_result=False, cal_cp=True, cal_rmse=True, end_align=True,
             show_process=True):
    """"""

    est_log = file2dict(est_log_path, dtype=float)
    gt_log = file2dict(gt_log_path, dtype=float)

    statistic_dir = os.path.join(
        os.path.dirname(os.path.dirname(est_log_path)),
        'statistic')

    est_log_name = os.path.basename(est_log_path)
    statistic_log_path = f'{statistic_dir}/log/{est_log_name}.txt'
    os.makedirs(os.path.dirname(statistic_log_path), exist_ok=True)
    statistic_logger = open(statistic_log_path, 'x')
    statistic_logger.write('# key: CP RMSE\n')
    statistic_logger.write(f'# {chunksize}\n')
    cp_all = []
    rmse_all = []
    if show_process:
        pb = ProcessBar(len(list(est_log.keys())))
    for key in est_log.keys():
        if show_process:
            pb.update()

        y_est = np.asarray(est_log[key])
        if end_align:
            y_gt = np.asarray(gt_log[key])[-y_est.shape[0]:]
        else:
            y_gt = np.asarray(gt_log[key])[:y_est.shape[0]]

        y_est = frame_data(y_est, frame_len=chunksize, shift_len=chunksize)
        y_gt = frame_data(y_gt, frame_len=chunksize, shift_len=chunksize)

        n_sample = y_est.shape[0]  # row: n_sample; column: var
        diff_squre = np.sum((y_est-y_gt)**2, axis=1)
        cp = -1
        rmse = -1
        if cal_cp:
            equality = np.equal(diff_squre, 0)
            n_eq = np.where(equality)[0].shape[0]
            cp = n_eq/n_sample
        if cal_rmse:
            rmse = np.sqrt(np.mean(diff_squre))
        add_log(statistic_logger, key, [[cp, rmse]])

        cp_all.append(cp)
        rmse_all.append(rmse)

    # average over all feat files
    cp_mean, rmse_mean = np.mean(cp_all), np.mean(rmse_all)
    statistic_logger.write(f'# cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}\n')
    print('average result')
    print(f'# cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}')


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--est-log-path', dest='est_log_path', required=True,
                        type=str, help='the path of estimation log')
    parser.add_argument('--gt-log-path', dest='gt_log_path', type=str,
                        default=None, help='the path of grandtruth log')
    parser.add_argument('--result-dir', dest='result_dir', type=str,
                        default=None, help='where to save result files')
    parser.add_argument('--chunksize', dest='chunksize', type=int,
                        default=1, help='sample number of a chunk')
    parser.add_argument('--cal-cp', dest='cal_cp', type=str,
                        default='true', choices=['true', 'false'], help='')
    parser.add_argument('--cal-rmse', dest='cal_rmse', type=str,
                        default='true', choices=['true', 'false'], help='')
    parser.add_argument('--end-align', dest='end_align', type=str,
                        default='true', choices=['true', 'false'], help='')
    parser.add_argument('--show-process', dest='show_process', type=str,
                        default='true', choices=['true', 'false'], help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    load_log(est_log_path=args.est_log_path,
             gt_log_path=args.gt_log_path,
             result_dir=args.result_dir,
             chunksize=args.chunksize,
             cal_cp=args.cal_cp == 'true',
             cal_rmse=args.cal_rmse == 'true',
             end_align=args.end_align == 'true',
             show_process=args.show_process == 'true')


if __name__ == '__main__':
    main()
