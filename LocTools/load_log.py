import numpy as np
import os
import argparse
from BasicTools.parse_file import file2dict, dict2file
from BasicTools.ProcessBar import ProcessBar


def list2str(x):
    return ' '.join(map(str, x))


def convert2list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x


def load_log(est_log_path, gt_log_path, result_dir=None, print_result=False,
             cal_cp=True, cal_rmse=True, end_align=True, show_process=True):
    """"""

    if result_dir is None:
        result_dir = os.path.dirname(os.path.dirname(est_log_path))
    statistic_dir = f'{result_dir}/chunksize1'
    os.makedirs(statistic_dir)
    est_log_name = os.path.basename(est_log_path)[:-4]
    statistic_log_path = f'{statistic_dir}/{est_log_name}.txt'
    if os.path.exists(statistic_log_path):
        raise FileExistsError(statistic_log_path)

    os.makedirs(f'{statistic_dir}/log')
    result_log_path = f'{statistic_dir}/log/{est_log_name}.txt'
    if os.path.exists(result_log_path):
        raise FileExistsError(result_log_path)

    est_log = file2dict(est_log_path, numeric=True, repeat_processor='none')
    gt_log = file2dict(gt_log_path, numeric=True, repeat_processor='none')

    result_log = {}
    statistic_log = {}
    pb = ProcessBar(len(list(est_log.keys())))
    for key in est_log.keys():
        pb.update()
        result_log[key] = np.expand_dims(
            np.argmax(est_log[key], axis=1),
            axis=1)
        n_sample = result_log[key].shape[0]

        diff = np.abs((result_log[key] - gt_log[key]))
        rmse = np.sqrt(np.sum(diff**2)/n_sample)
        cp = np.nonzero(diff < 1e-5)[0].shape[0]/n_sample
        statistic_log[key] = [[cp, rmse]]

    # write to file
    dict2file(result_log, result_log_path, item_format='2d')
    dict2file(statistic_log, statistic_log_path, item_format='.4f')

    # average over all feat files
    cp_mean, rmse_mean = np.mean(
        np.squeeze(np.asarray(list(statistic_log.values()))),
        axis=0)
    print('average result')
    print(f'cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}')
    with open(statistic_log_path, 'a') as statistic_logger:
        statistic_logger.write('# average result\n')
        statistic_logger.write(f'# cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}\n')


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--est-log-path', dest='est_log_path', required=True,
                        type=str, help='the path of estimation log')
    parser.add_argument('--gt-log-path', dest='gt_log_path', type=str,
                        default=None, help='the path of grandtruth log')
    parser.add_argument('--vad-log-path', dest='vad_log_path', type=str,
                        default=None, help='the path of grandtruth log')
    parser.add_argument('--result-dir', dest='result_dir', type=str,
                        default=None, help='where to save result files')
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
             cal_cp=args.cal_cp == 'true',
             cal_rmse=args.cal_rmse == 'true',
             end_align=args.end_align == 'true',
             show_process=args.show_process == 'true')


if __name__ == '__main__':
    main()
