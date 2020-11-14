import numpy as np
import os
import argparse
from BasicTools.parse_file import file2dict, dict2file
from BasicTools.wav_tools import frame_data


def list2str(x):
    return ' '.join(map(str, x))


def convert2list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x


def load_log(loc_log_path, vad_log_path, chunksize, chunkshift,
             result_dir=None, azi_pos=0, print_result=False):
    """"""
    azi_pos = convert2list(azi_pos)

    # check if result_log exists
    log_name = os.path.basename(loc_log_path).split('.')[0]
    if result_dir is None:
        result_dir = os.path.dirname(os.path.dirname(loc_log_path))
    statistic_dir = (f'{result_dir}/'
                     + '_'.join((f'chunksize-{chunksize}',
                                 f'chunkshift-{chunkshift}',
                                 f'azipos-{list2str(azi_pos)}')))
    if vad_log_path is not None:
        statistic_dir = statistic_dir + '_vad'
    os.makedirs(statistic_dir)

    statistic_log_path = f'{statistic_dir}/{log_name}.txt'
    if os.path.exists(statistic_log_path):
        raise FileExistsError(statistic_log_path)

    os.makedirs(f'{statistic_dir}/log')
    result_log_path = f'{statistic_dir}/log/{log_name}.txt'
    if os.path.exists(result_log_path):
        raise FileExistsError(result_log_path)

    loc_log = file2dict(loc_log_path, numeric=True)
    if vad_log_path is None:
        vad_log = None
    else:
        vad_log = file2dict(vad_log_path, numeric=True)
        vad_log = {key: np.squeeze(np.asarray(value))
                   for key, value in vad_log.items()}

    loc_result_all = {}
    performance_all = {}
    for feat_path in loc_log.keys():
        mean_output_chunk = np.mean(
            frame_data(loc_log[feat_path],
                       frame_len=chunksize,
                       shift_len=chunkshift),
            axis=1)
        if chunksize == 1 and vad_log is not None:
            mean_output_chunk = \
                mean_output_chunk[np.where(vad_log[feat_path] == 1)]

        loc_result_all[feat_path] = np.argmax(mean_output_chunk, axis=1)
        n_chunk = loc_result_all[feat_path].shape[0]

        feat_name = os.path.basename(feat_path).split('.')[0]
        conditions = list(map(float, feat_name.split('_')))
        azi_grandtrue = np.asarray([conditions[i] for i in azi_pos])
        diff = np.min(
            np.abs(
                (loc_result_all[feat_path][:, np.newaxis]
                 - azi_grandtrue[np.newaxis, :])),
            axis=1)
        if n_chunk > 0:
            rmse = np.sqrt(np.sum(diff**2)/n_chunk)
            cp = np.nonzero(diff < 1e-5)[0].shape[0]/n_chunk
        else:
            rmse = -1
            cp = -1
        performance_all[feat_path] = [cp, rmse]

    # write to file
    dict2file(performance_all, statistic_log_path, item_format='.4f')
    dict2file(loc_result_all, result_log_path, item_format='2d')

    # average over all feat files
    cp_mean, rmse_mean = np.mean(
        np.asarray(
            list(performance_all.values())),
        axis=0)
    print('average result')
    print(f'cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}')
    with open(statistic_log_path, 'a') as statistic_logger:
        statistic_logger.write('# average result\n')
        statistic_logger.write(f'# cp:{cp_mean:.4e} rmse:{rmse_mean:.4e}\n')


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--loc-log', dest='loc_log_path', required=True,
                        type=str, help='localization log path')
    parser.add_argument('--vad-log', dest='vad_log_path', type=str,
                        default=None, help='vad log path')
    parser.add_argument('--result-dir', dest='result_dir', type=str,
                        default=None, help='where to save result files')
    parser.add_argument('--chunksize', dest='chunksize', required=True,
                        type=int, help='sample number of a chunk')
    parser.add_argument('--chunkshift', dest='chunkshift', required=True,
                        type=int, help='sample shift between adjacent chunks')
    parser.add_argument('--azi-pos', dest='azi_pos', type=int,
                        nargs='+', default=0)
    parser.add_argument('--print-result', dest='print_result', type=str,
                        default='true', choices=['true', 'false'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    load_log(loc_log_path=args.loc_log_path,
             vad_log_path=args.vad_log_path,
             chunksize=args.chunksize,
             chunkshift=args.chunkshift,
             result_dir=args.result_dir,
             azi_pos=args.azi_pos,
             print_result=args.print_result == 'true')


if __name__ == '__main__':
    main()
