import numpy as np
import os
import argparse
from BasicTools.parse_file import file2dict, dict2file
from BasicTools.wav_tools import frame_data
from BasicTools.ProcessBar import ProcessBar


def list2str(x):
    return ' '.join(map(str, x))


def convert2list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x


def load_log(loc_log_path, vad_log_path, chunksize, result_dir=None,
             azi_pos=0, keep_sample_num=False, print_result=False):
    """load loc log

    """
    azi_pos_all = convert2list(azi_pos)  # allow multiple azi grandtruth

    # check if result_log exists
    log_name = os.path.basename(loc_log_path).split('.')[0]
    if result_dir is None:
        result_dir = os.path.dirname(os.path.dirname(loc_log_path))
    statistic_dir = (f'{result_dir}/'
                     + '-'.join((f'chunksize_{chunksize}',
                                 f'azipos_{list2str(azi_pos_all)}')))
    if vad_log_path is not None:
        statistic_dir = statistic_dir + '-vad'
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

    result_log = {}
    performance_log = {}
    feat_paths = list(loc_log.keys())
    pb = ProcessBar(feat_paths)
    for feat_path in feat_paths:
        pb.update()
        output = loc_log[feat_path]
        if chunksize > 1:
            if keep_sample_num:
                # padd chunksize-1 in the begining, as result, output will have
                # the same shape after framing and averaging
                output = np.pad(output, ((chunksize-1, 0), (0, 0)))
            output = np.mean(
                frame_data(output, frame_len=chunksize, frame_shift=1),
                axis=1)
        else:  # chunksize == 1
            if vad_log is not None:
                n_sample = min((vad_log[feat_path].shape[0], output.shape[0]))
                vad = vad_log[feat_path][-n_sample:]
                output = output[-n_sample:][np.where(vad == 1)]

        result_log[feat_path] = np.argmax(output, axis=1)
        n_sample = result_log[feat_path].shape[0]

        feat_name = os.path.basename(feat_path).split('.')[0]
        attrs = list(map(float, feat_name.split('_')))
        azi_grandtrue = np.asarray([attrs[i] for i in azi_pos_all])
        diff = np.min(
            np.abs(
                (result_log[feat_path][:, np.newaxis]
                 - azi_grandtrue[np.newaxis, :])),
            axis=1)
        if n_sample > 0:
            rmse = np.sqrt(np.sum(diff**2)/n_sample)
            cp = np.nonzero(diff < 1e-5)[0].shape[0]/n_sample
        else:
            rmse = -1
            cp = -1
        performance_log[feat_path] = [cp, rmse]

    # write to file
    dict2file(performance_log, statistic_log_path, item_format='.4f')
    dict2file(result_log, result_log_path, item_format='2d')

    # average over all feat files
    cp_mean, rmse_mean = np.mean(
        np.asarray(
            list(performance_log.values())),
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
    parser.add_argument('--azi-pos', dest='azi_pos', required=True, type=int,
                        nargs='+', default=0)
    parser.add_argument('--keep-sample-num', dest='keep_sample_num', type=str,
                        default='false', choices=['true', 'false'])
    parser.add_argument('--print-result', dest='print_result', type=str,
                        default='true', choices=['true', 'false'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    load_log(loc_log_path=args.loc_log_path,
             vad_log_path=args.vad_log_path,
             chunksize=args.chunksize,
             result_dir=args.result_dir,
             azi_pos=args.azi_pos,
             keep_sample_num=args.keep_sample_num == 'true',
             print_result=args.print_result == 'true')


if __name__ == '__main__':
    main()
