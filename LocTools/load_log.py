import numpy as np
import os
import argparse
import pandas


def list2str(x, separator='_'):
    return separator.join(map(str, x))


def load_log(log_path, fig_path=None):
    with open(log_path) as log_file:
        lines = log_file.readlines()
        # remove line start with #
        lines = filter(lambda line: line.strip()[0] != '#', lines)

    loc_info_file_all = []
    for line_i, line in enumerate(lines):
        file_i_str, file_path, *prob_frame_str_all = line.strip().split(';')  #
        conditions = list(
            map(int,
                os.path.basename(file_path)[:-4].strip().split('_')))
        prob_frame_all = np.asarray(
            [list(map(float, item.strip().split())) for item in prob_frame_str_all])
        loc_info_file_all.append([conditions, prob_frame_all])
    return loc_info_file_all


def decision_maker(prob_frame, chunk_size):
    n_sample = prob_frame.shape[0]
    n_chunk = n_sample - chunk_size + 1
    azi = np.zeros(n_chunk)
    for chun_i in range(n_chunk):
        chunk_slice = slice(chun_i, chun_i+chunk_size)
        azi[chun_i] = np.argmax(np.mean(prob_frame[chunk_slice], axis=0))
    return azi


def perform_measure(result_log_path, statistic_log_path, chunk_size,
                    azi_index=0):
    if not isinstance(azi_index, list):
        azi_index = [azi_index]
    
    if not os.path.exists(result_log_path):
        print(f'{result_log_path} do not exist')
        return 0, 0

    loc_info_file_all = load_log(result_log_path)

    os.makedirs(os.path.dirname(statistic_log_path), exist_ok=True)
    logger = open(statistic_log_path, 'w')
    cp = 0
    mse = 0
    n_sample = 0
    for loc_info_file in loc_info_file_all:
        conditions, prob_frame_all = loc_info_file
        azi_est = decision_maker(prob_frame_all, chunk_size)
        n_sample_file = azi_est.shape[0]

        src_azi_all = np.asarray([conditions[i] for i in azi_index])

        success_loc = np.logical_or.reduce(
            [azi_est == src_azi for src_azi in src_azi_all])
        cp_file = np.nonzero(success_loc)[0].shape[0]

        mse_file = np.sum(
            np.min(
                np.abs(azi_est[:, np.newaxis] - src_azi_all[np.newaxis, :]),
                axis=1))

        conditions_str = ' '.join([f'{item}' for item in conditions])
        azi_est_str = ' '.join(list(map(str, azi_est)))
        logger.write('; '.join((f'{conditions_str}',
                                f'{cp_file/n_sample_file:.2f}',
                                f'{mse_file/n_sample_file:.2f}',
                                f'{azi_est_str}\n')))

        # mean across all azi
        cp = cp + cp_file
        mse = mse + mse_file
        n_sample = n_sample + n_sample_file

    cp = cp/n_sample
    mse = mse/n_sample * args.azi_resolution
    logger.write(f'mean \n\t mse:{mse} \n\t cp:{cp}')

    return mse, cp


def main(args):

    n_room = len(args.room_all)
    n_snr = len(args.snr_all)
    
    if not isinstance(args.run_id_all, list):
        run_id_all = [args.run_id_all] 
    else:
        run_id_all = args.run_id_all
    n_run = len(run_id_all)

    result_all = np.zeros((n_room, n_snr, n_run, 2))
    statistic_dir = ''.join((
        f'{args.model_dir}/statistic_log/',
        f'chunksize{args.chunk_size}_srcaziindex{list2str(args.azi_index)}'))
    for room_i, room in enumerate(args.room_all):
        for snr_i, snr in enumerate(args.snr_all):
            for run_i, run_id in enumerate(run_id_all):
                result_all[room_i, snr_i, run_i] = perform_measure(
                    f'{args.model_dir}/loc_log/{room}_{snr}_{run_id}.txt',
                    f'{statistic_dir}/{room}_{snr}_{run_id}.txt',
                    args.chunk_size,
                    args.azi_index)
    # average across all run
    print(f'average across run {run_id_all}')
    result_all = np.mean(result_all, axis=2)
    if args.print_result:
        print('mse')
        print(result_all[:, :, 0])
        print('cp')
        print(result_all[:, :, 1])
    # save to csv
    data_frame = pandas.DataFrame(
        columns=['Measure', 'Room', '0dB', '10dB', '20dB'])
    row_mse_all = []
    row_cp_all = []
    for room_i, room in enumerate(args.room_all):
        tmp = {'Room': room}
        row_mse_tmp = {f'{snr}dB': result_all[room_i, snr_i, 0]
                       for snr_i, snr in enumerate(args.snr_all)}
        row_mse_all.append({**tmp, **row_mse_tmp})

        row_cp_tmp = {f'{snr}dB': result_all[room_i, snr_i, 1]
                      for snr_i, snr in enumerate(args.snr_all)}
        row_cp_all.append({**tmp, **row_cp_tmp})

    data_frame = data_frame.append({'Measure': 'mse'}, ignore_index=True)
    data_frame = data_frame.append(row_mse_all, ignore_index=True)

    data_frame = data_frame.append({'Measure': 'cp'}, ignore_index=True)
    data_frame = data_frame.append(row_cp_all, ignore_index=True)

    csv_path = ''.join((
        f'{args.model_dir}/result_csv/',
        '_'.join((
            f'runid{list2str(args.run_id_all)}',
            f'chunksize{args.chunk_size}',
            f'aziindex{list2str(args.azi_index)}.csv'))))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    data_frame.to_csv(csv_path, index=False, index_label=None)


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model-dir', dest='model_dir',
                        required=True, type=str, help='directory of model')
    parser.add_argument('--chunk-size', dest='chunk_size', required=True,
                        type=int, help='sample number of a chunk')
    parser.add_argument('--room-all', dest='room_all', type=str,
                        nargs='+', default=['Room_A', 'Room_B', 'Room_C', 'Room_D'])
    parser.add_argument('--snr-all', dest='snr_all', type=int, 
                        nargs='+', default=[0, 10, 20])
    parser.add_argument('--run-id-all', dest='run_id_all', type=int, 
                        nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--azi-resolution', dest='azi_resolution',
                        type=int, default=5)
    parser.add_argument('--azi-index', dest='azi_index', type=int, 
                        nargs='+', default=0)
    parser.add_argument('--print-result', dest='print_result', type=bool, 
                        default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    main(args)
