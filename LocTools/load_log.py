import numpy as np
import os
import argparse
import pandas
from .easy_parallel import easy_parallel


def list2str(x, separator='_'):
    if not isinstance(x, list):
        x = [x]
    return separator.join(map(str, x))


def load_loc_log(log_path, fig_path=None):
    with open(log_path) as log_file:
        lines = log_file.readlines()
        # remove line start with #
        lines = filter(lambda line: line.strip()[0] != '#', lines)

    loc_info_file_all = []
    for line_i, line in enumerate(lines):
        file_path, *prob_frame_str_all = line.strip().split(';')  #
        file_name = os.path.basename(file_path).strip().split('.')[0]
        conditions = list(map(int, file_name.split('_')))
        prob_frame_all = np.asarray(
            [list(map(float, item.strip().split())) for item in prob_frame_str_all])
        loc_info_file_all.append([conditions, prob_frame_all])
    return loc_info_file_all


def decision_maker(prob_frame, chunksize):
    n_sample = prob_frame.shape[0]
    n_chunk = n_sample - chunksize + 1
    azi = np.zeros(n_chunk)
    for chun_i in range(n_chunk):
        chunk_slice = slice(chun_i, chun_i+chunksize)
        azi[chun_i] = np.argmax(np.mean(prob_frame[chunk_slice], axis=0))
    return azi


def perform_measure(result_log_path, statistic_log_path, chunksize,
                    azi_index=0):
    if not isinstance(azi_index, list):
        azi_index = [azi_index]
    
    if not os.path.exists(result_log_path):
        raise Exception(f'{result_log_path} do not exist')

    os.makedirs(os.path.dirname(statistic_log_path), exist_ok=True)
    logger = open(statistic_log_path, 'w')
    logger.write('# label_info; cp mse; results \n')
    
    loc_info_all = load_loc_log(result_log_path)
    n_entry = len(loc_info_all)
    cp_all = np.zeros(n_entry)
    mse_all = np.zeros(n_entry)
    for entry_i, loc_info in enumerate(loc_info_all):
        conditions, prob_frame_all = loc_info

        est_azi = decision_maker(prob_frame_all, chunksize)
        src_azi = np.asarray([conditions[i] for i in azi_index])
        diff = np.min(np.abs(est_azi[:, np.newaxis] - src_azi[np.newaxis, :]), axis=1)

        n_sample = prob_frame_all.shape[0]
        mse_all[entry_i] = np.sqrt(np.sum(diff**2)/n_sample)*args.azi_resolution
        cp_all[entry_i] = np.nonzero(diff<1e-5)[0].shape[0]/n_sample
        
        conditions_str = ' '.join([f'{item}' for item in conditions])
        est_azi_str = ' '.join(list(map(str, est_azi)))
        logger.write('; '.join((f'{conditions_str}',
                                f'{est_azi_str}\n')))

    cp = np.mean(cp_all) 
    mse = np.mean(mse_all)
    logger.write(f'# mean \n # \t mse:{mse} \n # \t cp:{cp}')

    return mse, cp


def main(args):
    model_dir = args.model_dir
    chunksize = args.chunksize
    
    if args.no_snr:
        snr_all = [None]
    else:
        snr_all = args.snr_all
    n_snr = len(snr_all)
    
    if not isinstance(args.run_id_all, list):
        run_id_all = [args.run_id_all] 
    else:
        run_id_all = args.run_id_all

    if not isinstance(args.azi_index, list):
        azi_index = [args.azi_index] 
    else:
        azi_index = args.azi_index

    if not isinstance(args.room_all, list):
        room_all = [args.room_all]
    else:
        room_all = args.room_all

    statistic_dir = ''.join((
        f'{model_dir}/statistic_log/',
        f'chunksize{chunksize}_srcaziindex{list2str(azi_index)}'))
    n_room = len(room_all)
    n_run = len(run_id_all)
    result_all = np.zeros((n_room, n_snr, n_run, 2))
    tasks = []
    for room_i, room in enumerate(room_all):
        for snr_i, snr in enumerate(snr_all):
            for run_i, run_id in enumerate(run_id_all):
                if snr is None:
                    loc_log_path = f'{model_dir}/loc_log/{room}_{run_id}.txt'
                    statistic_log_path = f'{statistic_dir}/{room}_{run_id}.txt'
                else:
                    loc_log_path = f'{model_dir}/loc_log/{room}_{snr}_{run_id}.txt'
                    statistic_log_path = f'{statistic_dir}/{room}_{snr}_{run_id}.txt'

                # result_all[room_i, snr_i, run_i] = perform_measure(
                #         loc_log_path, statistic_log_path, chunksize, azi_index)
       
                tasks.append([loc_log_path, statistic_log_path, chunksize, azi_index])
    parallel_ouputs = easy_parallel(perform_measure, tasks, len(tasks)) 

    task_count = 0
    for room_i, room in enumerate(room_all):
        for snr_i, snr in enumerate(snr_all):
           for run_i, run_id in enumerate(run_id_all):
               result_all[room_i, snr_i, run_i] = parallel_ouputs[task_count]
               task_count = task_count + 1
    
    # average across all run
    print(f'average across run {run_id_all}')
    result_all = np.mean(result_all, axis=2)
    if args.print_result:
        print('mse')
        print(result_all[:, :, 0])
        print('cp')
        print(result_all[:, :, 1])
    # save to csv
    snr_str_all = [f'{snr}dB' for snr in snr_all]
    data_frame = pandas.DataFrame(
        columns=['Measure', 'Room', *snr_str_all])
    row_mse_all = []
    row_cp_all = []
    for room_i, room in enumerate(room_all):
        tmp = {'Room': room}
        row_mse_tmp = {f'{snr}dB': result_all[room_i, snr_i, 0]
                       for snr_i, snr in enumerate(snr_all)}
        row_mse_all.append({**tmp, **row_mse_tmp})

        row_cp_tmp = {f'{snr}dB': result_all[room_i, snr_i, 1]
                      for snr_i, snr in enumerate(snr_all)}
        row_cp_all.append({**tmp, **row_cp_tmp})

    data_frame = data_frame.append({'Measure': 'mse'}, ignore_index=True)
    data_frame = data_frame.append(row_mse_all, ignore_index=True)

    data_frame = data_frame.append({'Measure': 'cp'}, ignore_index=True)
    data_frame = data_frame.append(row_cp_all, ignore_index=True)

    csv_path = ''.join((
        f'{model_dir}/result_csv/',
        '_'.join((
            f'runid{list2str(run_id_all)}',
            f'chunksize{chunksize}',
            f'aziindex{list2str(azi_index)}.csv'))))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    data_frame.to_csv(csv_path, index=False, index_label=None)


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model-dir', dest='model_dir',
                        required=True, type=str, help='directory of model')
    parser.add_argument('--chunksize', dest='chunksize', required=True,
                        type=int, help='sample number of a chunk')
    parser.add_argument('--room-all', dest='room_all', type=str,
                        nargs='+', default=['Room_A', 'Room_B', 'Room_C', 'Room_D'])
    parser.add_argument('--snr-all', dest='snr_all', type=int, 
                        nargs='+', default=[0, 10, 20])
    parser.add_argument('--no-snr', dest='no_snr', type=bool, 
                        default=False)
    parser.add_argument('--run-id', dest='run_id_all', type=int, 
                        nargs='+', default=1)
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
