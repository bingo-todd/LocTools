import numpy as np
import os
import argparse
import pandas
from .easy_parallel import easy_parallel


def list2str(x, separator='_'):
    if not isinstance(x, list):
        x = [x]
    return separator.join(map(str, x))


def parse_log(log_path, fig_path=None):
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
        loc_info_file_all.append([file_path, conditions, prob_frame_all])
    return loc_info_file_all


def make_decision(prob_frame, chunksize):
    n_sample = prob_frame.shape[0]
    n_chunk = n_sample - chunksize + 1
    azi = np.zeros(n_chunk)
    for chun_i in range(n_chunk):
        chunk_slice = slice(chun_i, chun_i+chunksize)
        azi[chun_i] = np.argmax(np.mean(prob_frame[chunk_slice], axis=0))
    return azi


def measure_perform(raw_log_path, result_log_path, statistic_log_path, chunksize,
                    azi_index, azi_resolution):
    if not isinstance(azi_index, list):
        azi_index = [azi_index]
    
    if not os.path.exists(raw_log_path):
        raise Exception(f'{result_log_path} do not exist')

    os.makedirs(os.path.dirname(result_log_path), exist_ok=True)
    result_logger = open(result_log_path, 'w')
    
    os.makedirs(os.path.dirname(statistic_log_path), exist_ok=True)
    statistic_logger = open(statistic_log_path, 'w')
    statistic_logger.write('# label_info; cp mse; results \n')
    
    loc_info_all = parse_log(raw_log_path)
    n_entry = len(loc_info_all)
    cp_all = np.zeros(n_entry)
    mse_all = np.zeros(n_entry)
    for entry_i, loc_info in enumerate(loc_info_all):
        file_path, conditions, prob_frame_all = loc_info

        azi_est = make_decision(prob_frame_all, chunksize)
        azi_est_str = '; '.join(list(map(str, azi_est)))
        result_logger.write(': '.join((f'{file_path}', f'{azi_est_str}\n')))

        azi_true = np.asarray([conditions[i] for i in azi_index])
        diff = np.min(np.abs(azi_est[:, np.newaxis] - azi_true[np.newaxis, :]), axis=1)

        n_sample = prob_frame_all.shape[0]
        mse_all[entry_i] = np.sqrt(np.sum(diff**2)/n_sample)*azi_resolution
        cp_all[entry_i] = np.nonzero(diff<1e-5)[0].shape[0]/n_sample
        statistic_logger.write(f'{file_path}: {mse_all[entry_i]:.4f}; {cp_all[entry_i]:.4f}\n')
        
    cp = np.mean(cp_all) 
    mse = np.mean(mse_all)
    statistic_logger.write(f'# mean \n # \t mse:{mse} \n # \t cp:{cp}')

    return mse, cp


def save_result(results, rooms, snrs, csv_path):
    snr_str_all = [f'{snr}dB' for snr in snrs]
    data_frame = pandas.DataFrame(
            columns=['Measure', 'Room', *snr_str_all])

    row_mse_all = []
    row_cp_all = []
    for room_i, room in enumerate(rooms):
        tmp = {'Room': room}
        row_mse_tmp = {f'{snr}dB': results[room_i, snr_i, 0]
                       for snr_i, snr in enumerate(snrs)}
        row_mse_all.append({**tmp, **row_mse_tmp})

        row_cp_tmp = {f'{snr}dB': results[room_i, snr_i, 1]
                      for snr_i, snr in enumerate(snrs)}
        row_cp_all.append({**tmp, **row_cp_tmp})

    data_frame = data_frame.append({'Measure': 'mse'}, ignore_index=True)
    data_frame = data_frame.append(row_mse_all, ignore_index=True)

    data_frame = data_frame.append({'Measure': 'cp'}, ignore_index=True)
    data_frame = data_frame.append(row_cp_all, ignore_index=True)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    data_frame.to_csv(csv_path, index=False, index_label=None)



def load_log(model_dir, chunksize, room=None, snr=None, run_id=1, 
        azi_resolution=5, azi_index=0, print_result=False):
    """"""
   
    if not isinstance(room, list):
        rooms = [room]
    else:
        rooms = room

    if not isinstance(snr, list):
        snrs = [snr]
    else:
        snrs = snr
    
    if not isinstance(run_id, list):
        run_ids = [run_id] 
    else:
        run_ids = run_id

    if not isinstance(azi_index, list):
        azi_indexs = [azi_index] 
    else:
        azi_indexs = azi_index

    statistic_dir = ''.join((
        f'{model_dir}/statistic/',
        f'chunksize{chunksize}_srcaziindex{list2str(azi_index)}'))
    n_room = len(rooms)
    n_snr = len(snrs)
    n_run = len(run_ids)
    results = np.zeros((n_room, n_snr, n_run, 2))
    tasks = []
    for room_i, room in enumerate(rooms):
        for snr_i, snr in enumerate(snrs):
            for run_i, run_id in enumerate(run_ids):
                file_name = ''
                if room is not None:
                    file_name = f'{file_name}{room}_'
                if snr is not None:
                    file_name = f'{file_name}{snr}_'
                file_name = f'{file_name}{run_id}'

                raw_log_path = f'{model_dir}/loc_log/{file_name}.txt'
                result_log_path = f'{statistic_dir}/estimate_log/{file_name}.txt'
                statistic_log_path = f'{statistic_dir}/{file_name}.txt'
                tasks.append([raw_log_path, result_log_path, statistic_log_path, 
                    chunksize, azi_index, azi_resolution])
    parallel_ouputs = easy_parallel(measure_perform, tasks, len(tasks)) 
    task_count = 0
    for room_i, room in enumerate(rooms):
        for snr_i, snr in enumerate(snrs):
           for run_i, run_id in enumerate(run_ids):
               results[room_i, snr_i, run_i] = parallel_ouputs[task_count]
               task_count = task_count + 1
    
    # average across all run
    print(f'average across run {run_ids}')
    results = np.mean(results, axis=2)
    if print_result:
        print('mse')
        print(results[:, :, 0])
        print('cp')
        print(results[:, :, 1])
    # save to csv
    csv_path = ''.join((
        f'{statistic_dir}/',
        '_'.join((
            f'runid{list2str(run_ids)}',
            f'chunksize{chunksize}',
            f'aziindex{list2str(azi_index)}.csv'))))
    save_result(results, rooms, snrs, csv_path)
    

def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model-dir', dest='model_dir',
                        required=True, type=str, help='directory of model')
    parser.add_argument('--chunksize', dest='chunksize', required=True,
                        type=int, help='sample number of a chunk')
    parser.add_argument('--room', dest='room', type=str,
                        nargs='+', default=None)
    parser.add_argument('--snr', dest='snr', type=int, 
                        nargs='+', default=None)
    parser.add_argument('--run-id', dest='run_id', type=int, 
                        nargs='+', default=1)
    parser.add_argument('--azi-resolution', dest='azi_resolution',
                        type=int, default=5)
    parser.add_argument('--azi-index', dest='azi_index', type=int, 
                        nargs='+', default=0)
    parser.add_argument('--print-result', dest='print_result', type=str, 
                        default='true', choices=['true', 'false'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    print_result = args.print_result == 'true'
    load_log(args.model_dir, args.chunksize, args.room, args.snr, args.run_id, 
            args.azi_resolution, args.azi_index, print_result)

if __name__ == '__main__':
    main()
