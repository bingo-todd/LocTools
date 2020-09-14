import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from BasicTools import get_file_path
from .add_loc_log import add_loc_log
from .easy_parallel import easy_parallel


def _cal_hist(log_path, chunksize, result_path):
    result_logger = open(result_path, 'w')
    with open(log_path) as log_file:
        line_all = log_file.readlines()
        for line_i, line in enumerate(line_all):
            line = line.strip()
            if line.startswith('#'):
                continue

            file_path, results_str = line.strip().split(';')
            results = np.asarray(
                    list(
                        map(lambda x: int(float(x)), results_str.split())))
            max_result_value = np.max(results)
            values, counts = np.unique(results, return_counts=True)
            n_sample = results.shape[0]
            if chunksize < 0:
                chunksize = n_sample
            else:
                chunksize = chunksize
            n_chunk = np.int(np.floor(n_sample/chunksize))
            hist = np.zeros((n_chunk, max_result_value+1), dtype=np.float32) 
            for chunk_i in range(n_chunk):
                chunk_slice = slice(chunk_i*chunksize, (chunk_i+1)*chunksize)
                values, counts = np.unique(results[chunk_slice], return_counts=True)
                hist[chunk_i, values] = counts/chunksize
            hist_str = '; '.join(
                    [' '.join(
                        map(lambda x:f'{x:>.4f}', hist[chunk_i])) 
                        for chunk_i in range(n_chunk)])
            result_logger.write(f'{file_path}; {hist_str}\n')
            result_logger.flush()
    result_logger.close()


def cal_hist(log_path, chunksize=None, result_path=None):
   
    log_path = os.path.realpath(log_path)
    if os.path.isdir(log_path):
        log_dir = log_path
        log_path_all = get_file_path(log_dir, suffix='.txt', is_absolute=True)
    else:
        log_dir = os.path.dirname(log_path)
        log_path_all = [log_path]

    if result_path is None:
        result_dir = f'{os.path.dirname(log_dir)}/hist/chunksize{chunksize}'
    elif os.path.isdir(result_path):
        result_dir = f'{result_path}/hist/chunksize{chunksize}'
        result_path = None
    os.makedirs(result_dir, exist_ok=True)

    task_all = []
    for log_path_tmp in log_path_all:
        log_name = os.path.basename(log_path_tmp).split('.')[0]
        if result_path is None:
            result_path_tmp = f'{result_dir}/{log_name}.txt' 
        task_all.append([log_path_tmp, chunksize, result_path_tmp])
    easy_parallel(_cal_hist, task_all, len(task_all))


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log-path', dest='log_path', required=True, 
            type=str, help='path of the input file')
    parser.add_argument('--chunksize', dest='chunksize', 
            type=int, default=-1, help='chunksize')
    parser.add_argument('--result-path', dest='result_path',
            type=str, default=None, help='path of the output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cal_hist(args.log_path, args.chunksize, args.result_path)


if __name__ == '__main__':
    main()


