import numpy as np
import os
import argparse
from BasicTools import parse_file


def merge_logs(log0_path, log1_path, merged_log_path):
    log0 = parse_file.file2dict(log0_path)
    log1 = parse_file.file2dict(log1_path)
    
    merged_log = {}
    for key in log0.keys():
        if key not in log1.keys():
            raise Exception(f'{key} not in {log1_path}')
        merged_log[log0[key]] = log1[key]
    parse_file.dict2file(merged_log, merged_log_path)
    

def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--log-path', dest='log_path', nargs='+',
            required=True, type=str, help='')
    parser.add_argument('--merged-log-path', dest='merged_log_path', 
            required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    merge_logs(args.log_path[0], args.log_path[1], args.merged_log_path)


if __name__ == '__main__':
    main()
