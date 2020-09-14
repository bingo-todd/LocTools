import logging
import argparse
import os
import string
import random


class Exp_Logger(object):
    log_dir_base = os.path.expanduser('~/Work_Space/Exp_logs')
    id_len = 10

    def __init__(self, cmd=None, pwd=None):

        if cmd is None:
            self.print_log()
            return
        
        while True: # 
            id = self.gen_id()
            self.log_dir = f'{self.log_dir_base}/{id}'
            if os.path.exists(self.log_dir):
                continue 
            
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = f'{self.log_dir}/log'
            print(f'Exp logged in {self.log_dir}')
            break

        cmd_str = ' '.join(f'{item}' for item in cmd)
        logger = logging.getLogger(f'cmd_str')
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s; %(levelname)s; %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(level=logging.DEBUG)
        #
        logger.info(cmd_str)
        logger.info(pwd)

        self._log_path = log_path
        self._logger = logger
        print('logger')
        self._file_handler = file_handler

    def gen_id(self):
        possible_characters = string.ascii_letters + string.digits
        id = ''.join([random.choice(possible_characters) for i in range(self.id_len)])
        return id

    def start(self):
        self._logger.info(f'start')

    def add_message(self, message):
        self._logger.info(message)

    def finish(self):
        self._logger.info(f'finish')
        self._logger.removeHandler(self._file_handler) 

    def load_exp_log(self):
        log_dir_all = os.listdir(self.log_dir_base)
        finished_exp_info_all = []
        running_exp_info_all = []
        for log_dir in log_dir_all:
            log_path = f'{self.log_dir_base}/{log_dir}/log'
            if not os.path.exists(log_path):
                print(log_path)
                continue
            exp_info_tmp = {'start_time':None, 'cmd': None, 'finish_time':None, 'is_finish':False}
            with open(log_path) as log_file:
                lines = log_file.readlines()
                if len(lines) < 2:
                    raise Exception(f'illegal log in {log_path}')
                
                # command
                exp_info_tmp['cmd'] = lines[0].split(';')[2].strip()
                exp_info_tmp['pwd'] = lines[1].split(';')[2].strip()
                # start time
                start_time, _, token = [item.strip() for item in lines[1].strip().split(';')]
                if 'start' != token:
                    raise Exception(f'illegal log in {log_path}')
                exp_info_tmp['start_time'] = start_time
                # finsh time
                try:
                   finish_time, _, token = [item.strip() for item in lines[-1].strip().split(';')]
                except Exception:
                    token = ''
                if 'finish' == token:
                    exp_info_tmp['is_finish'] = True
                    exp_info_tmp['finish_time'] = finish_time
                    finished_exp_info_all.append(exp_info_tmp)
                else:
                    exp_info_tmp['is_finish'] = False
                    running_exp_info_all.append(exp_info_tmp)
        return finished_exp_info_all, running_exp_info_all

    def _print_log(self, exp_i, exp_info):
        print(f"  {exp_i:0>4d}| {exp_info['cmd']} {exp_info['pwd']}")
        print(f"      | start:{exp_info['start_time']} finish:{exp_info['finish_time']}")
        print('    '+'-'*50)

    def print_log(self):
        finished_exp_info_all, running_exp_info_all = self.load_exp_log() 
        # finished experiment
        if True:
            print('finished experiment')
            print('   ' + '='*50)
            for exp_i, exp_info in enumerate(finished_exp_info_all):
                self._print_log(exp_i, exp_info)
        # 
        print('\n')

        if True:
            print('running experiment')
            print('   ' + '='*50)
            for exp_i, exp_info in enumerate(running_exp_info_all):
                self._print_log(exp_i, exp_info)
             
            
        

if __name__ == '__main__':
    # print all experiments
    Exp_Logger()
