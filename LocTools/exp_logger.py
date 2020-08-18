import logging
import argparse
import os


class Exp_Logger(object):
    
    log_dir = os.path.expanduser('~/Exp_logs')

    def __init__(self, exp_name, exp_info=None):
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = f'{self.log_dir}/{exp_name}'
        if os.path.exists(log_path):
            raise Exception(f'{exp_name} already exists in {self.log_dir}')

        logger = logging.getLogger(f'{exp_name}')
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(level=logging.DEBUG)

        self._log_path = log_path
        self._logger = logger
        self._file_handler = file_handler

    def start(self, message=None):
        self._logger.info(f'start: {message}')

    def add_message(self, message):
        self._logger.info(message)

    def finish(self, message=None):
        self._logger.info(f'finish: {message}')
        

if __name__ == '__main__':
    import time
    exp_name = 'test'
    exp_logger = Exp_Logger(exp_name)
    exp_logger.start()
    exp_logger.add_message('test test')
    time.sleep(10)
    exp_logger.finish()
