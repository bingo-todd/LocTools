import importlib
import argparse
from multiprocessing import Queue, Process


def worker(func, tasks_queue):
    while True:
        task = tasks_queue.get()
        if task is None:
            return
        else:
            func(task)


def parallel(func, tasks, n_worker):
    tasks_queue = Queue()
    [tasks_queue.put(task) for task in tasks]
    [tasks_queue.put(None) for worker_i in range(n_worker)]

    threads = []
    for worker_i in range(n_worker):
        thread = Process(target=worker, args=(func, tasks_queue))
        thread.start()
        threads.append(thread)
    [thread.join() for thread in threads]


def parse_args():
    parser = argparse.ArgumentParser(description='parallel inerface')
    parser.add_argument('--task', type=str, dest='task', nargs='+',
                        required=True, action='append')
    parser.add_argument('--script-path', type=str, dest='script_path',
                        required=True,
                        help="python script containing func, replace '/' with '.'")
    parser.add_argument('--n-worker', type=int, dest='n_worker',
                        default=4, help='number of process')
    args = parser.parse_args()
    return args


def main(func, tasks, n_worker=4):
    parallel(func, tasks, n_worker)


if __name__ == '__main__':
    args = parse_args()

    script_path = args.script_path.strip('.py').replace('.', '/')
    script_obj = importlib.import_module(script_path)
    parallel(script_obj.main, args.task, args.n_worker)
