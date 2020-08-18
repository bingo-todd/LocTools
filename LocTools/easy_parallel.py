from multiprocessing import Queue, Process, Manager
from BasicTools.ProcessBarMulti import ProcessBarMulti


def worker(func, tasks_queue, outputs_dict, pb):
    while True:
        task = tasks_queue.get()
        if task is None:
            return
        else:
            task_id = task[0]
            result = func(*task[1:])
            outputs_dict[task_id] = result
            
            if pb is not None:
                pb.update()


def easy_parallel(func, tasks, n_worker=4, show_process=False):
    """
    Args:
        func: function to be called in parallel
        tasks: list of list, arguments of func
        n_worker: number of processes
    """

    tasks_queue = Queue()
    [tasks_queue.put([str(task_i), *task]) for task_i, task in enumerate(tasks)]
    [tasks_queue.put(None) for worker_i in range(n_worker)]

    threads = []
    outputs_dict = Manager().dict()
    if show_process:
        pb = ProcessBarMulti([len(tasks)])
    else:
        pb = None

    for worker_i in range(n_worker):
        thread = Process(target=worker, args=(func, tasks_queue, outputs_dict, pb))
        thread.start()
        threads.append(thread)
    [thread.join() for thread in threads]
    outputs = [outputs_dict[str(task_i)] for task_i, _ in enumerate(tasks)]
    return outputs


if __name__ == '__main__':
    import time
    import numpy as np
    def test_func(i):
        time.sleep(np.random.randint(5,size=1))
        return i+1

    tasks = [[i] for i in range(10)]
    outputs = easy_parallel(test_func, tasks, show_process=True)
    print(outputs)

        
