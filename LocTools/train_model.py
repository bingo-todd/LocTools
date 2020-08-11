from .send_email import send_email
from .parallel import parallel
import importlib


def train_model(args):
    trainer = importlib.import_module(args.train_script)

    if args.is_parallel:
        parallel(trainer.main, args.tasks, args.n_worker)
    else:
        for task in args.tasks:
            trainer.main(task)

    send_email.main(message=f'{args.task_descripton}')
