#!/usr/bin/env python
# coding=utf8

import functools as ft
import logging
import multiprocessing as mp
from contextlib import contextmanager

from path import Path


def listener_configurer(logfile, logs=None):
    handler = logging.FileHandler(Path(logfile), 'a')
    formatter = logging.Formatter(
        '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    for log in ["root"] if not logs else logs:
        logging.getLogger(log).addHandler(handler)


def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            import sys
            import traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker_configurer(queue, loglevel, logs=None):
    handler = logging.handlers.QueueHandler(queue)
    for log in ["root"] if not logs else logs:
        logging.getLogger(log).addHandler(handler)
        logging.getLogger(log).setLevel(loglevel)


@contextmanager
def log_pool(logfile="multiprocess.log", loglevel="INFO", logs=None,
             *args, **kwargs):
    queue = mp.Queue(-1)
    conf = ft.partial(listener_configurer, logfile, logs)
    listener = mp.Process(target=listener_process,
                          args=(queue, conf))
    listener.start()
    kwargs["initializer"] = ft.partial(worker_configurer,
                                       queue, loglevel, logs)
    with mp.Pool(*args, **kwargs) as p:
        yield p
    queue.put_nowait(None)
    listener.join()
