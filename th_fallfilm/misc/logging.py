#!/usr/bin/env python
# coding=utf8

import logging
import logging.handlers
import functools as ft
import multiprocessing as mp
from contextlib import contextmanager


def listener_configurer(logfile):
    root = logging.getLogger()
    handler = logging.handlers.RotatingFileHandler(logfile, 'a', 1000000, 1)
    formatter = logging.Formatter(
        '%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


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


def worker_configurer(queue, loglevel):
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(loglevel)


@contextmanager
def log_pool(logfile="multiprocess.log", loglevel="INFO", *args, **kwargs):
    queue = mp.Queue(-1)
    conf = ft.partial(listener_configurer, logfile)
    listener = mp.Process(target=listener_process,
                          args=(queue, conf))
    listener.start()
    kwargs["initializer"] = ft.partial(worker_configurer, queue, loglevel)
    with mp.Pool(*args, **kwargs) as p:
        yield p
    queue.put_nowait(None)
    listener.join()
