#!/usr/bin/env python
# coding=utf8

from importlib import import_module


def get_experiments(model, box):
    exp_class = ''.join(map(str.capitalize, [model, box]))
    module = import_module(".%s" % model, package='th_fallfilm.experiments')
    return getattr(module, exp_class)
