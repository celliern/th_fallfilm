#!/usr/bin/env python
# coding=utf8

from importlib import import_module


def get_experiments(model, box):
    exp_class = ''.join(map(str.capitalize, [*model.split("_"), box]))
    model = model.replace("reference_", "")
    model = model.replace("coarse_", "")
    module = import_module(".%s" % model,
                           package='th_fallfilm.experiments')
    return getattr(module, exp_class)
