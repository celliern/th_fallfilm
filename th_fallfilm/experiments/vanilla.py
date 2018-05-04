#!/usr/bin/env python
# coding=utf8

from ..models import vanilla
from .base_experiments import PeriodicBox, OpenBox
from .pprocess import compute_grad, compute_T

# Base for experiments


class VanillaPeriodic(PeriodicBox):
    name = "vanilla_periodic"
    model = vanilla()

    def post_processes_factory(self):
        return [("post_process_T", compute_T),
                ("post_process_grad", compute_grad)]


class VanillaOpen(OpenBox):
    name = "vanilla_open"
    model = vanilla()

    def post_processes_factory(self):
        return [("post_process_T", compute_T),
                ("post_process_grad", compute_grad)]
