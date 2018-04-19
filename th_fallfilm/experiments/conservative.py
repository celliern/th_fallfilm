#!/usr/bin/env python
# coding=utf8

from ..models import conservative
from .base_experiments import PeriodicBox
from .pprocess import compute_grad, compute_T

# Base for experiments


class ConservativePeriodic(PeriodicBox):
    name = "conservative_periodic"
    model = conservative()

    def post_processes_factory(self):
        return [("post_process_T", compute_T),
                ("post_process_grad", compute_grad)]
