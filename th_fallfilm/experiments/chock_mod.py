#!/usr/bin/env python
# coding=utf8

from ..models import chock_mod
from .base_experiments import PeriodicBox, OpenBox
from .pprocess import compute_grad, compute_T

# Base for experiments


class ChockModPeriodic(PeriodicBox):
    name = "chock_mod_periodic"
    model = chock_mod()

    def post_processes_factory(self):
        return [("post_process_T", compute_T),
                ("post_process_grad", compute_grad)]

    def init_fields(self, L, B, Pe, *, N=None, dx=None, h=1., q=None):
        fields = super().init_fields(L, B, Pe, N=N, dx=dx, h=h, q=q)

        return dict(**fields, Phi=fields["phi"] / fields["h"])


class ChockModOpen(OpenBox):
    name = "chock_mod_open"
    model = chock_mod()

    def post_processes_factory(self):
        return [("post_process_T", compute_T),
                ("post_process_grad", compute_grad)]

    def init_fields(self, L, B, Pe, *, N=None, dx=None, h=1., q=None):
        fields = super().init_fields(L, B, Pe, N=N, dx=dx, h=h, q=q)

        return dict(**fields, Phi=fields["phi"] / fields["h"])
