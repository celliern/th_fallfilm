#!/usr/bin/env python
# coding=utf8

from ..models import chock
from .base_experiments import PeriodicBox
from .pprocess import compute_grad, compute_T

# Base for experiments


class ChockPeriodic(PeriodicBox):
    name = "chock_periodic"
    model = chock()

    def post_processes_factory(self):
        def post_process_Phi(simul):
            simul.fields["phi"] = simul.fields["Phi"] * simul.fields["h"]
            return simul

        return [("post_process_phi", post_process_Phi),
                ("post_process_T", compute_T),
                ("post_process_grad", compute_grad)]

    def init_fields(self, L, B, Pe, *, N=None, dx=None, h=1., q=None):
        fields = super().init_fields(L, B, Pe, N=N, dx=dx, h=h, q=q)

        return dict(**fields, Phi=fields["phi"] / fields["h"])
