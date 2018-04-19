#!/usr/bin/env python
# coding=utf8

import numpy as np

from ..models import fourier
from .base_experiments import PeriodicBox
from .pprocess import compute_grad

# Base for experiments


# All periodic Fourier cases

class FourierPeriodic(PeriodicBox):
    model = fourier(periodic=True)

    def post_processes_factory(self):
        def post_process_theta(simul):
            simul.fields["theta"] = "x", simul.fields["T"].isel(y=-1)
            return simul

        def post_process_phi(simul):
            dy = simul.fields["y"].isel(y=1) - simul.fields["y"].isel(y=0)
            simul.fields["phi"] = ("x",
                                   simul.fields["T"].diff("y").isel(y=0) / dy)
            return simul
        return [("post_process_theta", post_process_theta),
                ("post_process_phi", post_process_phi),
                ("post_process_grad", compute_grad)]

    def init_fields(self, L, B, Pe, *, N=None, dx=None, h=1., q=None):
        fields = super().init_fields(L, B, Pe, N=N, dx=dx, h=h, q=q)
        if callable(self.Ny):
            Ny = self.Ny(Pe)
        else:
            Ny = self.Ny
        y = np.linspace(0, 1, Ny)
        T = y[None, :] * (fields["theta"][:, None] - 1) + 1
        return dict(**fields, T=T, y=y)


class CoarseFourierPeriodic(FourierPeriodic):
    name = "coarse_fourier_periodic"
    Ny = 4


class RefFourierPeriodic(FourierPeriodic):
    name = "reference_fourier_periodic"

    def Ny(self, Pe):
        return min(max(10, np.sqrt(Pe)), 40)
