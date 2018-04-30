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
        def post_process_theta(fields):
            fields["theta"] = "x", fields["T"].isel(y=-1)
            return fields

        def post_process_phi(fields):
            dy = fields["y"].isel(y=1) - fields["y"].isel(y=0)
            fields["phi"] = ("x", fields["T"].diff("y").isel(y=0) / dy)
            return fields
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
    Ny = 2


class RefFourierPeriodic(FourierPeriodic):
    name = "reference_fourier_periodic"

    def Ny(self, Pe):
        return min(max(10, np.sqrt(Pe)), 40)
