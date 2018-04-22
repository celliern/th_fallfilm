#!/usr/bin/env python
# coding=utf8

import functools as ft
import logging

import numpy as np
import path

import triflow as trf
from th_fallfilm.misc import helpers

from .pprocess import compute_timer, pprocess_proxy

log = logging.getLogger(__name__)

# Base for experiments


class Experiment:
    name = "Experiment"

    def __init__(self, working_dir='.', strict=True, save="all"):
        self.simuls = []
        self._save = save
        if not self.model:
            raise NotImplementedError("model not provided by %s" %
                                      self.name)
        self.working_dir = path.Path(working_dir) / self.name
        if strict:
            self.working_dir.makedirs()
        else:
            self.working_dir.makedirs_p()

    def post_processes_factory(self):
        pass

    def hook_factory(self):
        return trf.core.schemes.null_hook

    def iter_simul(self, simul):
        simul.attach_container(self.working_dir,
                               force=True, nbuffer=10, save=self._save)
        log.info(f"\nsimulation {simul.id}"
                 f"\nrunning {self.model.name}")
        for _, (t, _) in enumerate(simul):
            log.debug(f"simulation {simul.id} "
                        f"{self.model.name} t: {t/simul.tmax*100:g} %")
        log.info(f"\nsimulation {simul.id}"
                 f"\n{self.model.name}, success !")

    def insert_pprocesses(self, simul, *post_processes):
        post_processes = [*post_processes, *self.post_processes_factory()]
        post_processes.insert(0, ("timer_pprocess", compute_timer))
        for ppname, pprocess in post_processes:
            simul.add_post_process(ppname, pprocess_proxy(pprocess))

    def run(self, sample, hold=False, **kwargs):
        sample = dict(**sample)
        sample.update(kwargs)
        simul = self.init_simul(sample)
        self.insert_pprocesses(simul)
        self.simuls.append(simul)
        if hold:
            return simul
        self.iter_simul(simul)

    def init_simul(self):
        raise NotImplementedError


# Base for periodic simulations

class PeriodicBox(Experiment):
    name = "PeriodicBox"

    def init_simul(self, sample):
        lf, tf = sample["l_factor"], sample["t_factor"]
        N, dx = (sample.get(var, None) for var in ["N", "dx"])
        fields = self.init_fields(L=sample["L"] * lf, B=sample["B"],
                                  Pe=sample["Pe"], N=N, dx=dx * lf,
                                  h=ft.partial(helpers.make_initial_wave,
                                               ampl=.1, offset=1))

        dt = sample["dt"] * tf
        tmax = sample["tmax"] * tf
        simul = trf.Simulation(self.model, t=0, fields=fields,
                               parameters=dict(**sample, periodic=True),
                               hook=self.hook_factory(), dt=dt, tmax=tmax,
                               tol=sample['tol'], id=sample["name"])

        return simul

    def init_fields(self, L, B, Pe, *, N=None, dx=None, h=1., q=None):
        if all([N, dx]) or not any([N, dx]):
            raise ValueError("You have to fill either N or dx.")
        if N:
            x, dx = np.linspace(0, L, N, retstep=True)
        else:
            x = np.arange(0, L + dx, dx)

        def ensure_field(var):
            if callable(var):
                return var(x)
            if isinstance(var, (float, int)):
                return np.zeros_like(x) + var
            return var

        h = ensure_field(h)
        if not q:
            q = h ** 3 / 3
        else:
            q = ensure_field(q)
        theta = 1 / (1 + B * h)
        phi = - B / (1 + B * h) * h

        return dict(x=x, h=h, q=q, theta=theta, phi=phi)
