#!/usr/bin/env python
# coding=utf8

import numpy as np


def compute_timer(simul):
    simul.fields["timer_last"] = simul.timer.last
    simul.fields["timer_total"] = simul.timer.total
    return simul


def compute_T(simul):
    y = np.linspace(0, 1, 10)
    simul.fields = simul.fields.assign_coords(y=y)

    h = np.repeat(simul.fields.h.expand_dims("y", -1),
                  y.size, axis=1)
    theta = np.repeat(simul.fields.theta.expand_dims("y", -1),
                      y.size, axis=1)
    phi = np.repeat(simul.fields.phi.expand_dims("y", -1),
                    y.size, axis=1)

    B = simul.parameters["B"]
    T = ((2 + phi*(-2 + y)*(-1 + y)*y + 2*(-1 + theta)*y**2 +
         B*h*(2 + phi*(-2 + y)*(-1 + y)*y + y**2*(-3 + 2*theta + y)))
         / (2 + 2*B*h))
    simul.fields["T"] = ("x", "y"), T
    return simul


def compute_grad(simul):
    data = simul.fields

    dxT, dyT = np.gradient(data["T"],
                           (data.x[1] - data.x[0]).values,
                           1, axis=(0, 1))
    dyT /= ((data.y[1] - data.y[0]) * data.h).values[:, None]

    dxh = np.gradient(data["h"], (data.x[1] - data.x[0]).values, axis=0)

    simul.fields["dxT"] = ("x", "y"), dxT
    simul.fields["dyT"] = ("x", "y"), dyT
    simul.fields["mag"] = ("x", "y"), dxT ** 2 + dyT ** 2

    flux_h = -(dxh * data.dxT.isel(y=-1) + data.dyT.isel(y=-1))
    flux_s = -(data.dxT.isel(y=0) + data.dyT.isel(y=0))

    simul.fields["flux_h"] = ("x",), flux_h
    simul.fields["flux_s"] = ("x",), flux_s

    return simul


class Error:
    def __init__(self, value, reference, norm):
        self.value = value
        self.ref = reference
        self.norm = norm

    @property
    def absolute(self):
        return np.linalg.norm(self.value - self.ref, self.norm)

    @property
    def relative(self):
        return (2 * self.absolute /
                (np.linalg.norm(self.ref, self.norm) +
                 np.linalg.norm(self.value, self.norm)))


def compute_err(name, norm, models):
    name = sample["name"]
    print(("sample: %s, norm: %s" % (name, norm)).ljust(80), end="\r")
    data_dns = trf.retrieve_container(
        "./data/Fourier/%s" % name, isel=-1).data.load()
    data_ch = trf.retrieve_container(
        "./data/chock/%s" % name, isel=-1).data.load()
    data_van = trf.retrieve_container(
        "./data/vanilla/%s" % name, isel=-1).data.load()
    data_nch = trf.retrieve_container(
        "./data/no_chock/%s" % name, isel=-1).data.load()

    phi_van = data_van["phi"]
    phi_nch = data_nch["phi"]
    phi_ch = data_ch["phi"]
    phi_dns = (data_dns["T"].diff("y") / data_dns.y.diff("y").mean()).isel(y=0)

    theta_van = data_van["theta"]
    theta_nch = data_nch["theta"]
    theta_ch = data_ch["theta"]
    theta_dns = data_dns["T"].isel(y=-1)

    err_theta_van = Error(theta_van, theta_dns, norm)
    err_theta_nch = Error(theta_nch, theta_dns, norm)
    err_theta_ch = Error(theta_ch, theta_dns, norm)

    err_phi_van = Error(phi_van, phi_dns, norm)
    err_phi_nch = Error(phi_nch, phi_dns, norm)
    err_phi_ch = Error(phi_ch, phi_dns, norm)
