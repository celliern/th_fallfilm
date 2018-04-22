#!/usr/bin/env python
# coding=utf8

import numpy as np
import inspect


def compute_timer(fields, timer):
    fields["timer_last"] = timer.last
    fields["timer_total"] = timer.total
    return fields


def compute_T(fields, parameters):
    y = np.linspace(0, 1, 10)
    fields = fields.assign_coords(y=y)

    h = np.repeat(fields.h.expand_dims("y", -1),
                  y.size, axis=1)
    theta = np.repeat(fields.theta.expand_dims("y", -1),
                      y.size, axis=1)
    phi = np.repeat(fields.phi.expand_dims("y", -1),
                    y.size, axis=1)

    B = parameters["B"]
    T = ((2 + phi*(-2 + y)*(-1 + y)*y + 2*(-1 + theta)*y**2 +
          B*h*(2 + phi*(-2 + y)*(-1 + y)*y + y**2*(-3 + 2*theta + y)))
         / (2 + 2*B*h))
    fields["T"] = ("x", "y"), T
    return fields


def compute_grad(fields):

    dxT, dyT = np.gradient(fields["T"],
                           (fields.x[1] - fields.x[0]).values,
                           1, axis=(0, 1))
    dyT /= ((fields.y[1] - fields.y[0]) * fields.h).values[:, None]

    dxh = np.gradient(fields["h"], (fields.x[1] - fields.x[0]).values, axis=0)

    fields["dxT"] = ("x", "y"), dxT
    fields["dyT"] = ("x", "y"), dyT
    fields["mag"] = ("x", "y"), dxT ** 2 + dyT ** 2

    flux_h = -(dxh * fields.dxT.isel(y=-1) + fields.dyT.isel(y=-1))
    flux_s = -(fields.dxT.isel(y=0) + fields.dyT.isel(y=0))

    fields["flux_h"] = ("x",), flux_h
    fields["flux_s"] = ("x",), flux_s

    return fields


def compute_grad_asympt(fields):

    dxT, dyT = np.gradient(fields["T"],
                           (fields.x[1] - fields.x[0]).values,
                           1, axis=(0, 1))
    dyT /= ((fields.y[1] - fields.y[0]) * fields.h).values[:, None]

    dxh = np.gradient(fields["h"], (fields.x[1] - fields.x[0]).values, axis=0)

    fields["dxT"] = ("x", "y"), dxT
    fields["dyT"] = ("x", "y"), dyT
    fields["mag"] = ("x", "y"), dxT ** 2 + dyT ** 2

    nx_h = -dxh / (np.sqrt(1 + dxh**2))
    ny_h = -1 / (np.sqrt(1 + dxh**2))

    flux_h = nx_h * fields.dxT.isel(y=-1) + ny_h * fields.dyT.isel(y=-1)
    flux_s = -(fields.dxT.isel(y=0) + fields.dyT.isel(y=0))

    fields["Phi_h"] = ("x",), flux_h
    fields["Phi_s"] = ("x",), flux_s

    return fields


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


def pprocess_proxy(pprocess):
    requested_attributes = inspect.getargspec(pprocess)

    def proxied_pprocess(simul):
        kwargs = {key: getattr(simul, key)
                  for key in requested_attributes[0]}
        simul.fields = pprocess(**kwargs)
    return proxied_pprocess
