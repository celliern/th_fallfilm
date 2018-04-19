#!/usr/bin/env python
# coding=utf8

from sympy import S, Symbol, symbols, solve, lambdify, Max, Min, Function

import numpy as np
import scipy.sparse as sps
import theano as th
from theano import tensor as T
import triflow as trf


def null_hook(t, fields, pars):
    return fields, pars


dth = S("-dxq")
dtq = S("(546*dxxq*h**2 - 420*q + 1302*dxh**2*q -"
        " 9*h*q*(77*dxxh + 136*dxq*Re) -"
        " 2*dxh*(693*dxq*h + 70*Ct*h**3 -"
        " 324*q**2*Re) + 140*h**3*(1 + dxxxh*We))/(504.*h**2*Re)")

dtT = S("dxxT/(3.*Pe)"
        " + dyyT/(3.*h**2*Pe)"
        " - upwind(u, T)"
        " + (dyT*u*y)/h * dxh"
        " - (dyT*v)/h"
        " - (dxq*dyT*y)/h"
        " + (2*dxh**2*dyT*y)/(3.*h**2*Pe)"
        " - (2*dxh*dxyT*y)/(3.*h*Pe)"
        " - (dxxh*dyT*y)/(3.*h*Pe)"
        " + (dxh**2*dyyT*y**2)/(3.*h**2*Pe)")

u = S("(-3*q*(-2 + y)*y)/(2.*h)")
v = S("((dxq*h*(-3 + y) - 3*dxh*q*(-2 + y))*y**2)/(2.*h)")

cdiff_sub = {
    "h": S("h_i"),
    "q": S("q_i"),
    "T": S("T_i_j"),
    "y": S("y_j"),
    "dxh": S("(.5 * h_ip1 - .5 * h_im1) / dx"),
    "dxq": S("(.5 * q_ip1 - .5 * q_im1) / dx"),
    "dxxh": S("(h_ip1 - 2 * h_i + h_im1) / dx**2"),
    "dxxq": S("(q_ip1 - 2 * q_i + q_im1) / dx**2"),
    "dxxxh": S("(-.5 * h_im2 + h_im1 - h_ip1 + .5 * h_ip2) / dx**3"),
    "dxT": S("(.5 * T_ip1_j - .5 * T_im1_j) / dx"),
    "dxxT": S("(T_ip1_j - 2 * T_i_j + T_im1_j) / dx**2"),
    "dyT": S("T_i_j*((-y_j + y_jp1)/(y_j - y_jm1) - 1)/(-y_j + y_jp1)"
             "- T_i_jm1*(-y_j + y_jp1)/((y_j - y_jm1)*(-y_jm1 + y_jp1))"
             "+ T_i_jp1*(y_j - y_jm1)/((-y_j + y_jp1)*(-y_jm1 + y_jp1))"),
    "dyyT": S("-2*T_i_j/((-y_j + y_jp1)*(y_j - y_jm1))"
              "+ 2*T_i_jm1/((y_j - y_jm1)*(-y_jm1 + y_jp1))"
              "+ 2*T_i_jp1/((-y_j + y_jp1)*(-y_jm1 + y_jp1))"),
    "dxyT": S("(.5 * T_ip1_j - .5 * T_im1_j) / dx"
              "*((-y_j + y_jp1)/(y_j - y_jm1) - 1)/(-y_j + y_jp1)"
              "- (.5 * T_ip1_jm1 - .5 * T_im1_jm1) / dx*(-y_j + y_jp1)"
              "/((y_j - y_jm1)"
              "*(-y_jm1 + y_jp1)) + (.5 * T_ip1_jp1 - .5 * T_im1_jp1) / dx"
              "*(y_j - y_jm1)/((-y_j + y_jp1)*(-y_jm1 + y_jp1))"),
    "s": 0,
    "dxs": 0,
    "dxxs": 0,
    "dxxxs": 0
}


def upwind(a, U):
    dx = Symbol('dx')
    var_label = str(U)
    ap = Max(a, 0)
    am = Min(a, 0)
    Um1 = Symbol('%s_im1_j' % var_label)
    Up1 = Symbol('%s_ip1_j' % var_label)
    Um2 = Symbol('%s_im2_j' % var_label)
    Up2 = Symbol('%s_ip2_j' % var_label)
    ap = Max(a, 0)
    am = Min(a, 0)
    Um = (2 * Up1 + 3 * U - 6 * Um1 + Um2) / (6 * dx)
    Up = (-2 * Um1 - 3 * U + 6 * Up1 - Up2) / (6 * dx)
    return ap * Um + am * Up


dth = dth.subs(cdiff_sub).simplify()
dtq = dtq.subs(cdiff_sub).simplify()
dtT = dtT.replace(Function("upwind"), upwind)
dtT = dtT.subs("u", u)
dtT = dtT.subs("v", v)
dtT = dtT.subs(cdiff_sub).simplify()

upper_bdc = S(
    "dyT - (2*dxh*dxT*h - B*(2 + dxh**2)*h*T)/(2.*(1 + dxh**2))")
upper_bdc = upper_bdc.subs(cdiff_sub)
T_i_jp1 = solve(upper_bdc, "T_i_jp1")[0]

# shift left
T_im1_jp1 = T_i_jp1.copy()
T_im1_jp1 = T_im1_jp1.subs("T_im1_j", "T_im2_j")
T_im1_jp1 = T_im1_jp1.subs("T_i_j", "T_im1_j")
T_im1_jp1 = T_im1_jp1.subs("T_p1_j", "T_i_j")

T_im1_jp1 = T_im1_jp1.subs("h_im1", "h_im2")
T_im1_jp1 = T_im1_jp1.subs("h_i", "h_im1")
T_im1_jp1 = T_im1_jp1.subs("h_p1", "h_i")

# shift right
T_ip1_jp1 = T_i_jp1.copy()
T_ip1_jp1 = T_ip1_jp1.subs("T_ip1_j", "T_ip2_j")
T_ip1_jp1 = T_ip1_jp1.subs("T_i_j", "T_ip1_j")
T_ip1_jp1 = T_ip1_jp1.subs("T_m1_j", "T_i_j")
T_ip1_jp1 = T_ip1_jp1.subs("T_m1_j", "T_i_j")

T_ip1_jp1 = T_ip1_jp1.subs("h_ip1", "h_ip2")
T_ip1_jp1 = T_ip1_jp1.subs("h_i", "h_ip1")
T_ip1_jp1 = T_ip1_jp1.subs("h_m1", "h_i")

upper_dtT = dtT.copy()
upper_dtT = upper_dtT.subs("T_im1_jp1", T_im1_jp1)
upper_dtT = upper_dtT.subs("T_ip1_jp1", T_ip1_jp1)
upper_dtT = upper_dtT.subs("T_i_jp1",
                           T_i_jp1).subs("y_j", 1)

# # we fix dy constant between y_j and y_jp1
upper_dtT = upper_dtT.subs("y_jp1", 2 - Symbol("y_jm1"))

symbolic_parameters = symbols("Re We Ct Pe B dx")
dependant_variable = symbols("x y_jm1 y_j y_jp1")

si, sj = symbols("i j")
sNx, sNy = symbols("Nx Ny")

y_window_range = 3


def generate_numbering(x_window_range):
    xrange = range(-x_window_range // 2 + 1, x_window_range // 2 + 1)
    varnumbering = np.repeat(
        np.array([[np.repeat([0], 5)],
                  [np.repeat([1], 5)],
                  [np.repeat([2], 5)],
                  [np.repeat([2], 5)],
                  [np.repeat([2], 5)]]).flatten("F")[np.newaxis, :],
        3,
        axis=0)

    xnumbering = np.repeat(
        np.array([[si + idx for idx in xrange],
                  [si + idx for idx in xrange],
                  [si + idx for idx in xrange],
                  [si + idx for idx in xrange],
                  [si + idx for idx in xrange]]).flatten("F")[np.newaxis, :],
        3,
        axis=0)

    ynumbering = np.repeat(
        np.array([[0 for idx in xrange], [0 for idx in xrange],
                  [sj - 1 for idx in xrange],
                  [sj for idx in xrange],
                  [sj + 1 for idx in xrange]]).flatten("F")[np.newaxis, :],
        3,
        axis=0)
    return varnumbering, xnumbering, ynumbering


def generate_bulk():
    """for i in [2, Nx - 2] and j in [1, Ny - 1] """
    F = np.array([dth, dtq, dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1",
          "h_ip2"], ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1",
          "T_ip2_jm1"], ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"],
         ["T_im2_jp1", "T_im1_jp1", "T_i_jp1", "T_ip1_jp1", "T_ip2_jp1"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    J_array = np.array([[f.diff(u).simplify() for u in U_] for f in F])
    F_array = np.array(F.tolist()).flatten()

    sparse_indices = np.where(J_array != 0)
    J_sparse_array = J_array[sparse_indices]
    srows = np.repeat(
        np.array([si * (2 + sNy) + idx
                  for idx in range(2)] +
                 [si * (2 + sNy) + 2 + sj])[:, np.newaxis],
        len(U_),
        axis=1)

    varnumbering, xnumbering, ynumbering = generate_numbering(5)

    scols = (xnumbering * (sNy + 2) + ynumbering + varnumbering)

    sparse_rows = srows[sparse_indices]
    sparse_cols = scols[sparse_indices]

    return (U_, (F_array, srows[:, 0]),
            (J_sparse_array, sparse_rows, sparse_cols))


def generate_upper():
    """for i in [2, Nx - 2] and j == Ny-1 """

    F = np.array([upper_dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    varnumbering, xnumbering, ynumbering = map(lambda array: array[0]
                                               .reshape((-1, 5))[:, :4]
                                               .reshape((1, -1)),
                                               generate_numbering(5))

    scols = (xnumbering * (sNy + 2) + ynumbering + varnumbering)

    srows = np.repeat(
        np.array([si * (2 + sNy) + 1 + sNy])[:, np.newaxis],
        len(U_),
        axis=1)

    F_array = np.array(F.tolist()).flatten()

    upper_J_array = np.array([upper_dtT.diff(u) for u in U_])
    upper_sparse_indices = np.where(upper_J_array != 0)
    upper_J_sparse_array = upper_J_array[upper_sparse_indices]

    upper_sparse_rows = [
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in srows[0, upper_sparse_indices].flatten()
    ]
    upper_sparse_cols = [
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in scols[0, upper_sparse_indices].flatten()
    ]

    return (U_, (F_array, srows[:, 0]),
            (upper_J_sparse_array, upper_sparse_rows, upper_sparse_cols))


def generate_periodic_left():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([dth, dtq, dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"],
         ["T_im2_jp1", "T_im1_jp1", "T_i_jp1", "T_ip1_jp1", "T_ip2_jp1"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_right(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_0 = [u.subs(cdiff_right(0)) for u in U_]
    U_1 = [u.subs(cdiff_right(1)) for u in U_]
    U_full = np.vstack([U_0, U_1])

    F_array = np.array(F.tolist()).flatten()
    F_0 = [f.subs(cdiff_right(0)) for f in F_array]
    F_1 = [f.subs(cdiff_right(1)) for f in F_array]
    F_full = np.vstack([F_0, F_1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)

    sparse_indices = np.where(J_array != 0)
    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = generate_numbering(5)

    def eval_left_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x < 0:
            x += sNx
        return x

    left_xnumbering_func = np.vectorize(eval_left_bdc_numbering)
    left_xnumbering = left_xnumbering_func(xnumbering[np.newaxis],
                                           np.array([0, 1]).reshape((2, 1, 1)))
    left_cols = (left_xnumbering * (sNy + 2) + ynumbering + varnumbering)
    srows = np.repeat(np.array([si * (2 + sNy) + idx
                                for idx in range(2)] +
                               [si * (2 + sNy) + 2 + sj])[:, np.newaxis],
                      len(U_),
                      axis=1)

    left_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([0, 1]).reshape((2, 1, 1)))

    return (U_full,
            (F_full,
             left_rows[:2, :, 0]),
            (J_sparse_array,
             left_rows[sparse_indices],
             left_cols[sparse_indices]))


def generate_dirichlet_left():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([dth, dtq, dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"],
         ["T_im2_jp1", "T_im1_jp1", "T_i_jp1", "T_ip1_jp1", "T_ip2_jp1"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_right(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_0 = [u.subs(cdiff_right(0)) for u in U_]
    U_1 = [u.subs(cdiff_right(1)) for u in U_]
    U_full = np.vstack([U_0, U_1])

    F_array = np.array(F.tolist()).flatten()
    F_0 = [f.subs(cdiff_right(0)) for f in F_array]
    F_1 = [f.subs(cdiff_right(1)) for f in F_array]
    F_full = np.vstack([F_0, F_1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)

    sparse_indices = np.where(J_array != 0)
    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = generate_numbering(5)

    def eval_left_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x < 0:
            x = 0
        return x

    left_xnumbering_func = np.vectorize(eval_left_bdc_numbering)
    left_xnumbering = left_xnumbering_func(xnumbering[np.newaxis],
                                           np.array([0, 1]).reshape((2, 1, 1)))
    left_cols = (left_xnumbering * (sNy + 2) + ynumbering + varnumbering)
    srows = np.repeat(np.array([si * (2 + sNy) + idx
                                for idx in range(2)] +
                               [si * (2 + sNy) + 2 + sj])[:, np.newaxis],
                      len(U_),
                      axis=1)

    left_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([0, 1]).reshape((2, 1, 1)))

    return (U_full,
            (F_full,
             left_rows[:2, :, 0]),
            (J_sparse_array,
             left_rows[sparse_indices],
             left_cols[sparse_indices]))


def generate_periodic_right():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([dth, dtq, dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"],
         ["T_im2_jp1", "T_im1_jp1", "T_i_jp1", "T_ip1_jp1", "T_ip2_jp1"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_right(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_Nm2 = [u.subs(cdiff_right("Nxm2")) for u in U_]
    U_Nm1 = [u.subs(cdiff_right("Nxm1")) for u in U_]
    U_full = np.vstack([U_Nm2, U_Nm1])

    F_array = np.array(F.tolist()).flatten()
    F_Nm2 = [f.subs(cdiff_right("Nxm2")) for f in F_array]
    F_Nm1 = [f.subs(cdiff_right("Nxm1")) for f in F_array]
    F_full = np.vstack([F_Nm2, F_Nm1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)
    sparse_indices = np.where(J_array != 0)

    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = generate_numbering(5)

    def eval_right_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x - sNx >= 0:
            x -= sNx
        return x

    right_xnumbering_func = np.vectorize(eval_right_bdc_numbering)
    right_xnumbering = right_xnumbering_func(xnumbering[np.newaxis],
                                             np.array([sNx - 2,
                                                       sNx - 1])
                                             .reshape((2, 1, 1)))

    right_cols = (right_xnumbering * (sNy + 2) +
                  ynumbering[np.newaxis] + varnumbering[np.newaxis])

    srows = np.repeat(np.array([si * (2 + sNy) + idx
                                for idx in range(2)] +
                               [si * (2 + sNy) + 2 + sj])[:, np.newaxis],
                      len(U_),
                      axis=1)

    right_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([sNx - 2, sNx - 1]).reshape((2, 1, 1)))

    return (U_full,
            (F_full,
             right_rows[:2, :, 0]),
            (J_sparse_array,
             right_rows[sparse_indices],
             right_cols[sparse_indices]))


def generate_openflow_right():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([dth, dtq, dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"],
         ["T_im2_jp1", "T_im1_jp1", "T_i_jp1", "T_ip1_jp1", "T_ip2_jp1"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_right(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_Nm2 = [u.subs(cdiff_right("Nxm2")) for u in U_]
    U_Nm1 = [u.subs(cdiff_right("Nxm1")) for u in U_]
    U_full = np.vstack([U_Nm2, U_Nm1])

    F_array = np.array(F.tolist()).flatten()
    F_Nm2 = [f.subs(cdiff_right("Nxm2")) for f in F_array]
    F_Nm1 = [f.subs(cdiff_right("Nxm1")) for f in F_array]
    F_full = np.vstack([F_Nm2, F_Nm1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)
    sparse_indices = np.where(J_array != 0)

    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = generate_numbering(5)

    def eval_right_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x - sNx >= 0:
            x = sNx - 1
        return x

    right_xnumbering_func = np.vectorize(eval_right_bdc_numbering)
    right_xnumbering = right_xnumbering_func(xnumbering[np.newaxis],
                                             np.array([sNx - 2,
                                                       sNx - 1])
                                             .reshape((2, 1, 1)))

    right_cols = (right_xnumbering * (sNy + 2) +
                  ynumbering[np.newaxis] + varnumbering[np.newaxis])

    srows = np.repeat(np.array([si * (2 + sNy) + idx
                                for idx in range(2)] +
                               [si * (2 + sNy) + 2 + sj])[:, np.newaxis],
                      len(U_),
                      axis=1)

    right_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([sNx - 2, sNx - 1]).reshape((2, 1, 1)))

    return (U_full,
            (F_full,
             right_rows[:2, :, 0]),
            (J_sparse_array,
             right_rows[sparse_indices],
             right_cols[sparse_indices]))


def generate_upper_periodic_left():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([upper_dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_left(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_0 = [u.subs(cdiff_left(0)) for u in U_]
    U_1 = [u.subs(cdiff_left(1)) for u in U_]
    U_full = np.vstack([U_0, U_1])

    F_array = np.array(F.tolist()).flatten()
    F_0 = [f.subs(cdiff_left(0)) for f in F_array]
    F_1 = [f.subs(cdiff_left(1)) for f in F_array]
    F_full = np.vstack([F_0, F_1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)

    sparse_indices = np.where(J_array != 0)
    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = map(lambda array: array[0]
                                               .reshape((-1, 5))[:, :4]
                                               .reshape((1, -1)),
                                               generate_numbering(5))

    def eval_left_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x < 0:
            x += sNx
        return x

    left_xnumbering_func = np.vectorize(eval_left_bdc_numbering)
    left_xnumbering = left_xnumbering_func(xnumbering[np.newaxis],
                                           np.array([0, 1]).reshape((2, 1, 1)))
    left_cols = (left_xnumbering * (sNy + 2) + ynumbering + varnumbering)

    srows = np.repeat(
        np.array([si * (2 + sNy) + 1 + sNy])[:, np.newaxis],
        len(U_),
        axis=1)
    left_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([0, 1]).reshape((2, 1, 1)))

    upper_left_sparse_rows = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in left_rows[sparse_indices].flatten()
    ])
    upper_left_sparse_cols = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in left_cols[sparse_indices].flatten()
    ])

    return (U_full,
            (F_full,
             left_rows[:2, :, 0]),
            (J_sparse_array,
             upper_left_sparse_rows,
             upper_left_sparse_cols))


def generate_upper_dirichlet_left():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([upper_dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_left(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_0 = [u.subs(cdiff_left(0)) for u in U_]
    U_1 = [u.subs(cdiff_left(1)) for u in U_]
    U_full = np.vstack([U_0, U_1])

    F_array = np.array(F.tolist()).flatten()
    F_0 = [f.subs(cdiff_left(0)) for f in F_array]
    F_1 = [f.subs(cdiff_left(1)) for f in F_array]
    F_full = np.vstack([F_0, F_1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)

    sparse_indices = np.where(J_array != 0)
    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = map(lambda array: array[0]
                                               .reshape((-1, 5))[:, :4]
                                               .reshape((1, -1)),
                                               generate_numbering(5))

    def eval_left_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x < 0:
            x = 0
        return x

    left_xnumbering_func = np.vectorize(eval_left_bdc_numbering)
    left_xnumbering = left_xnumbering_func(xnumbering[np.newaxis],
                                           np.array([0, 1]).reshape((2, 1, 1)))
    left_cols = (left_xnumbering * (sNy + 2) + ynumbering + varnumbering)

    srows = np.repeat(
        np.array([si * (2 + sNy) + 1 + sNy])[:, np.newaxis],
        len(U_),
        axis=1)
    left_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([0, 1]).reshape((2, 1, 1)))

    upper_left_sparse_rows = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in left_rows[sparse_indices].flatten()
    ])
    upper_left_sparse_cols = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in left_cols[sparse_indices].flatten()
    ])

    return (U_full,
            (F_full,
             left_rows[:2, :, 0]),
            (J_sparse_array,
             upper_left_sparse_rows,
             upper_left_sparse_cols))


def generate_upper_periodic_right():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([upper_dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_right(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_Nm2 = [u.subs(cdiff_right("Nxm2")) for u in U_]
    U_Nm1 = [u.subs(cdiff_right("Nxm1")) for u in U_]
    U_full = np.vstack([U_Nm2, U_Nm1])

    F_array = np.array(F.tolist()).flatten()
    F_Nm2 = [f.subs(cdiff_right("Nxm2")) for f in F_array]
    F_Nm1 = [f.subs(cdiff_right("Nxm1")) for f in F_array]
    F_full = np.vstack([F_Nm2, F_Nm1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)
    sparse_indices = np.where(J_array != 0)

    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = map(lambda array: array[0]
                                               .reshape((-1, 5))[:, :4]
                                               .reshape((1, -1)),
                                               generate_numbering(5))

    def eval_right_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x - sNx >= 0:
            x -= sNx
        return x

    right_xnumbering_func = np.vectorize(eval_right_bdc_numbering)
    right_xnumbering = right_xnumbering_func(xnumbering[np.newaxis],
                                             np.array([sNx - 2,
                                                       sNx - 1])
                                             .reshape((2, 1, 1)))

    right_cols = (right_xnumbering * (sNy + 2) +
                  ynumbering[np.newaxis] + varnumbering[np.newaxis])

    srows = np.repeat(
        np.array([si * (2 + sNy) + 1 + sNy])[:, np.newaxis],
        len(U_),
        axis=1)

    right_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([sNx - 2, sNx - 1]).reshape((2, 1, 1)))

    upper_right_sparse_rows = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in right_rows[sparse_indices].flatten()
    ])
    upper_right_sparse_cols = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in right_cols[sparse_indices].flatten()
    ])

    return (U_full,
            (F_full,
             right_rows[:2, :, 0]),
            (J_sparse_array,
             upper_right_sparse_rows,
             upper_right_sparse_cols))


def generate_upper_openflow_right():
    """for i in [0, 2[ and j in [1, Ny - 1]"""
    F = np.array([upper_dtT])

    U = [["h_im2", "h_im1", "h_i", "h_ip1", "h_ip2"],
         ["q_im2", "q_im1", "q_i", "q_ip1", "q_ip2"],
         ["T_im2_jm1", "T_im1_jm1", "T_i_jm1", "T_ip1_jm1", "T_ip2_jm1"],
         ["T_im2_j", "T_im1_j", "T_i_j", "T_ip1_j", "T_ip2_j"]]

    U = np.array(U)
    U_ = U.flatten("F").tolist()
    U_ = [Symbol(u) for u in U_]

    def cdiff_right(idx):
        return {u: Symbol(str(u).replace("i", "%s" % idx)) for u in U_}

    U_Nm2 = [u.subs(cdiff_right("Nxm2")) for u in U_]
    U_Nm1 = [u.subs(cdiff_right("Nxm1")) for u in U_]
    U_full = np.vstack([U_Nm2, U_Nm1])

    F_array = np.array(F.tolist()).flatten()
    F_Nm2 = [f.subs(cdiff_right("Nxm2")) for f in F_array]
    F_Nm1 = [f.subs(cdiff_right("Nxm1")) for f in F_array]
    F_full = np.vstack([F_Nm2, F_Nm1])

    J_array = np.concatenate(
        [np.array([[f.diff(u) for u in U_i]
                   for f in F_i])[np.newaxis]
         for U_i, F_i in zip(U_full, F_full)], axis=0)
    sparse_indices = np.where(J_array != 0)

    J_sparse_array = J_array[sparse_indices]

    varnumbering, xnumbering, ynumbering = map(lambda array: array[0]
                                               .reshape((-1, 5))[:, :4]
                                               .reshape((1, -1)),
                                               generate_numbering(5))

    def eval_right_bdc_numbering(x, idx):
        x = x.subs("i", idx)
        if x - sNx >= 0:
            x = sNx - 1
        return x

    right_xnumbering_func = np.vectorize(eval_right_bdc_numbering)
    right_xnumbering = right_xnumbering_func(xnumbering[np.newaxis],
                                             np.array([sNx - 2,
                                                       sNx - 1])
                                             .reshape((2, 1, 1)))

    right_cols = (right_xnumbering * (sNy + 2) +
                  ynumbering[np.newaxis] + varnumbering[np.newaxis])

    srows = np.repeat(
        np.array([si * (2 + sNy) + 1 + sNy])[:, np.newaxis],
        len(U_),
        axis=1)

    right_rows = np.vectorize(lambda x, idx: x.subs("i", idx))(
        srows[np.newaxis],
        np.array([sNx - 2, sNx - 1]).reshape((2, 1, 1)))

    upper_right_sparse_rows = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in right_rows[sparse_indices].flatten()
    ])
    upper_right_sparse_cols = np.array([
        line.subs(sj, sNy - 1) if not isinstance(line, int) else line
        for line in right_cols[sparse_indices].flatten()
    ])

    return (U_full,
            (F_full,
             right_rows[:2, :, 0]),
            (J_sparse_array,
             upper_right_sparse_rows,
             upper_right_sparse_cols))


def generate_theano_model(periodic=True):
    def th_Min(a, b):
        if isinstance(a, T.TensorVariable) or isinstance(b, T.TensorVariable):
            return T.where(a < b, a, b)
        return min(a, b)

    def th_Max(a, b):
        if isinstance(a, T.TensorVariable) or isinstance(b, T.TensorVariable):
            return T.where(a < b, b, a)
        return max(a, b)

    def th_Heaviside(a):
        if isinstance(a, T.TensorVariable):
            return T.where(a < 0, 1, 1)
        return 0 if a < 0 else 1
    x = T.vector("x")
    y = T.vector("y")

    # x.tag.test_value = np.linspace(0, 200, 100, endpoint=False)
    # y.tag.test_value = np.linspace(0, 1, 50)

    Nx = x.size
    Ny = y.size

    i = T.arange(Nx)
    j = T.arange(Ny)

    h_user = T.vector("h")
    q_user = T.vector("q")
    T_user = T.matrix("T")

    # h_user.tag.test_value = np.cos(2 * np.pi * x.tag.test_value /
    #                                x.tag.test_value.max() * 2) * .1 + 1
    # q_user.tag.test_value = h_user.tag.test_value ** 3 / 3
    # T_user.tag.test_value = np.zeros((x.tag.test_value.size,
    #                                   y.tag.test_value.size)) + .5

    U = T.concatenate([h_user[:, None],
                       q_user[:, None],
                       T_user
                       ], axis=1).flatten()

    h_th = U.reshape((Nx, -1))[:, 0]
    q_th = U.reshape((Nx, -1))[:, 1]
    T_th = U.reshape((Nx, -1))[:, 2:]

    Ct = T.vector("Ct")
    Re = T.scalar("Re")
    We = T.scalar("We")
    Pe = T.scalar("Pe")
    B = T.scalar("B")

    # Ct.tag.test_value = np.zeros((Nx.tag.test_value,))
    # Re.tag.test_value = 15.
    # We.tag.test_value = 30.
    # Pe.tag.test_value = 200.
    # B.tag.test_value = .1

    delta_x = (x[-1] - x[0]) / (Nx - 1)

    def clip_rank(tensor):
        rank = tensor.ndim
        new_rank = (tensor
                    if rank == 2
                    else tensor.reshape([-1, *[1] * (1 - rank)]))
        return new_rank

    def align_shape(tensor, ref):
        reshaped_tensor = clip_rank(tensor)
        return T.tile(reshaped_tensor,
                      ref.shape // reshaped_tensor.shape,
                      ndim=2)

    def bulk():
        U = [[h_th[idx:idx - 4 if (idx - 4) else None, None]
              for idx in range(5)],
             [q_th[idx:idx - 4 if (idx - 4) else None, None]
              for idx in range(5)],
             [T_th[idx:idx - 4 if (idx - 4) else None, :-2]
              for idx in range(5)],
             [T_th[idx:idx - 4 if (idx - 4) else None, 1:-1]
              for idx in range(5)],
             [T_th[idx:idx - 4 if (idx - 4) else None, 2:]
              for idx in range(5)]]

        U = np.array(U).flatten('F')

        depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) = generate_bulk()

        symb_args = [*dependant_variable, *np.array(depvar).flatten().tolist(),
                     *symbolic_parameters,
                     si, sj, sNx, sNy]

        pars_th = [Re, We, Ct[2:-2, None], Pe, B, delta_x]
        th_args = [x[2:-2], y[:-2], y[1:-1], y[2:],
                   *U, *pars_th,
                   i[2:-2, None], j[None, 1:-1],
                   Nx, Ny]

        F, Fidx = [lambdify(symb_args, f.tolist(),
                            modules=[T, {"Max": th_Max,
                                         "Min": th_Min,
                                         "Heaviside":
                                         th_Heaviside}],
                            dummify=True)(*th_args)
                   for f in (F_s, Fidx_s)]

        J, rows, cols = [lambdify(symb_args,
                                  f,
                                  modules=[T, {"Max": th_Max,
                                               "Min": th_Min,
                                               "Heaviside": th_Heaviside}],
                                  dummify=True)(*th_args)
                         for f in (J_s, rows_s, cols_s)]

        F = T.concatenate(F, axis=1).reshape((-1,))
        Fidx = T.concatenate(Fidx, axis=1).reshape((-1,))

        def flatten_tensor(tensor):
            tensor = [align_shape(tens, r) for tens, r in zip(tensor, rows)]
            tensor = T.concatenate([tens.reshape((-1, ))
                                    for tens in tensor], axis=0)
            return tensor

        J, cols, rows = map(flatten_tensor, (J, cols, rows))

        return (F, Fidx), (J, cols, rows)

    def top():
        U = [
            [h_th[idx:idx - 4 if (idx - 4) else None]
             for idx in range(5)],
            [q_th[idx:idx - 4 if (idx - 4) else None]
             for idx in range(5)],
            [T_th[idx:idx - 4 if (idx - 4) else None, -2]
             for idx in range(5)],
            [T_th[idx:idx - 4 if (idx - 4) else None, -1]
             for idx in range(5)]
        ]

        U = np.array(U).flatten("F")

        depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) = generate_upper()

        symb_args = [*dependant_variable[:-1],
                     *np.array(depvar).flatten().tolist(),
                     *symbolic_parameters,
                     si, sj, sNx, sNy]

        pars_th = [Re, We, Ct[2:-2, None], Pe, B, delta_x]

        th_args = [x[2:-2], y[-2], y[-1],
                   *U, *pars_th,
                   i[2:-2, None], j[None, 1:-1],
                   Nx, Ny]
        F, Fidx = [lambdify(symb_args, f.tolist(),
                            modules=[T, {"Max": th_Max,
                                         "Min": th_Min,
                                         "Heaviside":
                                         th_Heaviside}],
                            dummify=True)(*th_args)
                   for f in (F_s, Fidx_s)]

        J, rows, cols = [lambdify(symb_args,
                                  f,
                                  modules=[T, {"Max": th_Max,
                                               "Min": th_Min,
                                               "Heaviside": th_Heaviside}],
                                  dummify=True)(*th_args)
                         for f in (J_s, rows_s, cols_s)]

        F = T.concatenate(F, axis=1).reshape((-1,))
        Fidx = T.concatenate(Fidx, axis=1).reshape((-1,))

        def flatten_tensor(tensor):
            # tensor = [align_shape(tens, r) for tens, r in zip(tensor, rows)]
            tensor = T.concatenate([tens.reshape((-1, ))
                                    for tens in tensor], axis=0)
            return tensor

        J, cols, rows = map(flatten_tensor, (J, cols, rows))

        return (F, Fidx), (J, cols, rows)

    def left():
        if periodic:
            idxs0 = [-2, -1, 0, 1, 2]
            idxs1 = [-1, 0, 1, 2, 3]
            U = [
                [
                    [h_th[idx] for idx in idxs0],
                    [q_th[idx] for idx in idxs0],
                    [T_th[idx, :-2] for idx in idxs0],
                    [T_th[idx, 1:-1] for idx in idxs0],
                    [T_th[idx, 2:] for idx in idxs0],
                ],
                [
                    [h_th[idx] for idx in idxs1],
                    [q_th[idx] for idx in idxs1],
                    [T_th[idx, :-2] for idx in idxs1],
                    [T_th[idx, 1:-1] for idx in idxs1],
                    [T_th[idx, 2:] for idx in idxs1],
                ]
            ]

        else:
            idxs0 = [0, 0, 0, 1, 2]
            idxs1 = [0, 0, 1, 2, 3]
            U = [
                [
                    [h_th[idx] for idx in idxs0],
                    [q_th[idx] for idx in idxs0],
                    [T_th[idx, :-2] for idx in idxs0],
                    [T_th[idx, 1:-1] for idx in idxs0],
                    [T_th[idx, 2:] for idx in idxs0],
                ],
                [
                    [h_th[idx] for idx in idxs1],
                    [q_th[idx] for idx in idxs1],
                    [T_th[idx, :-2] for idx in idxs1],
                    [T_th[idx, 1:-1] for idx in idxs1],
                    [T_th[idx, 2:] for idx in idxs1],
                ]
            ]

        U = np.array(U).reshape((2, -1), order="F").flatten()

        if periodic:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_periodic_left()
        else:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_dirichlet_left()

        symb_args = [*dependant_variable, *np.array(depvar).flatten().tolist(),
                     *symbolic_parameters,
                     si, sj, sNx, sNy]

        pars_th = [Re, We, Ct[0], Pe, B, delta_x]

        th_args = [x[2:-2], y[:-2], y[1:-1], y[2:],
                   *U, *pars_th,
                   i[2:-2, None], j[None, 1:-1],
                   Nx, Ny]

        F, Fidx = [lambdify(symb_args, f.tolist(),
                            modules=[T, {"Max": th_Max,
                                         "Min": th_Min,
                                         "Heaviside":
                                         th_Heaviside}],
                            dummify=True)(*th_args)
                   for f in (F_s, Fidx_s)]

        J, rows, cols = [lambdify(symb_args,
                                  f,
                                  modules=[T, {"Max": th_Max,
                                               "Min": th_Min,
                                               "Heaviside": th_Heaviside}],
                                  dummify=True)(*th_args)
                         for f in (J_s, rows_s, cols_s)]

        F = [f.reshape((1, -1))
             for f in np.array(F).flatten()]

        Fidx = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(Fidx).flatten()]

        J = [f.reshape((1, -1))
             for f in np.array(J).flatten()]

        rows = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(rows).flatten()]

        cols = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(cols).flatten()]

        F = T.concatenate(F, axis=1).reshape((-1,))
        Fidx = T.concatenate(Fidx, axis=1).reshape((-1,))

        def flatten_tensor(tensor):
            tensor = [align_shape(tens, r) for tens, r in zip(tensor, rows)]
            tensor = T.concatenate([tens.reshape((-1, ))
                                    for tens in tensor], axis=0)
            return tensor

        J, cols, rows = map(flatten_tensor, (J, cols, rows))

        return (F, Fidx), (J, cols, rows)

    def top_left():
        if periodic:
            idxs0 = [-2, -1, 0, 1, 2]
            idxs1 = [-1, 0, 1, 2, 3]
            U = [
                [
                    [h_th[idx] for idx in idxs0],
                    [q_th[idx] for idx in idxs0],
                    [T_th[idx, -2] for idx in idxs0],
                    [T_th[idx, -1] for idx in idxs0],
                ],
                [
                    [h_th[idx] for idx in idxs1],
                    [q_th[idx] for idx in idxs1],
                    [T_th[idx, -2] for idx in idxs1],
                    [T_th[idx, -1] for idx in idxs1],
                ]
            ]
        else:
            idxs0 = [0, 0, 0, 1, 2]
            idxs1 = [0, 0, 1, 2, 3]
            U = [
                [
                    [h_th[idx] for idx in idxs0],
                    [q_th[idx] for idx in idxs0],
                    [T_th[idx, -2] for idx in idxs0],
                    [T_th[idx, -1] for idx in idxs0],
                ],
                [
                    [h_th[idx] for idx in idxs1],
                    [q_th[idx] for idx in idxs1],
                    [T_th[idx, -2] for idx in idxs1],
                    [T_th[idx, -1] for idx in idxs1],
                ]
            ]

        U = np.array(U).reshape((2, -1), order="F").flatten()

        if periodic:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_upper_periodic_left()
        else:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_upper_dirichlet_left()

        symb_args = [*dependant_variable[:-1],
                     *np.array(depvar).flatten().tolist(),
                     *symbolic_parameters,
                     si, sj, sNx, sNy]
        pars_th = [Re, We, Ct[0], Pe, B, delta_x]

        th_args = [x[2:-2], y[-2], y[-1],
                   *U, *pars_th,
                   i[2:-2, None], j[None, 1:-1],
                   Nx, Ny]

        F, Fidx = [lambdify(symb_args, f.tolist(),
                            modules=[T, {"Max": th_Max,
                                         "Min": th_Min,
                                         "Heaviside":
                                         th_Heaviside}],
                            dummify=True)(*th_args)
                   for f in (F_s, Fidx_s)]

        J, rows, cols = [lambdify(symb_args,
                                  f,
                                  modules=[T, {"Max": th_Max,
                                               "Min": th_Min,
                                               "Heaviside": th_Heaviside}],
                                  dummify=True)(*th_args)
                         for f in (J_s, rows_s, cols_s)]

        F = [f.reshape((1, -1))
             for f in np.array(F).flatten()]

        Fidx = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(Fidx).flatten()]

        J = [f.reshape((1, -1))
             for f in np.array(J).flatten()]

        rows = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(rows).flatten()]

        cols = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(cols).flatten()]

        F = T.concatenate(F, axis=1).reshape((-1,))
        Fidx = T.concatenate(Fidx, axis=1).reshape((-1,))

        def flatten_tensor(tensor):
            tensor = [align_shape(tens, r) for tens, r in zip(tensor, rows)]
            tensor = T.concatenate([tens.reshape((-1, ))
                                    for tens in tensor], axis=0)
            return tensor

        J, cols, rows = map(flatten_tensor, (J, cols, rows))

        return (F, Fidx), (J, cols, rows)

    def right():
        if periodic:
            idxsNm1 = [-4, -3, -2, -1, 0]
            idxsN = [-3, -2, -1, 0, 1]
            U = [
                [[h_th[idx] for idx in idxsNm1],
                 [q_th[idx] for idx in idxsNm1],
                 [T_th[idx, :-2] for idx in idxsNm1],
                 [T_th[idx, 1:-1] for idx in idxsNm1],
                 [T_th[idx, 2:] for idx in idxsNm1]],
                [[h_th[idx] for idx in idxsN],
                 [q_th[idx] for idx in idxsN],
                 [T_th[idx, :-2] for idx in idxsN],
                 [T_th[idx, 1:-1] for idx in idxsN],
                 [T_th[idx, 2:] for idx in idxsN]]
            ]
        else:
            idxsNm1 = [-4, -3, -2, -1, -1]
            idxsN = [-3, -2, -1, -1, -1]
            U = [
                [[h_th[idx] for idx in idxsNm1],
                 [q_th[idx] for idx in idxsNm1],
                 [T_th[idx, :-2] for idx in idxsNm1],
                 [T_th[idx, 1:-1] for idx in idxsNm1],
                 [T_th[idx, 2:] for idx in idxsNm1]],
                [[h_th[idx] for idx in idxsN],
                 [q_th[idx] for idx in idxsN],
                 [T_th[idx, :-2] for idx in idxsN],
                 [T_th[idx, 1:-1] for idx in idxsN],
                 [T_th[idx, 2:] for idx in idxsN]]
            ]

        U = np.array(U).reshape((2, -1), order="F").flatten()

        if periodic:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_periodic_right()
        else:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_openflow_right()

        symb_args = [*dependant_variable, *np.array(depvar).flatten().tolist(),
                     *symbolic_parameters,
                     si, sj, sNx, sNy]

        pars_th = [Re, We, Ct[-1], Pe, B, delta_x]

        th_args = [x[2:-2], y[:-2], y[1:-1], y[2:],
                   *U, *pars_th,
                   i[2:-2, None], j[None, 1:-1],
                   Nx, Ny]

        F, Fidx = [lambdify(symb_args, f.tolist(),
                            modules=[T, {"Max": th_Max,
                                         "Min": th_Min,
                                         "Heaviside":
                                         th_Heaviside}],
                            dummify=True)(*th_args)
                   for f in (F_s, Fidx_s)]

        J, rows, cols = [lambdify(symb_args,
                                  f,
                                  modules=[T, {"Max": th_Max,
                                               "Min": th_Min,
                                               "Heaviside": th_Heaviside}],
                                  dummify=True)(*th_args)
                         for f in (J_s, rows_s, cols_s)]

        F = [f.reshape((1, -1))
             for f in np.array(F).flatten()]

        Fidx = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(Fidx).flatten()]

        J = [f.reshape((1, -1))
             for f in np.array(J).flatten()]

        rows = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(rows).flatten()]

        cols = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(cols).flatten()]

        F = T.concatenate(F, axis=1).reshape((-1,))
        Fidx = T.concatenate(Fidx, axis=1).reshape((-1,))

        def flatten_tensor(tensor):
            tensor = [align_shape(tens, r) for tens, r in zip(tensor, rows)]
            tensor = T.concatenate([tens.reshape((-1, ))
                                    for tens in tensor], axis=0)
            return tensor

        J, cols, rows = map(flatten_tensor, (J, cols, rows))

        return (F, Fidx), (J, cols, rows)

    def top_right():
        if periodic:
            idxsNm1 = [-4, -3, -2, -1, 0]
            idxsN = [-3, -2, -1, 0, 1]
            U = [
                [[h_th[idx] for idx in idxsNm1],
                 [q_th[idx] for idx in idxsNm1],
                 [T_th[idx, -2] for idx in idxsNm1],
                 [T_th[idx, -1] for idx in idxsNm1]],
                [[h_th[idx] for idx in idxsN],
                 [q_th[idx] for idx in idxsN],
                 [T_th[idx, -2] for idx in idxsN],
                 [T_th[idx, -1] for idx in idxsN]]
            ]
        else:
            idxsNm1 = [-4, -3, -2, -1, -1]
            idxsN = [-3, -2, -1, -1, -1]
            U = [
                [[h_th[idx] for idx in idxsNm1],
                 [q_th[idx] for idx in idxsNm1],
                 [T_th[idx, -2] for idx in idxsNm1],
                 [T_th[idx, -1] for idx in idxsNm1]],
                [[h_th[idx] for idx in idxsN],
                 [q_th[idx] for idx in idxsN],
                 [T_th[idx, -2] for idx in idxsN],
                 [T_th[idx, -1] for idx in idxsN]]
            ]

        U = np.array(U).reshape((2, -1), order="F").flatten()

        if periodic:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_upper_periodic_right()
        else:
            depvar, (F_s, Fidx_s), (J_s, rows_s, cols_s) =\
                generate_upper_openflow_right()

        symb_args = [*dependant_variable[:-1],
                     *np.array(depvar).flatten().tolist(),
                     *symbolic_parameters,
                     si, sj, sNx, sNy]

        pars_th = [Re, We, Ct[-1], Pe, B, delta_x]

        th_args = [x[2:-2], y[-2], y[-1],
                   *U, *pars_th,
                   i[2:-2, None], j[None, 1:-1],
                   Nx, Ny]

        F, Fidx = [lambdify(symb_args, f.tolist(),
                            modules=[T, {"Max": th_Max,
                                         "Min": th_Min,
                                         "Heaviside":
                                         th_Heaviside}],
                            dummify=True)(*th_args)
                   for f in (F_s, Fidx_s)]

        J, rows, cols = [lambdify(symb_args,
                                  f,
                                  modules=[T, {"Max": th_Max,
                                               "Min": th_Min,
                                               "Heaviside": th_Heaviside}],
                                  dummify=True)(*th_args)
                         for f in (J_s, rows_s, cols_s)]

        F = [f.reshape((1, -1))
             for f in np.array(F).flatten()]

        Fidx = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(Fidx).flatten()]

        J = [f.reshape((1, -1))
             for f in np.array(J).flatten()]

        rows = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(rows).flatten()]

        cols = [f.reshape((1, -1))
                if isinstance(f, T.TensorVariable)
                else T.constant(f).reshape((1, 1))
                for f in np.array(cols).flatten()]

        F = T.concatenate(F, axis=1).reshape((-1,))
        Fidx = T.concatenate(Fidx, axis=1).reshape((-1,))

        def flatten_tensor(tensor):
            tensor = [align_shape(tens, r) for tens, r in zip(tensor, rows)]
            tensor = T.concatenate([tens.reshape((-1, ))
                                    for tens in tensor], axis=0)
            return tensor

        J, cols, rows = map(flatten_tensor, (J, cols, rows))

        return (F, Fidx), (J, cols, rows)

    ((F_bulk, Fidx_bulk),
     (J_bulk, cols_bulk, rows_bulk)) = bulk()
    ((F_top, Fidx_top),
     (J_top, cols_top, rows_top)) = top()
    ((F_left, Fidx_left),
     (J_left, cols_left, rows_left)) = left()
    ((F_right, Fidx_right),
     (J_right, cols_right, rows_right)) = right()
    ((F_top_left, Fidx_top_left),
     (J_top_left, cols_top_left, rows_top_left)) = top_left()
    ((F_top_right, Fidx_top_right),
     (J_top_right, cols_top_right, rows_top_right)) = top_right()

    Fdata = T.concatenate([F_bulk, F_left, F_right,
                           F_top, F_top_right, F_top_left], axis=0)
    Fidx = T.concatenate([Fidx_bulk, Fidx_left, Fidx_right,
                          Fidx_top, Fidx_top_right, Fidx_top_left], axis=0)
    F = T.zeros(shape=(Nx * (2 + Ny),))
    F = T.set_subtensor(F[Fidx.reshape((-1, ))], Fdata)

    J = T.concatenate([J_bulk, J_left, J_right,
                       J_top, J_top_right, J_top_left], axis=0)
    cols = T.concatenate([cols_bulk, cols_left, cols_right,
                          cols_top, cols_top_right, cols_top_left], axis=0)
    rows = T.concatenate([rows_bulk, rows_left, rows_right,
                          rows_top, rows_top_right, rows_top_left], axis=0)
    return {"F_func": ([x, y, h_user, q_user, T_user, Ct, Re, We, Pe, B],
                       F),
            "J_func": ([x, y, h_user, q_user, T_user, Ct, Re, We, Pe, B],
                       (J, rows, cols))}


class FullFourierModel:
    def __init__(self, model_th, periodic):
        self._J_func = J_func = th.function(*model_th["J_func"],
                                            on_unused_input="ignore",
                                            allow_input_downcast=True)
        self._F_func = F_func = th.function(*model_th["F_func"],
                                            on_unused_input="ignore",
                                            allow_input_downcast=True)

        def J(fields, pars):
            if isinstance(pars["Ct"], float):
                pars["Ct"] = np.zeros(fields["x"].shape) + pars["Ct"]
            fields = map(fields.get, [*"xyhqT"])
            pars = map(pars.get, "Ct, Re, We, Pe, B".split(", "))
            Jdata, rows, cols = J_func(*fields, *pars)
            return sps.csc_matrix((Jdata, (rows, cols)))

        def F(fields, pars):
            if isinstance(pars["Ct"], float):
                pars["Ct"] = np.zeros(fields["x"].shape) + pars["Ct"]
            fields = map(fields.get, [*"xyhqT"])
            pars = map(pars.get, "Ct, Re, We, Pe, B".split(", "))
            return F_func(*fields, *pars)

        self.F = F
        self.J = J
        self.pars_name = ("Ct", "Re", "We", "Pe", "B")
        self.periodic = periodic
        self.fields_template = trf.core.fields.BaseFields.factory(["x", "y"],
                                                                  [("h", "x"),
                                                                   ("q", "x"),
                                                                   ("T",
                                                                    ("x",
                                                                     "y"))],
                                                                  [])
        self._indep_vars = ["x", "y"]


def generate_model(periodic=True):
    model_th = generate_theano_model(periodic=periodic)
    return FullFourierModel(model_th, periodic)
