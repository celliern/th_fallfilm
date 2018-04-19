#!/usr/bin/env python
# coding=utf8

import triflow as trf


def hydro(compiler="theano"):
    func = ["-dxq",
            """(546*dxxq*h**2 - 420*q + 1302*dxh**2*q - 9*h*q*(77*dxxh + 136*dxq*Re) -
            2*dxh*(693*dxq*h + 70*Ct*h**3 - 324*q**2*Re) + 140*h**3*(1 + dxxxh*We))/(504.*h**2*Re)"""]  # noqa
    var = ["h", "q"]
    pars = ["Re", "Ct", "We"]
    model = trf.Model(func, var, pars, compiler=compiler)
    model.name = "hydro"
    return model


def vanilla(compiler="theano"):
    func = ["-dxq",
            """(546*dxxq*h**2 - 420*q + 1302*dxh**2*q - 9*h*q*(77*dxxh + 136*dxq*Re) -
            2*dxh*(693*dxq*h + 70*Ct*h**3 - 324*q**2*Re) + 140*h**3*(1 + dxxxh*We))/(504.*h**2*Re)""",  # noqa
            """-(-dxxtheta - (21*dxh*dxphi)/(17.*h) - (28*dxh*dxtheta)/(17.*h) -
                 dxxh*((68 + 3*B*h*(41 + 19*B*h))/(34.*h*(1 + B*h)**2) + (21*phi)/(34.*h) - (2*theta)/h) +
                 (12*(-15 - 15*B*h - 7*phi - 7*B*h*phi + 15*theta + 23*B*h*theta + 8*B**2*h**2*theta))/(17.*h**2*(1 + B*h)) -
                 dxh**2*((90 + B*h*(250 + 81*B*h*(3 + B*h)))/(17.*h**2*(1 + B*h)**3) + (21*phi)/(17.*h**2) -
                    (6*(15 + 8*B*h)*theta)/(17.*h**2)) + 3*Pe*upwind((3*q)/(2.*h), theta, 3))/(3.*Pe)""",  # noqa
            """-(-dxxphi + (22*dxh*dxphi)/(17.*h) + (120*dxh*dxtheta)/(17.*h) + (3*dxq*Pe*phi)/h -
                dxxh*((3*B*(2 + 3*B*h))/(17.*(1 + B*h)**2) - (11*phi)/(17.*h)) -
                (120*(-4 - 4*B*h - 3*phi - 3*B*h*phi + 4*theta + 5*B*h*theta + B**2*h**2*theta))/(17.*h**2*(1 + B*h)) -
                dxh**2*((-6*(5 + 6*B*h)*(8 + B*h*(13 + 6*B*h)))/(17.*h**2*(1 + B*h)**3) - (56*phi)/(17.*h**2) +
                   (60*(4 + B*h)*theta)/(17.*h**2)))/(3.*Pe)"""]  # noqa
    var = ["h", "q", "theta", "phi"]
    pars = ["Re", "Ct", "We", "Pe", "B"]
    model = trf.Model(func, var, pars, compiler=compiler)
    model.name = "vanilla"
    return model


def chock(compiler="theano"):
    func = ["-dxq",
            """(546*dxxq*h**2 - 420*q + 1302*dxh**2*q - 9*h*q*(77*dxxh + 136*dxq*Re) -
            2*dxh*(693*dxq*h + 70*Ct*h**3 - 324*q**2*Re) + 140*h**3*(1 + dxxxh*We))/(504.*h**2*Re)""",  # noqa
            """(dxxtheta + (28*dxh*dxtheta)/(17.*h) + (21*dxh*(dxPhi*h + dxh*Phi))/(17.*h) -
            (12*(-15 - 7*h*Phi + 15*theta + 8*B*h*theta))/(17.*h**2) +
     (dxxh*(68 + 123*B*h + 57*B**2*h**2 + 21*h*Phi + 42*B*h**2*Phi +
     21*B**2*h**3*Phi - 68*theta - 136*B*h*theta - 68*B**2*h**2*theta))/(34.*h*(1 + B*h)**2) -
     (dxh**2*(-90 - 250*B*h - 243*B**2*h**2 - 81*B**3*h**3 - 21*h*Phi -
     63*B*h**2*Phi - 63*B**2*h**3*Phi - 21*B**3*h**4*Phi + 90*theta + 318*B*h*theta + 414*B**2*h**2*theta +
     234*B**3*h**3*theta + 48*B**4*h**4*theta))/(17.*h**2*(1 + B*h)**3) - 3*Pe*upwind((3*q)/(2.*h),theta,3))/(3.*Pe)""",  # noqa
            """(2*dxh*dxPhi - (120*dxh*dxtheta)/(17.*h) +
            dxxPhi*h + dxxh*Phi -(22*dxh*(dxPhi*h + dxh*Phi))/(17.*h) -
     (dxxh*(-6*B - 9*B**2*h + 11*Phi + 22*B*h*Phi + 11*B**2*h**2*Phi))/
     (17.*(1 + B*h)**2) + (120*(-4 - 3*h*Phi + 4*theta + B*h*theta))/(17.*h**2) +
     (2*dxh**2*(-120 - 339*B*h - 324*B**2*h**2 - 108*B**3*h**3 - 28*h*Phi -
     84*B*h**2*Phi - 84*B**2*h**3*Phi - 28*B**3*h**4*Phi + 120*theta +
     390*B*h*theta + 450*B**2*h**2*theta +
          210*B**3*h**3*theta + 30*B**4*h**4*theta))/(17.*h**2*(1 + B*h)**3) -
          3*h*Pe*upwind((3*q)/h,Phi,3))/(6.*h*Pe)"""]  # noqa
    var = ["h", "q", "theta", "Phi"]
    pars = ["Re", "Ct", "We", "Pe", "B"]
    model = trf.Model(func, var, pars, compiler=compiler)
    model.name = "chock"
    return model


def conservative(compiler="theano"):
    func = ["-dxq",
            """(546*dxxq*h**2 - 420*q + 1302*dxh**2*q - 9*h*q*(77*dxxh + 136*dxq*Re) -
            2*dxh*(693*dxq*h + 70*Ct*h**3 - 324*q**2*Re) + 140*h**3*(1 + dxxxh*We))/(504.*h**2*Re)""",  # noqa
            """(-2*(3 + 2*B*h)*(-dxxtheta - (21*dxh*dxphi)/(17.*h) - (28*dxh*dxtheta)/(17.*h) +
        (9*dxtheta*(2 + B*h)*Pe*q)/(2.*h*(3 + 2*B*h)) - (3*B*dxq*Pe*theta)/(2.*(3 + 2*B*h)) -
        dxxh*((68 + 3*B*h*(41 + 19*B*h))/(34.*h*(1 + B*h)**2) + (21*phi)/(34.*h) - (2*theta)/h) +
        (12*(-15 - 7*phi + 15*theta + 8*B*h*theta))/(17.*h**2) -
        dxh**2*((90 + B*h*(250 + 81*B*h*(3 + B*h)))/(17.*h**2*(1 + B*h)**3) +
           (21*phi)/(17.*h**2) - (6*(15 + 8*B*h)*theta)/(17.*h**2))))/(3.*(6 + 4*B*h)*Pe)""",  # noqa
            """(dxxphi - (22*dxh*dxphi)/(17.*h) - (120*dxh*dxtheta)/(17.*h) - (3*dxq*Pe*phi)/h -
      (dxxh*(-6*B*h - 9*B**2*h**2 + 11*phi + 22*B*h*phi + 11*B**2*h**2*phi))/
       (17.*h*(1 + B*h)**2) + (120*(-4 - 3*phi + 4*theta + B*h*theta))/(17.*h**2) +
      (2*dxh**2*(-120 - 339*B*h - 324*B**2*h**2 - 108*B**3*h**3 - 28*phi - 84*B*h*phi -
           84*B**2*h**2*phi - 28*B**3*h**3*phi + 120*theta + 390*B*h*theta + 450*B**2*h**2*theta +
           210*B**3*h**3*theta + 30*B**4*h**4*theta))/(17.*h**2*(1 + B*h)**3))/(3.*Pe)"""]  # noqa
    var = ["h", "q", "theta", "phi"]
    pars = ["Re", "Ct", "We", "Pe", "B"]
    model = trf.Model(func, var, pars, compiler=compiler)
    model.name = "conservative"
    return model


def fourier(periodic=True):
    from .model_fourier import generate_model
    model = generate_model(periodic=periodic)
    model.name = "Fourier"
    return model
