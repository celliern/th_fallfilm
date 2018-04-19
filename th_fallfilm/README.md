# Thermal Falling Film, helper module

This module has for only purpose to help the computation of non insothermal falling film. The functions are documented (all public functions as docstring, or are supposed to...).

## ***Warning***

This package purpose is not to be distributed. That means that a `pip install .` will not install all the depandancies (even if an up-to-date requirements.txt is provided and can be used).

## `th_fallfilm.models`

Contains all the developped models for such films.

## `th_fallfilm.misc.helpers`

Contains some tools to initialize the fields for common cases (periodic box and exchanger plate) and some hook for the appropriate limit condition for the simulations.

## `th_fallfilm.misc.materials`

Contain all the relation between the numbers describing the fluids. For now, water has its values filled for 20°C.

## `th_fallfilm.misc.signals`

Is a minimalistic library that contains usual input signal for falling films : sinusoidal signal,
white and brown noise and constant signal. More can be implemented.

## `th_fallfilm.parametric`

Allows to generate a full design (as product of all varying parameters) or a randomized sampling
design based on a LHS sampler (see [pyDOE lhs function](https://pythonhosted.org/pyDOE/randomized.html#randomized) for more informations).

This designs feed the `generate_sample` function : it will receive receive as input a material, a dict containing fixed parameters and the design for the varying parameters.