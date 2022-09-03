# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

A set of physcis utilities and vector convenience functions for XICSRT.
"""

import numpy as np

from scipy import constants as const

def energy_from_wavelenth(wave):
    energy = const.h * const.c / wave / const.e * 1e10
    return energy

def wavelength_from_energy(energy):
    wave = const.h * const.c / energy / const.e * 1e10
    return wave

