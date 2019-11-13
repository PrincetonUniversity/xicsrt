# -*- coding: utf-8 -*-
"""
Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
A plasma source based on a VMEC equilibrium.
"""

import numpy as np
import logging

from xicsrt.plasma.xics_rt_plasma_vmec import FluxSurfacePlasma

from mirutil import hdf5

from mirutil.classes import MirSignalObject
from mirfusion.xics.analysis.config._XcSystemPaths import XcSystemPaths


class W7xSimplePlasma(FluxSurfacePlasma):
    def __init__(self, config):
        super().__init__(config)

    def getEmissivity(self, rho):
        """
        A made up emissivity profile with moderate hollowness.
        Peak value at 1.0.
        """
        coeff = np.array(
            [24.3604, 0.0000, -160.6740, 0.0000
            ,438.7522, 0.0000, -634.2482, 0.0000
            ,509.6072, 0.0000, -212.1327, 0.0000
            ,32.6059, 0.0000, 1.2304, 0.0000
            ,0.5000])
        value = np.polyval(coeff, rho)
        return value

    def getTemperature(self, rho):
        """
        A made up temperature profile with moderate flatness.
        Peak value at 1.0
        """
        coeff = np.array(
            [8.6914, 0.0000, -39.3415, 0.0000
            ,63.1003, 0.0000, -38.8874, 0.0000
            ,2.0835, 0.0000, 6.2036, 0.0000
            ,-2.8290, 0.0000, -0.0146, 0.0000
            ,0.9999])
        value = np.polyval(coeff, rho)
        return value

