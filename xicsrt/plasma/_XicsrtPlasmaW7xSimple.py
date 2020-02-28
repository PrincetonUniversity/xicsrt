# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir A. Pablant <nablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""

import numpy as np
import logging

from xicsrt.plasma._XicsrtPlasmaVmec import XicsrtPlasmaVmec

class XicsrtPlasmaW7xSimple(XicsrtPlasmaVmec):
    """
    A simple set of plasma profiles based on polynomials.

    This class is meant to be used for a specific XICS validation
    study undertaken by N. Pablant in 2020-02.
    """

    def get_emissivity(self, rho):
        """
        A made up emissivity profile with moderate hollowness.
        Peak value at 1.0.
        """

        # moderately Hollow profile.
        coeff = np.array(
            [24.3604, 0.0000, -160.6740, 0.0000
            ,438.7522, 0.0000, -634.2482, 0.0000
            ,509.6072, 0.0000, -212.1327, 0.0000
            ,32.6059, 0.0000, 1.2304, 0.0000
            ,0.5000])
        value = np.polyval(coeff, rho)
        return value

    def get_temperature(self, rho):
        """
        A made up temperature profile with moderate flatness.
        Peak value at 1.0
        """


        if True:
            # Flat profile.
            coeff = np.array(
                [8.6914, 0.0000, -39.3415, 0.0000
                ,63.1003, 0.0000, -38.8874, 0.0000
                ,2.0835, 0.0000, 6.2036, 0.0000
                ,-2.8290, 0.0000, -0.0146, 0.0000
                ,0.9999])

        if True:
            # Moderately peaked profile.
            coeff = np.array(
                [4.6488, 0.0000, -28.2995, 0.0000
                ,70.8956, 0.0000, -92.7439, 0.0000
                ,64.3911, 0.0000, -19.0179, 0.0000
                ,-0.4734, 0.0000, -0.3997, 0.0000
                ,1.0000])

        value = np.polyval(coeff, rho)
        return value

