# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir A. Pablant <nablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import logging
import numpy as np
from copy import copy

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric

import xicsrt.tools.xicsrt_math as xm

@dochelper
class XicsrtPlasmaToroidal(XicsrtPlasmaGeneric):
    """
    A plasma object with toroidal geometry and a circular
    cross-section.
    """
        
    def default_config(self):
        config = super().default_config()
        config['major_radius'] = 0.0
        config['minor_radius'] = 0.0
        config['torus_origin'] = np.array([0.0, 0.0, 0.0])
        config['emissivity_scale']  = 1.0
        config['temperature_scale'] = 1.0
        config['velocity_scale']    = 1.0
        return config

    def flx_from_car(self, point_car):
        point_flx = xm.tor_from_car(point_car - self.param['torus_origin'], self.param['major_radius'])
        point_flx[0] = point_flx[0]**2
        point_flx[0] /= self.param['minor_radius']
        return point_flx

    def rho_from_car(self, point_car):
        point_flx = self.flx_from_car(point_car)
        return np.sqrt(point_flx[0])

    def car_from_flx(self, point_flx):
        point_tor = copy(point_flx)
        point_tor[...,0] = np.sqrt(point_tor[...,0])*self.param['minor_radius']
        point_car = xm.car_from_tor(point_tor, self.param['major_radius'])
        return point_car

    def bundle_generate(self, bundle_input):

        profiler.start("Bundle Input Generation")
        m = bundle_input['mask']

        # Attempt to generate the specified number of bundles, but throw out
        # bundles that our outside of the last closed flux surface.
        #
        # This loop was setup for VMEC. Here we could do this as a single
        # vectorized operation instead.
        rho = np.zeros(len(m[m]))
        for ii in range(len(m[m])):
            # convert from cartesian coordinates to normalized radial coordinate.
            profiler.start("Fluxspace from Realspace")
            rho[ii] = self.rho_from_car(bundle_input['origin'][m][ii, :])
            profiler.stop("Fluxspace from Realspace")

        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m] = self.get_temperature(rho) * self.param['temperature_scale']
        bundle_input['emissivity'][m] = self.get_emissivity(rho) * self.param['emissivity_scale']
        bundle_input['velocity'][m] = self.get_velocity(rho) * self.param['velocity_scale']

        fintest = np.isfinite(bundle_input['temperature'])

        m &= fintest

        profiler.stop("Bundle Input Generation")

        return bundle_input
