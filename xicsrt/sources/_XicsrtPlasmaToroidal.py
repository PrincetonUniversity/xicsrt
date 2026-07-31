# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Opus 5, Fable 5)
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
    A plasma object with toroidal geometry and a circular cross-section.
    """
        
    def default_config(self):
        config = super().default_config()
        config['major_radius'] = 0.0
        config['minor_radius'] = 0.0
        config['torus_origin'] = np.array([0.0, 0.0, 0.0])
        config['emissivity_scale']    = 1.0
        config['temperature_scale']   = 1.0
        config['temperature_e_scale'] = 1.0
        config['velocity_scale']      = 1.0
        return config

    def flx_from_car(self, point_car):
        """
        Convert cartesian points to flux coordinates [rho, theta, zeta],
        where rho = r/minor_radius is the normalized minor radius.

        Accepts either a single point of shape (3,) or an array of points
        of shape (N, 3); the flux-label transformation is applied along the
        last axis in either case.
        """
        point_flx = xm.tor_from_car(point_car - self.param['torus_origin'], self.param['major_radius'])
        point_flx[..., 0] /= self.param['minor_radius']
        return point_flx

    def rho_from_car(self, point_car):
        point_flx = self.flx_from_car(point_car)
        return point_flx[..., 0]

    def car_from_flx(self, point_flx):
        point_tor = copy(point_flx)
        point_tor[...,0] = point_tor[...,0]*self.param['minor_radius']
        point_car = xm.car_from_tor(point_tor, self.param['major_radius'])
        return point_car

    def bundle_generate(self, bundle_input):

        profiler.start("Bundle Input Generation")
        m = bundle_input['mask']

        # Convert from cartesian coordinates to flux coordinates for all
        # bundles in a single vectorized operation. Bundles outside the
        # last closed flux surface are masked out below (non-finite
        # profile values). The profile hooks receive full-length arrays
        # (see the XicsrtPlasmaGeneric hook convention); the mask is
        # applied to the results here.
        profiler.start("Fluxspace from Realspace")
        point_flx = self.flx_from_car(bundle_input['origin'])
        rho = point_flx[..., 0].copy()
        rho[rho > 1.0] = np.nan
        point_flx[..., 0] = rho
        profiler.stop("Fluxspace from Realspace")

        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m]   = np.broadcast_to(
            self.get_temperature(rho) * self.param['temperature_scale'],
            rho.shape)[m]
        bundle_input['temperature_e'][m] = np.broadcast_to(
            self.get_temperature_e(rho) * self.param['temperature_e_scale'],
            rho.shape)[m]
        bundle_input['emissivity'][m]    = np.broadcast_to(
            self.get_emissivity(rho) * self.param['emissivity_scale'],
            rho.shape)[m]
        bundle_input['velocity'][m]      = np.broadcast_to(
            np.asarray(self.get_velocity(point_flx)) * self.param['velocity_scale'],
            (len(rho), 3))[m]

        fintest = np.isfinite(bundle_input['temperature'])

        m &= fintest

        profiler.stop("Bundle Input Generation")

        return bundle_input
