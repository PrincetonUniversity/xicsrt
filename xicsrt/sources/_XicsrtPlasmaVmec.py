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

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric

import stelltools

@dochelper
class XicsrtPlasmaVmec(XicsrtPlasmaGeneric):

    def default_config(self):
        config = super().default_config()
        config['wout_file']         = None
        config['emissivity_scale']  = 1.0
        config['temperature_scale'] = 1.0
        config['velocity_scale']    = 1.0
        return config
        
    def initialize_vmec(self, wout=None):
        if wout is None:
            wout = self.param['wout_file']
        stelltools.initialize_from_wout(wout)

    def flx_from_car(self, point_car):
        return stelltools.flx_from_car(point_car)

    def car_from_flx(self, point_flx):
        return stelltools.car_from_flx(point_flx)

    def flx_from_cyl(self, point_cyl):
        return stelltools.flx_from_cyl(point_cyl)

    def cyl_from_flx(self, point_flx):
        return stelltools.cyl_from_flx(point_flx)

    def cyl_from_car(self, point_car):
        return stelltools.cyl_from_car(point_car)

    def car_from_cyl(self, point_cyl):
        return stelltools.car_from_cyl(point_cyl)

    def rho_from_car(self, point_car):
        point_flx = self.flx_from_car(point_car)
        return np.sqrt(point_flx[0])

    def bundle_generate(self, bundle_input):
        self.initialize_vmec()
        
        profiler.start("Bundle Input Generation")
        m = bundle_input['mask']

        # Attempt to generate the specified number of bundles, but throw out
        # bundles that our outside of the last closed flux surface.
        #
        # Currently stelltools can only handle one point at a time, so a
        # loop is required. This will be improved eventually.
        rho = np.zeros(len(m[m]))
        for ii in range(len(m[m])):
            # convert from cartesian coordinates to normalized radial coordinate.
            profiler.start("Fluxspace from Realspace")
            try:
                rho[ii] = self.rho_from_car(bundle_input['origin'][m][ii,:])
            except stelltools.DomainError:
                rho[ii] = np.nan
            profiler.stop("Fluxspace from Realspace")
        
        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m] = self.get_temperature(rho) * self.param['temperature_scale']
        bundle_input['emissivity'][m]  = self.get_emissivity(rho)  * self.param['emissivity_scale']
        bundle_input['velocity'][m]    = self.get_velocity(rho)    * self.param['velocity_scale']
        
        fintest = np.isfinite(bundle_input['temperature'])
        m &= fintest
        
        profiler.stop("Bundle Input Generation")

        return bundle_input
