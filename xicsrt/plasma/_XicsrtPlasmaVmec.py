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

from xicsrt.plasma._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
from xicsrt.util import profiler

import stelltools

class XicsrtPlasmaVmec(XicsrtPlasmaGeneric):

    def initialize_vmec(self, wout=None):
        if wout is None:
            wout = self.config['wout_file']
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

    def get_emissivity(self, rho):
        return self.emissivity

    def get_temperature(self, rho):
        return self.temperature

    def get_velocity(self, rho):
        return self.velocity

    def bundle_generate(self, bundle_input):

        self.initialize_vmec()
        
        profiler.start("Bundle Input Generation")
        
        m = bundle_input['mask']
        
        # create a long list containing random points within the cube's dimensions
        offset = np.zeros((self.config['bundle_count'],3))
        offset[:,0] = np.random.uniform(-1 * self.width/2, self.width/2, self.bundle_count)
        offset[:,1] = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        offset[:,2] = np.random.uniform(-1 * self.depth/2, self.depth/2, self.bundle_count)

        # unlike the other plasmas, the toroidal plasma has fixed orientation to
        # prevent confusion
        bundle_input['position'][:] = self.point_to_external(offset)

        # Attempt to generate the specified number of bundles, but throw out
        # bundles that our outside of the last closed flux surface.
        #
        # Currently stelltools can only handle one point at a time, so a
        # loop is required. This will be improved eventually.
        rho = np.zeros(len(m))
        for ii in range(self.config['bundle_count']):

            # convert from cartesian coordinates to normalized radial coordinate.
            profiler.start("Fluxspace from Realspace")
            try:
                rho[ii] = self.rho_from_car(bundle_input['position'][ii,:])
            except stelltools.DomainError:
                rho[ii] = np.nan
            profiler.stop("Fluxspace from Realspace")

        m &= np.isfinite(rho)
        
        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m] = self.get_temperature(rho[m]) * self.config['temperature_scale']
        bundle_input['emissivity'][m] = self.get_emissivity(rho[m]) * self.config['emissivity_scale']
        bundle_input['velocity'][m] = self.get_velocity(rho[m]) * self.config['velocity_scale']

        profiler.stop("Bundle Input Generation")

        return bundle_input
