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

from xicsrt.xics_rt_plasmas import GenericPlasma
from xicsrt.util import profiler

from mirfusion.vmec import mirtools

class FluxSurfacePlasma(GenericPlasma):
    def __init__(self, config):
        super().__init__(config)

        self.obj_vmec = None

    def initialize_vmec(self, wout=None):
        if wout is None:
            wout = self.config['wout_file']
        mirtools.initialize_from_vmec(wout)

    def flx_from_car(self, point_car):
        return mirtools.flx_from_car(point_car)

    def car_from_flx(self, point_flx):
        return mirtools.car_from_flx(point_flx)

    def flx_from_cyl(self, point_cyl):
        return mirtools.flx_from_cyl(point_cyl)

    def cyl_from_flx(self, point_flx):
        return mirtools.cyl_from_flx(point_flx)

    def cyl_from_car(self, point_car):
        return mirtools.cyl_from_car(point_car)

    def car_from_cyl(self, point_cyl):
        return mirtools.car_from_cyl(point_cyl)

    def rho_from_car(self, point_car):
        point_flx = self.flx_from_car(point_car)
        return np.sqrt(point_flx[0])

    def getEmissivity(self, rho):
        return 1.0

    def getTemperature(self, rho):
        return self.temp

    def generate_bundles(self, bundle_input):

        profiler.start("Bundle Input Generation")

        # create a long list containing random points within the cube's dimensions
        offset = np.zeros((self.config['bundle_count'],3))
        offset[:,0] = np.random.uniform(-1 * self.width / 2, self.width / 2, self.bundle_count)
        offset[:,1] = np.random.uniform(-1 * self.height / 2, self.height / 2, self.bundle_count)
        offset[:,2] = np.random.uniform(-1 * self.depth / 2, self.depth / 2, self.bundle_count)

        # unlike the other plasmas, the toroidal plasma has fixed orientation to
        # prevent confusion
        bundle_input['position'][:] = self.point_to_external(offset)

        # print('self.position', self.position)

        # This will attempt to generate the specified number of bundles, but throwout
        # bundles that our outside of the last closed flux surface.
        for ii in range(self.config['bundle_count']):

            # convert from cartesian coordinates to normalized radial coordinate.
            profiler.start("Fluxspace from Realspace")
            try:
                rho = self.rho_from_car(bundle_input['position'][ii,:])
            except mirtools.DomainError:
                rho = np.nan
            profiler.stop("Fluxspace from Realspace")

            # print(rho, bundle_input['position'][ii,0], bundle_input['position'][ii,1], bundle_input['position'][ii,2])

            if np.isfinite(rho):
                # evaluate temperature at each point
                # plasma torus temperature falls off as a function of radius
                bundle_input['temp'][ii] = self.getTemperature(rho) * self.config['temperature_factor']

                # evaluate emissivity at each point
                # plasma torus emissivity falls off as a function of radius
                bundle_input['emissivity'][ii] = self.getEmissivity(rho) * self.config['emissivity_factor']
            else:
                bundle_input['temp'][ii] = 0
                bundle_input['emissivity'][ii] = 0

        profiler.stop("Bundle Input Generation")

        return bundle_input

    def generate_rays(self):

        self.initialize_vmec()

        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.generate_bundles(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays