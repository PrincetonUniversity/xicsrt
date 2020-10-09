# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir A. Pablant <nablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""

import numpy as np
import logging

import stelltools
from mirutil.math import vector

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaVmec import XicsrtPlasmaVmec


@dochelper
class XicsrtPlasmaW7xSimple(XicsrtPlasmaVmec):
    """
    A simple set of plasma profiles based on polynomials.

    This class is meant to be used for a specific XICS validation
    study undertaken by N. Pablant in 2020-02.
    """

    def default_config(self):
        config = super().default_config()
        config['enable_velocity'] = True
        return config
    
    def get_emissivity(self, flx):
        """
        A made up emissivity profile with moderate hollowness.
        Peak value at 1.0.
        """

        rho = np.sqrt(flx[:,0])
            
        # moderately Hollow profile.
        coeff = np.array(
            [24.3604, 0.0000, -160.6740, 0.0000
            ,438.7522, 0.0000, -634.2482, 0.0000
            ,509.6072, 0.0000, -212.1327, 0.0000
            ,32.6059, 0.0000, 1.2304, 0.0000
            ,0.5000])
        value = np.polyval(coeff, rho)
        return value

    def get_temperature(self, flx):
        """
        A made up temperature profile with moderate flatness.
        Peak value at 1.0
        """
        
        rho = np.sqrt(flx[:,0])
        
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

    def get_velocity(self, flx):
        """
        Calculate velocity vectors accounting for fluxspace compression effects.
        """
        profiler.start('get_velocity')
        
        num_points = flx.shape[0]
        output = np.zeros((num_points, 3))

        if self.param['enable_velocity']:
            rho = np.sqrt(flx[:,0])

            coeff = np.array(
                [5.3768, 0.0000, 32.6873, 0.0000
                 ,-179.0737, 0.0000, 65.3916, 0.0000
                 ,557.5741, 0.0000, -338.5262, 0.0000
                 ,-1697.5982, 0.0000, 3447.8770, 0.0000
                 ,-2958.4238, 0.0000, 1372.4606, 0.0000
                 ,-341.9065, 0.0000, 34.1610, 0.0000
                 ,0.0000])

            value = np.polyval(coeff, rho)

            for ii in range(num_points):
                # Get the direction normal to the flux surface.
                # This will point away from the magnetic axis.
                gradrho = stelltools.gradrho_car_from_flx(flx[ii,:])
                radial = vector.normalize(gradrho)

                # Define the parallel direction as parallel to B.
                b_car = stelltools.b_car_from_flx(flx[ii,:])
                parallel = vector.normalize(b_car)

                # Now we can define the perpendicular direction.
                perpendicular = vector.normalize(np.cross(radial, parallel))

                # Calculate the value of <|grad(rho)|>/<|B|>, which is used to scale
                # the vp factor.
                fsa_gradrho = stelltools.fsa_gradrho_from_s(flx[ii, 0])
                fsa_bmod = stelltools.fsa_modb_from_s(flx[ii, 0])
                norm_vp_factor = fsa_gradrho/fsa_bmod

                modb = np.linalg.norm(b_car)
                vp_factor = vector.magnitude(gradrho)/modb/norm_vp_factor

                output[ii,:] = perpendicular*value[ii]/vp_factor

        profiler.stop('get_velocity')
        return output

    def bundle_generate(self, bundle_input):
        """
        Similar to XicsrtPlasmaVmec.bundle_generate, except that this uses the
        full flx coordinate in calls to getEmissivity, getTemperature and
        getVelocity (as opposed to only using rho).
        """

        self.initialize_vmec()
        
        profiler.start("Bundle Input Generation")        
        m = bundle_input['mask']
        
        # Attempt to generate the specified number of bundles, but throw out
        # bundles that our outside of the last closed flux surface.
        #
        # Currently stelltools can only handle one point at a time, so a
        # loop is required. This will be improved eventually.
        flx = np.zeros((len(m),3))
        for ii in range(self.param['bundle_count']):

            # convert from cartesian coordinates to normalized radial coordinate.
            profiler.start("Fluxspace from Realspace")
            try:
                flx[ii,:] = self.flx_from_car(bundle_input['origin'][ii,:])
            except stelltools.DomainError:
                flx[ii,0] = np.nan
            profiler.stop("Fluxspace from Realspace")

        # Update the mask to only include points within the domain.
        m &= np.isfinite(flx[:,0])
        
        # evaluate emissivity, temperature and velocity at each bundle location.
        bundle_input['temperature'][m] = self.get_temperature(flx[m]) * self.param['temperature_scale']
        bundle_input['emissivity'][m] = self.get_emissivity(flx[m]) * self.param['emissivity_scale']
        bundle_input['velocity'][m] = self.get_velocity(flx[m]) * self.param['velocity_scale']
                
        profiler.stop("Bundle Input Generation")
        
        return bundle_input
