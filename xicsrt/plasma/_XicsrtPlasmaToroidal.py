# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir A. Pablant <nablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import logging
import numpy as np   
from collections import OrderedDict

from xicsrt.util import profiler
from xicsrt.plasma._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
from xicsrt.xics_rt_math    import cart2cyl, cart2toro

class  XicsrtPlasmaToroidal(XicsrtPlasmaGeneric):
    """
    A plasma object with toroidal geometery.

    This class is meant only to be used as an exmple for generating 
    more complecated classes for specific plasmas.
    """
    def __init__(self, config):
        super().__init__(config)

        self.major_radius   = config['major_radius']
        self.minor_radius   = config['minor_radius']
        
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2 , self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2 , self.depth/2,  self.bundle_count)        
        
        #unlike the other plasmas, the toroidal plasma has fixed orientation to
        #prevent confusion
        bundle_input['position'][:] = (
            self.position
            + np.einsum('i,j', x_offset, np.array([1, 0, 0]))
            + np.einsum('i,j', y_offset, np.array([0, 1, 0]))
            + np.einsum('i,j', z_offset, np.array([0, 0, 1])))
        
        #convert from cartesian coordinates to toroidal coordinates [sigma, tau, phi]
        #torus is oriented along the Z axis
        rad, pol, tor = cart2toro(x_offset, y_offset, z_offset, self.major_radius)
        
        step_test = (rad <= self.minor_radius)

        # Let plasma temperature and emissivity fall off as a function of
        # radius.
        bundle_input['emissivity'][step_test]  = self.emissivity / radius
        bundle_input['temperature'][step_test] = self.temperature / radius
        bundle_input['velocity'][step_test]  = self.velocity
        
        return bundle_input
