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
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
from xicsrt.xicsrt_math    import cart2cyl, cart2toro

class  XicsrtPlasmaToroidal(XicsrtPlasmaGeneric):
    """
    A plasma object with toroidal geometery.

    This class is meant only to be used as an exmple for generating 
    more complecated classes for specific plasmas.
    """
        
    def get_default_config(self):
        config = super().get_default_config()
        config['major_radius'] = 0.0
        config['minor_radius'] = 0.0
        return config
        
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.param['width']/2 , self.param['width']/2,  self.param['bundle_count'])
        y_offset = np.random.uniform(-1 * self.param['height']/2, self.param['height']/2, self.param['bundle_count'])
        z_offset = np.random.uniform(-1 * self.param['depth']/2 , self.param['depth']/2,  self.param['bundle_count'])
        
        #unlike the other plasmas, the toroidal plasma has fixed orientation to
        #prevent confusion
        bundle_input['origin'][:] = (
            self.origin
            + np.einsum('i,j', x_offset, np.array([1, 0, 0]))
            + np.einsum('i,j', y_offset, np.array([0, 1, 0]))
            + np.einsum('i,j', z_offset, np.array([0, 0, 1])))
        
        #convert from cartesian coordinates to toroidal coordinates [sigma, tau, phi]
        #torus is oriented along the Z axis
        rad, pol, tor = cart2toro(x_offset, y_offset, z_offset, self.param['major_radius'])
        
        step_test = (rad <= self.param['minor_radius'])

        # Let plasma temperature and emissivity fall off as a function of
        # radius.
        bundle_input['emissivity'][step_test]  = self.param['emissivity'] / radius
        bundle_input['temperature'][step_test] = self.param['temperature'] / radius
        bundle_input['velocity'][step_test]    = self.param['velocity']
        
        return bundle_input
