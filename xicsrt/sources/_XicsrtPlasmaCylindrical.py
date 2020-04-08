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

from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
from xicsrt.util import profiler
from xicsrt_rt_math import cart2cyl

class  XicsrtPlasmaCylindrical(XicsrtPlasmaGeneric):
    """
    A cylindrical plasma ordiented along the Y axis.

    This class is meant only to be used as an exmple for generating 
    more complecated classes for specific plasmas.

    plasma normal           = absolute X
    plasma x orientation    = absolute Z
    plasma y orientation    = absolute Y
    """
                
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.param['width']/2,  self.param['width']/2,  self.param['bundle_count'])
        y_offset = np.random.uniform(-1 * self.param['height']/2, self.param['height']/2, self.param['bundle_count'])
        z_offset = np.random.uniform(-1 * self.param['depth']/2,  self.param['depth']/2,  self.param['bundle_count'])
                 
        bundle_input['origin'][:] = (
            self.origin
            + np.einsum('i,j', x_offset, self.xaxis)
            + np.einsum('i,j', y_offset, self.yaxis)
            + np.einsum('i,j', z_offset, self.zaxis))       

        #convert from cartesian coordinates to cylindrical coordinates [radius, azimuth, height]
        radius, azimuth, height = cart2cyl(z_offset, x_offset, y_offset)
        
        # Let plasma temperature and emissivity fall off as a function of
        # radius.
        bundle_input['emissivity'][step_test]  = self.emissivity / radius
        bundle_input['temperature'][step_test] = self.temperature / radius
        bundle_input['velocity'][step_test]  = self.velocity
        
        return bundle_input
    