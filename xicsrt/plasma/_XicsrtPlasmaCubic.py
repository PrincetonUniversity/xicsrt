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

from xicsrt.plasma._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
from xicsrt.util import profiler

class XicsrtPlasmaCubic(XicsrtPlasmaGeneric):
    """
    A cubic plasma.
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
        
        #evaluate temperature at each point
        #plasma cube has consistent temperature throughout
        bundle_input['temperature'][:] = self.param['temperature']
        
        #evaluate emissivity at each point
        #plasma cube has a constant emissivity througout.
        bundle_input['emissivity'][:] = self.param['emissivity']
            
        return bundle_input
