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
        x_offset = np.random.uniform(-1 * self.width/2,  self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2,  self.depth/2,  self.bundle_count)        
                
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', x_offset, self.xorientation)
                  + np.einsum('i,j', y_offset, self.yorientation)
                  + np.einsum('i,j', z_offset, self.normal))
        
        #evaluate temperature at each point
        #plasma cube has consistent temperature throughout
        bundle_input['temperature'][:] = self.temperature
        
        #evaluate emissivity at each point
        #plasma cube has a constant emissivity througout.
        bundle_input['emissivity'][:] = self.emissivity
            
        return bundle_input
