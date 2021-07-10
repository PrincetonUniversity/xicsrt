# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir A. Pablant <nablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""

import logging
import numpy as np

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric
from xicsrt.tools.xicsrt_math import cyl_from_car

@dochelper
class  XicsrtPlasmaCylindrical(XicsrtPlasmaGeneric):
    """
    A cylindrical plasma oriented along the Y axis.

    .. Warning::
      This class is broken and out of date and needs to be updated.

    This class is meant only to be used as an example for generating
    more complicated classes for specific plasmas.

    plasma normal           = absolute X
    plasma x orientation    = absolute Z
    plasma y orientation    = absolute Y
    """
                
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.param['xsize']/2,  self.param['xsize']/2,  self.param['bundle_count'])
        y_offset = np.random.uniform(-1 * self.param['ysize']/2, self.param['ysize']/2, self.param['bundle_count'])
        z_offset = np.random.uniform(-1 * self.param['zsize']/2,  self.param['zsize']/2,  self.param['bundle_count'])
                 
        bundle_input['origin'][:] = (
            self.origin
            + np.einsum('i,j', x_offset, self.xaxis)
            + np.einsum('i,j', y_offset, self.yaxis)
            + np.einsum('i,j', z_offset, self.zaxis))       

        #convert from cartesian coordinates to cylindrical coordinates [radius, azimuth, height]
        radius, azimuth, height = cyl_from_car(np.array([z_offset, x_offset, y_offset]))
        
        # Let plasma temperature and emissivity fall off as a function of
        # radius.
        bundle_input['emissivity'][:]  = self.emissivity / radius
        bundle_input['temperature'][:] = self.temperature / radius
        bundle_input['velocity'][:]  = self.velocity
        
        return bundle_input
    
