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


@dochelper
class XicsrtPlasmaCubic(XicsrtPlasmaGeneric):
    """
    A cubic plasma.
    """
                
    def bundle_generate(self, bundle_input):

        bundle_input = super().bundle_generate(bundle_input)

        #evaluate temperature at each point
        #plasma cube has consistent temperature throughout
        bundle_input['temperature'][:] = self.param['temperature']
        
        #evaluate emissivity at each point
        #plasma cube has a constant emissivity througout.
        bundle_input['emissivity'][:] = self.param['emissivity']
            
        return bundle_input
