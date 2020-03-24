# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir pablant <npablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  - James Kring <jdk0026@tigermail.auburn.edu>
"""


import numpy as np
import logging
from collections import OrderedDict

from xicsrt.xicsrt_objects import ConfigObject

class XicsrtGeneralConfig(ConfigObject):
    def get_default_config(self):
        config = super().get_default_config()

        config['general'] = OrderedDict()
        config['general']['number_of_iterations'] = 1
        config['general']['number_of_runs'] = 1
        config['general']['random_seed'] = None
        config['general']['optics_pathlist'] = []

        config['sources'] = OrderedDict()
    
        config['optics'] = OrderedDict()
    
        return config
        
def get_config(config_user=None):
    obj_config =  XicsrtGeneralConfig()
    obj_config.update_config(config_user, strict=False, update=True)
    config = obj_config.get_config()
    return config
