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
import os

from xicsrt import xicsrt_input
from xicsrt.xicsrt_objects import ConfigObject

class XicsrtGeneralConfig(ConfigObject):
    def get_default_config(self):
        config = super().get_default_config()

        config['general'] = OrderedDict()
        config['general']['number_of_iter'] = 1
        config['general']['number_of_runs'] = 1
        config['general']['random_seed'] = None
        config['general']['pathlist_objects'] = []
        config['general']['pathlist_default'] = get_pathlist_default()
        
        config['general']['output_path'] = None
        config['general']['output_prefix'] = 'xicsrt'
        config['general']['output_suffix'] = None
        config['general']['output_run_suffix'] = None
        config['general']['image_extension'] = '.tif'
        
        config['general']['keep_meta'] = True
        config['general']['keep_images'] = True
        config['general']['keep_history'] = True

        config['general']['save_meta'] = False
        config['general']['save_images'] = False
        config['general']['save_history'] = False
        config['general']['save_run_images'] = False
        
        config['general']['print_results'] = True
        
        # config['general'][] = 

        config['sources'] = OrderedDict()
    
        config['optics'] = OrderedDict()
    
        return config
        
def get_config(config_user=None):
    obj_config =  XicsrtGeneralConfig()
    obj_config.update_config(config_user, strict=False, update=True)
    config = obj_config.get_config()
    return config

def config_to_numpy(config):
    # Temporarily just call the routine from xicsrt_input.
    # This should actually go the opposite way.
    config = xicsrt_input.config_to_numpy(config)
    return config

def get_pathlist_default():
    """
    Return a list of the default sources and optics directories.
    These locations will be based on the location of this module.
    """
    path_module = os.path.dirname(os.path.abspath(__file__))
    pathlist_default = []
    pathlist_default.append(os.path.join(path_module, 'sources'))
    pathlist_default.append(os.path.join(path_module, 'optics'))
    return pathlist_default