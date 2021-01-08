# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
"""


import numpy as np
import logging
from collections import OrderedDict
import os

from xicsrt import xicsrt_input
from xicsrt.objects._ConfigObject import ConfigObject

class XicsrtGeneralConfig(ConfigObject):

    def default_config(self):
        """

        strict_config_check: bool (True)
          Use strict checking to ensure that the config only contains valid
          options for the given object. This helps avoid unexpected behavior
          as well and alerting to typos in the configuration. When set to
          False, unrecognized config options will be ignored.

        """
        config = super().default_config()

        config['general'] = OrderedDict()
        config['general']['number_of_iter'] = 1
        config['general']['number_of_runs'] = 1
        config['general']['random_seed'] = None
        config['general']['pathlist_objects'] = []
        config['general']['pathlist_default'] = get_pathlist_default()
        config['general']['strict_config_check'] = True

        config['general']['output_path'] = None
        config['general']['output_prefix'] = 'xicsrt'
        config['general']['output_suffix'] = None
        config['general']['output_run_suffix'] = None
        config['general']['image_extension'] = '.tif'

        config['general']['keep_meta'] = True
        config['general']['keep_images'] = True
        config['general']['keep_history'] = True

        config['general']['history_max_lost'] = 10000

        config['general']['save_config'] = False
        config['general']['save_meta'] = False
        config['general']['save_history'] = False
        config['general']['save_images'] = False

        config['general']['print_results'] = True

        config['general']['make_directories'] = False

        config['sources'] = OrderedDict()
        config['optics'] = OrderedDict()
        config['filters'] = OrderedDict()
        config['scenario'] = OrderedDict()
    
        return config
        
def get_config(config_user=None):
    obj_config =  XicsrtGeneralConfig()
    obj_config.update_config(config_user, strict=False, update=True)
    config = obj_config.get_config()
    return config

def update_config(config, config_user):
    obj_config =  XicsrtGeneralConfig()
    obj_config.update_config(config, strict=False, update=True)
    obj_config.update_config(config_user, strict=False, update=True)
    config_out = obj_config.get_config()
    return config_out

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
    pathlist_default.append(os.path.join(path_module, 'filters'))
    pathlist_default.append(os.path.join(path_module, 'sources'))
    pathlist_default.append(os.path.join(path_module, 'optics'))
    return pathlist_default
