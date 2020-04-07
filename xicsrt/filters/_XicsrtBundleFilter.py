# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>
"""

import numpy as np

from xicsrt.xicsrt_objects import ConfigObject

class XicsrtBundleFilter(ConfigObject):
    """
    A base class for bundle filters.
    """    
    def initialize(self):
        super().initialize()
    
    def get_default_config(self):
        config = super().get_default_config()
        return config
    
    def filter(self, bundle_input):
        return bundle_input
