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

    def __init__(self, config=None):
        self.config = self.getDefaultConfig()
        self.updateConfig(config)
    
    def getDefaultConfig(self):
        config = super().getDefaultConfig()
        return config
    
    def filter(self, bundle_input):
        return bundle_input
