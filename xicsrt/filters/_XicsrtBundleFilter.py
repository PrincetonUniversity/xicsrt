# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._ConfigObject import ConfigObject

@dochelper
class XicsrtBundleFilter(ConfigObject):
    """
    A base class for bundle filters.
    """    
    def initialize(self):
        super().initialize()
    
    def default_config(self):
        config = super().default_config()
        return config
    
    def filter(self, bundle_input):
        """
        This is the main filtering method that must be reimplemented
        for specific filter objects.
        """
        return bundle_input
