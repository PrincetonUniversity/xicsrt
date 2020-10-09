# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>

"""

import numpy as np
from collections import OrderedDict

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.filters._XicsrtBundleFilter import XicsrtBundleFilter

@dochelper
class XicsrtBundleFilterSightline(XicsrtBundleFilter):
    """
    A bundle filter based on proximity to sightline vectors.
    """

    def __init__(self, config=None, initialize=None):
        super().__init__(config)    
        
    def default_config(self):
        config = super().default_config()

        config['origin'] = None
        config['direction'] = None
        config['radius'] = None

        return config

    def filter(self, bundle_input):
        # Calculate whether the ray bundles are within range of the sightline
        # vector from sightline origin to bundle position.
        l_0 = self.config['origin'] - bundle_input['origin']
        
        # Projection of l_0 onto the sightline
        proj = np.einsum('j,ij->i', self.config['direction'], l_0)[np.newaxis]
        l_1  = np.dot(np.transpose(self.config['direction'][np.newaxis]), proj)
        l_1  = np.transpose(l_1)
        
        # Component of l_0 perpendicular to the sightline
        l_2 = l_0 - l_1
        
        # Sightline distance is the length of l_2
        distance = np.einsum('ij,ij->i', l_2, l_2) ** .5
        
        # Check to see if the bundle is close enough to the sightline
        mask = (self.config['radius'] >= distance)
        
        bundle_input['mask'] &= mask
        
        return bundle_input
