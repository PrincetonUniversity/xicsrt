# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>

"""

import numpy as np

from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric

class XicsrtSourceDirected(XicsrtSourceGeneric):
    """
    An extended rectangular ray source with rays emitted in a
    a preferred direction.

    This is essentially identical to the generic source except
    that an explicit direction can be provided.
    """

    def get_default_config(self):
        config = super().get_default_config()
        config['direction'] = None
        return config

    def initialize(self):
        super().initialize()
        # Setup a default direction along the zaxis.
        if self.param['direction'] is None:
            self.param['direction'] = self.param['zaxis']

    def make_normal(self):
        array = np.empty((self.param['intensity'], 3))
        array[:] = self.param['direction']
        normal = array / np.linalg.norm(array, axis=1)[:, np.newaxis]
        return normal
