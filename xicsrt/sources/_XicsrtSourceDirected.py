# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>

"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric

@dochelper
class XicsrtSourceDirected(XicsrtSourceGeneric):
    """
    An extended rectangular ray source with rays emitted in a preferred
    direction.

    This is similar to the SourceGeneric except that an explicit direction
    can be provided instead of always emitting rays along the z-axis.

    This is different from a SourceFocused in that the emission cone is
    always aimed in a fixed direction for every location in the source.
    The SourceFocused instead aims the emission cone at a specific target
    so that the aiming direction changes for different locations within
    the source.
    """

    def default_config(self):
        """
        direction
          The direction in which to emit rays. This direction will define the
          center of the emission code with angular spread `spread`.
        """
        config = super().default_config()
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
