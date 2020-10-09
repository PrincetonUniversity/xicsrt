# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""

import numpy as np

from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric

class XicsrtSourceFocused(XicsrtSourceGeneric):
    """
    An extended rectangular ray source that allows focusing towards a target.

    To model a planar source set depth to zero.
    """
    
    def default_config(self):
        config = super().default_config()
        config['target'] = None
        return config
    
    def generate_direction(self, origin):
        normal = self.make_normal_focused(origin)
        D = super().random_direction(normal)
        return D
    
    def make_normal_focused(self, origin):
        # Generate ray from the origin to the focus.
        normal = self.param['target'] - origin
        normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
        return normal
