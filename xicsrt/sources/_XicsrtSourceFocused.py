# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric

@dochelper
class XicsrtSourceFocused(XicsrtSourceGeneric):
    """
    An extended rectangular ray source that allows focusing towards a target.

    This is different to a SourceDirected in that the emission cone is aimed
    at the target for every location in the source. The SourceDirected instead
    uses a fixed direction for emission.
    """
    
    def default_config(self):
        """
        target
          The target at which to aim the emission cone at each point in the
          source volume. The emission cone aimed at the target will have
          an angular spread defined by `spread`.
        """
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
