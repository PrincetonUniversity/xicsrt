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

class XicsrtSourceDirected(XicsrtSourceGeneric):
    """
    An extended rectangular ray source with rays emitted in a directed code.

    To model a point source set width, height and depth to zero.
    To model a planar source set depth to zero.
    """
    
    # The directed source is just a generic source. However we want to
    # create a new object to facilitate user selection.
    pass
