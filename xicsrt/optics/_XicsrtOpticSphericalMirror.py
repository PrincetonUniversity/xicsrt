# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticSphericalMirror` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMirror import InteractMirror
from xicsrt.optics._ShapeSphere import ShapeSphere

@dochelper
class XicsrtOpticSphericalMirror(InteractMirror, ShapeSphere):
    """
    A spherical perfect mirror optic.
    """

    pass
