# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticSphericalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeSphere import ShapeSphere

@dochelper
class XicsrtOpticSphericalCrystal(InteractCrystal, ShapeSphere):
    """
    A spherical crystal optic.
    """

    pass
