# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticEllipticalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeEllipsoidCylinder import ShapeEllipsoidCylinder

@dochelper
class XicsrtOpticCrystalEllipsoidCylinder(InteractCrystal, ShapeEllipsoidCylinder):
    """
    A elliptical crystal optic.
    """

    pass
