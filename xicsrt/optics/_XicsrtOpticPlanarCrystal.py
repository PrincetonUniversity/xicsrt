# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticPlaneMosaicCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapePlane import ShapePlane

@dochelper
class XicsrtOpticPlanarCrystal(InteractCrystal, ShapePlane):
    """
    A planar crystal optic.
    """

    pass
