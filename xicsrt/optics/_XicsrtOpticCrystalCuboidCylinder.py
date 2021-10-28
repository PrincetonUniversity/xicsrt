# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticToroidalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeCuboidCylinder import ShapeCuboidCylinder

@dochelper
class XicsrtOpticCrystalCuboidCylinder(InteractCrystal, ShapeCuboidCylinder):
    """
    A toroidal crystal optic.
    """

    pass
