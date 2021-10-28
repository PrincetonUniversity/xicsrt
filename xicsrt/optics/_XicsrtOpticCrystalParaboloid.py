# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticToroidalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeParaboloid import ShapeParaboloid

@dochelper
class XicsrtOpticCrystalParaboloid(InteractCrystal, ShapeParaboloid):
    """
    A toroidal crystal optic.
    """

    pass
