# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticPlanarMosaicCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMosaicCrystal import InteractMosaicCrystal
from xicsrt.optics._ShapeSphere import ShapeSphere

@dochelper
class XicsrtOpticSphericalMosaicCrystal(InteractMosaicCrystal, ShapeSphere):
    """
    A shperical mosaic crystal optic.
    """

    pass
