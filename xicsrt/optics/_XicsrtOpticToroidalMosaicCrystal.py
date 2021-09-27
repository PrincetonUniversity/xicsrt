# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticToroidalMosaicCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMosaicCrystal import InteractMosaicCrystal
from xicsrt.optics._ShapeTorus import ShapeTorus

@dochelper
class XicsrtOpticToroidalMosaicCrystal(InteractMosaicCrystal, ShapeTorus):
    """
    A toroidal mosaic crystal optic.
    """

    pass
