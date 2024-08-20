# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticToroidalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeTorus import ShapeTorus

@dochelper
class XicsrtOpticToroidalCrystal(InteractCrystal, ShapeTorus):
    """
    A toroidal crystal optic.
    """

    pass