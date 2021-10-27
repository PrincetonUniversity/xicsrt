# -*- coding: utf-8 -*-
"""
.. Authors:
    Conor Perks <cjperks@psfc.mit.edu>
    Novimir Pablant <npablant@pppl.gov>
Define the :class:`XicsrtOpticCylindricalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeCylinder import ShapeCylinder

@dochelper
class XicsrtOpticCylindricalCrystal(InteractCrystal, ShapeCylinder):
    """
    A cylindrical crystal optic.
    """
    # Passes config to calculate if 1) the ray intersects with the shape and 2) if the ray is reflected
    pass