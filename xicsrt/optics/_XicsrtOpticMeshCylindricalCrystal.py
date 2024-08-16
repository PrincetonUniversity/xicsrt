# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticMeshCylindricalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeMeshCylinder import ShapeMeshCylinder

@dochelper
class XicsrtOpticMeshCylindricalCrystal(InteractCrystal, ShapeMeshCylinder):
    """
    A Cylindrical crystal optic implemented using a mesh.
    """

    pass
