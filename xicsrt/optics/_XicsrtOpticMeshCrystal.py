# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticMeshCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper
class XicsrtOpticMeshCrystal(InteractCrystal, ShapeMesh):
    """
    A meshgrid crystal optic.
    """

    pass
