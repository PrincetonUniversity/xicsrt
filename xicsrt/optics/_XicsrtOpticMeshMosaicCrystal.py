# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticMeshMosaicCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMosaicCrystal import InteractMosaicCrystal
from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper
class XicsrtOpticMeshMosaicCrystal(InteractMosaicCrystal, ShapeMesh):
    """
    A meshgrid mosaic crystal optic.
    """

    pass
