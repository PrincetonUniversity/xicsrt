# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticDetector` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractNone import InteractNone
from xicsrt.optics._ShapePlane import ShapePlane

@dochelper
class XicsrtOpticDetector(InteractNone, ShapePlane):
    """
    A detector optic.

    Programming Notes
    -----------------
    For now the detector class simply records intersections with a plane.
    In the future this class may be expanded to include effects such as quantum
    efficiency, readout noise, dark noise, etc.
    """

    pass
