# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticPlanarMirror` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMirror import InteractMirror
from xicsrt.optics._ShapePlane import ShapePlane

@dochelper
class XicsrtOpticPlanarMirror(InteractMirror, ShapePlane):
    """
    A planar perfect mirror optic.
    """

    pass
