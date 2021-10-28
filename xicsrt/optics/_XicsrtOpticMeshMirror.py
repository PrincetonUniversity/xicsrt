# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticMeshMirror` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMirror import InteractMirror
from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper
class XicsrtOpticMeshMirror(InteractMirror, ShapeMesh):
    """
    A meshgrid perfect mirror optic.
    """

    pass
