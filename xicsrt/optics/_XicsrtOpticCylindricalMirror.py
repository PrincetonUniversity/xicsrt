# -*- coding: utf-8 -*-
"""
.. Authors:
    Conor Perks <cjperks@psfc.mit.edu>
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticCylindricalMirror` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractMirror import InteractMirror
from xicsrt.optics._ShapeCylinder import ShapeCylinder

@dochelper
class XicsrtOpticCylindricalMirror(InteractMirror, ShapeCylinder):
    """
    A cylindrical mirror optic.
    """

    pass