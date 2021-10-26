# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticMeshToroidalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeMeshTorus import ShapeMeshTorus

@dochelper
class XicsrtOpticMeshToroidalCrystal(InteractCrystal, ShapeMeshTorus):
    """
    A meshgrid toroidal crystal optic.
    This Optic is only meant to be used as an example of how to implement
    geometry using a meshgrid. The analytical Optic :class:`ShapeTorus` should
    be used for all normal raytracing purposes.

    Programming Notes
    -----------------
    The geometry of toroidal mesh was implemented as a separate Shape object
    (:class:`ShapeMeshTorus`) to allow mix-and-match with various Interactions.
    It would have also been possible to simply inherit
    :class:`XicsrtOpticMeshCrystal` and define the geometry here instead. Doing
    so would have avoided the need to create two separate classes and files, but
    would have limit reuse of the defined geometry.
    """

    pass
