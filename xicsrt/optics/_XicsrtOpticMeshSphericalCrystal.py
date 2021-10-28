# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`XicsrtOpticMeshSphericalCrystal` class.
"""

import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal
from xicsrt.optics._ShapeMeshSphere import ShapeMeshSphere

@dochelper
class XicsrtOpticMeshSphericalCrystal(InteractCrystal, ShapeMeshSphere):
    """
    A meshgrid spherical crystal optic.
    This Optic is only meant to be used as an example of how to implement
    geometry using a meshgrid. The analytical Optic :class:`ShapeSphere` should
    be used for all normal raytracing purposes.

    Programming Notes
    -----------------
    The geometry of spherical mesh was implemented as a separate Shape object
    (:class:`ShapeMeshSphere`) to allow mix-and-match with various Interactions.
    It would have also been possible to simply inherit
    :class:`XicsrtOpticMeshCrystal` and define the geometry here instead. Doing
    so would have avoided the need to create two separate classes and files, but
    would have limit reuse of the defined geometry.
    """

    pass
