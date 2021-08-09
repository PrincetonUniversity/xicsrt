# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`InteractObject` class.
"""

import numpy as np
from copy import deepcopy

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._TraceObject import TraceObject
from xicsrt.tools import xicsrt_math as xm

@dochelper
class InteractObject(TraceObject):
    """
    The base class for interactions of rays with surfaces in XICSRT.

    This base class should be used to define behavior such as reflection,
    transmission and absorption.
    """

    def interact(self, rays, xloc, norm, mask=None):
        """
        Evaluate interaction with a surface.
        The base-class has no interaction, rays just pass through.
        """
        O = rays['origin']

        if mask is not None:
            rays['mask'][:] = mask

        # The following line does two things:
        # 1. Set the origin for rays (lost or found) with an intersection.
        # 2. Set the rays without an intersection to nan.
        O[:] = xloc[:]

        return rays
