# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`InteractNone` class.
"""

import numpy as np
from copy import deepcopy

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractObject import InteractObject

@dochelper
class InteractNone(InteractObject):
    """
    No interaction with surface, rays will pass through unchanged.
    """

    # The behavior is identical to that of the base class.
    pass