# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""


from PIL import Image
import numpy as np

from xicsrt.optics._XicsrtOpticGeneric import XicsrtOpticGeneric

class XicsrtOpticDetector(XicsrtOpticGeneric):
    """
    A class for detectors.

    Programming Notes
    -----------------
    For now the detector class is exactly the same as the generic optic.
    In the future this may change based on defaults or how absorbtion is
    handled.
    """
    pass
        
