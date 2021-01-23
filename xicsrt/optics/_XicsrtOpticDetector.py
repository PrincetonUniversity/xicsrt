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

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticMesh import XicsrtOpticMesh

@dochelper
class XicsrtOpticDetector(XicsrtOpticMesh):
    """
    A class for detectors.

    **Programming Notes**

    For now the detector class is exactly the same as the generic optic.
    In the future this may change based on defaults or how quantum efficiency is
    handled.
    """
    pass
        
