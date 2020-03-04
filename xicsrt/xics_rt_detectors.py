# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:09:40 2017
Edited on Fri Sep 06 10:41:00 2019

@author: James
@editor: Eugene

Description
-----------
The detector object collects rays and compiles them into a .tif image. It has
a position and rotation in 3D space, as well as a height and width.
"""
from PIL import Image
import numpy as np

from xicsrt.xics_rt_optics import GenericOptic

class Detector(GenericOptic):
    """
    A class for detectors.
    For now a detector is exactly the same as a generic planar optic.
    """
    pass
        
