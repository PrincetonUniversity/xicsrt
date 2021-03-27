# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    Nathan Bartlett <nbb0011@auburn.edu>
"""
import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticGeneric import XicsrtOpticGeneric

@dochelper

class XicsrtOpticAperture(XicsrtOpticGeneric):

    def default_config(self):
        """
        opt_size:  numpy array (None)
          The size of the actual optical elements. All rays hitting outside
          the optical element will be masked.
        """
        config = super().default_config()

        return config