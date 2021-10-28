# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`InteractMirror` class.
"""

import numpy as np
from copy import deepcopy


from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractObject import InteractObject
from xicsrt.tools import xicsrt_math as xm

@dochelper
class InteractMirror(InteractObject):
    """
    A perfect mirror interaction.
    """

    def interact(self, rays, xloc, norm, mask=None):
        rays = self.reflect_vectors(rays, xloc, norm, mask)
        if mask is not None:
            rays['mask'] = mask
        return rays

    def reflect_vectors(self, rays, xloc, normals, mask=None):
        if mask is None: mask = rays['mask']

        O = rays['origin']
        D = rays['direction']
        m = mask

        # Perform reflection around normal vector, creating updating rays with
        # new origin O = X and new direction D
        O[:] = xloc[:]
        D[m] -= 2 * (np.einsum('ij,ij->i', D[m], normals[m])[:, np.newaxis]
                     * normals[m])

        return rays
