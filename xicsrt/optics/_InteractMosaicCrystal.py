# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Define the :class:`InteractMosaicCrystal` class.
"""

import numpy as np

from xicsrt.util import profiler

from xicsrt.tools import xicsrt_spread
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._InteractCrystal import InteractCrystal

@dochelper
class InteractMosaicCrystal(InteractCrystal):
    """
    A class to handle Mosaic Crystal optics.

    .. Todo::
      InteractMosaicCrystal efficiency could be improved by including a
      pre-filter. The pre-filter would use a step-function rocking curve to
      exclude rays that are outside the likely range of reflection with the
      current mosaic spread.
    """

    def default_config(self):
        """
        mosaic_spread : float (0.0) [radians]
          The fwhm of the Gaussian distribution of crystalite normals around
          the nominal surface normal.

        mosaic_depth : int (15)
          The number of crystalite layers to model. This value will depend
          on the crystal structure and the incident x-ray energy.
        """
        config = super().default_config()
        config['mosaic_spread'] = 0.0
        config['mosaic_depth'] = 15
        return config

    def interact(self, rays, xloc, norm, mask=None):
        """
        Model reflections from a mosaic crystal using a multi-layer model.
        This is meant to simulate the penetration of x-rays into the HOPG until
        the rays either encounter a crystalite that satisfies the Bragg
        condition or get absorbed. This method of calculation replicates both
        the HOPG 'focusing' qualities as well as the expected throughput.
        """
        if mask is None: mask = rays['mask']

        mosaic_mask = np.zeros(rays['mask'].shape, dtype=np.bool)

        for ii in range(self.param['mosaic_depth']):
            temp_mask = (~ mosaic_mask) & mask
            self.log.debug('  Mosaic iteration: {} rays: {}'.format(ii, sum(temp_mask)))
            norm_mosaic = self.mosaic_normals(norm, temp_mask)
            temp_mask = self.angle_check(rays, norm_mosaic, temp_mask)
            rays = self.reflect_vectors(rays, xloc, norm_mosaic, temp_mask)
            mosaic_mask[temp_mask] = True
        mask &= mosaic_mask
        rays['mask'] = mask

        return rays

    def mosaic_normals(self, normals, mask, copy=True):
        """
        Add mosaic spread to the normals.
        Generates a list of crystallite normal vectors in a Gaussian
        distribution around the nominal surface normals.
        """
        m = mask

        rad_hwhm = self.param['mosaic_spread']/2.0

        profiler.start('mosaic: vector_dist_flat_gaussian')
        dir_local = xicsrt_spread.vector_dist_flat_gaussian(rad_hwhm, np.sum(m))
        profiler.stop('mosaic: vector_dist_flat_gaussian')

        R = np.empty((np.sum(m), 3, 3,), dtype=np.float64)

        # Create two vectors perpendicular to the surface normal,
        # it doesn't matter how they are oriented otherwise.
        R[:, 0, :]  = np.cross(normals[m], [0,0,1])
        R[:, 0, :] /= np.linalg.norm(R[:, 0, :], axis=1)[:, np.newaxis]
        R[:, 1, :]  = np.cross(normals[m], R[:, 0, :])
        R[:, 1, :] /= np.linalg.norm(R[:, 1, :], axis=1)[:, np.newaxis]
        R[:, 2, :] = normals[m]

        if copy:
            norm_out = normals.copy()
        else:
            norm_out = normals

        norm_out[m] = np.einsum('ij,ijk->ik', dir_local, R)
        return norm_out


