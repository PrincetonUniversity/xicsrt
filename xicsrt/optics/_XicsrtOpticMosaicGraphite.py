# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import numpy as np

from xicsrt.tools import xicsrt_dist
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

@dochelper
class XicsrtOpticMosaicGraphite(XicsrtOpticCrystal):
    """
    A class to handle Mosaic Graphite optics.

    .. Todo::
      XicsrtOpticMosaicGraphite efficiency could be improved by
      including pre-filter. The prefilter would use a step-function
      rocking curve to exclude rays that are outside the likely range
      of reflection with the current mosaic spread.
    """

    def default_config(self):
        config = super().default_config()
        config['mosaic_spread'] = 0.0
        config['mosaic_depth'] = 15
        return config

    def trace(self, rays):
        """
        Reimplement the light method to allow testing of multiple
        mosaic crystal angles. This is meant to simulate the penetration
        of x-rays into the HOPG until the eventually encounter a
        crystalite that satisfies the Bragg condition. This method
        of calculation replicates both the HOPG 'focusing' qualities
        as well as the expected througput.
        """
        m = rays['mask']

        mosaic_mask = np.zeros(rays['mask'].shape, dtype=np.bool)

        if self.param['use_meshgrid'] is False:
            distance = self.intersect(rays)
            X, rays  = self.intersect_check(rays, distance)
            self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
            for ii in range(self.param['mosaic_depth']):
                temp_mask = (~ mosaic_mask) & m
                self.log.debug('  Mosaic iteration: {} rays: {}'.format(ii, sum(temp_mask)))
                normals  = self.generate_normals(X, rays, temp_mask)
                rays     = self.reflect_vectors(X, rays, normals, temp_mask)
                mosaic_mask[temp_mask] = True
            m[:] &= mosaic_mask
            self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
        else:
            X, rays, hits = self.mesh_intersect_1(rays, self.param['mesh'])
            self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
            for ii in range(self.param['mosaic_depth']):
                temp_mask = (~ mosaic_mask) & m
                self.log.debug('  Mosaic iteration: {} rays: {}'.format(ii, sum(temp_mask)))
                normals = self.mesh_normals(X, rays, hits, temp_mask)
                rays = self.reflect_vectors(X, rays, normals, temp_mask)
                mosaic_mask[temp_mask] = True
            m[:] &= mosaic_mask
            self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))

        return rays

    def mosaic_normals(self, normals, rays, mask=None):
        """
        Add mosaic spread to the normals.
        """
        if mask is None:
            mask = rays['mask']

        # Pulled from Novi's FocusedExtendedSource
        # Generates a list of crystallite norms normally distributed around the
        # average graphite mirror norm
        m = mask

        rad_spread = self.param['mosaic_spread']
        dir_local = xicsrt_dist.vector_dist_gaussian(rad_spread, np.sum(m))

        R = np.empty((np.sum(m), 3, 3,), dtype=np.float64)

        # Create two vectors perpendicular to the surface normal,
        # it doesn't matter how they are oriented otherwise.
        R[:, 0, :]  = np.cross(normals[m], [0,0,1])
        R[:, 0, :] /= np.linalg.norm(R[:, 0, :], axis=1)[:, np.newaxis]
        R[:, 1, :]  = np.cross(normals[m], R[:, 0, :])
        R[:, 1, :] /= np.linalg.norm(R[:, 1, :], axis=1)[:, np.newaxis]
        R[:, 2, :] = normals[m]

        normals[m] = np.einsum('ij,ijk->ik', dir_local, R)
        return normals

    def generate_normals(self, X, rays, mask=None):
        normals = super().generate_normals(X, rays)
        normals = self.mosaic_normals(normals, rays, mask)
        return normals
    
    def mesh_normals(self, X, rays, hits, mask=None):
        normals = super().mesh_normals(hits, self.param['mesh'])
        normals = self.mosaic_normals(normals, rays, mask)
        return normals

