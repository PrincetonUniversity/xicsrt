# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>
   James Kring <jdk0026@tigermail.auburn.edu>
   Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

@dochelper
class XicsrtOpticCrystalSpherical(XicsrtOpticCrystal):

    def default_config(self):
        config = super().default_config()
        config['radius'] = 1.0
        return config

    def initialize(self):
        super().initialize()
        self.param['center'] = self.param['radius'] * self.param['zaxis'] + self.param['origin']
    
    def intersect(self, rays):
        """
        Calulate the distance to the intersection of the rays with the
        spherical optic.

        This calculation is copied from:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
        """
        #setup variables
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        d        = np.zeros(m.shape, dtype=np.float64)
        t_hc     = np.zeros(m.shape, dtype=np.float64)
        t_0      = np.zeros(m.shape, dtype=np.float64)
        t_1      = np.zeros(m.shape, dtype=np.float64)

        # L is the destance from the ray origin to the center of the sphere.
        # t_ca is the projection of this distance along the ray direction.
        L     = self.param['center'] - O
        t_ca  = np.einsum('ij,ij->i', L, D)
        
        # If t_ca is less than zero, then there is no intersection in the
        # the direction of the ray (there might be an intersection behind.)
        # Use mask to only perform calculations on rays that hit the crystal
        # m[m] &= (t_ca[m] >= 0)
        
        # d is the minimum distance between a ray and center of curvature.
        d[m] = np.sqrt(np.einsum('ij,ij->i',L[m] ,L[m]) - t_ca[m]**2)

        # If d is larger than the radius, the ray misses the sphere.
        m[m] &= (d[m] <= self.param['radius'])
        
        # t_hc is the distance from d to the intersection points.
        t_hc[m] = np.sqrt(self.param['radius']**2 - d[m]**2)
        
        t_0[m] = t_ca[m] - t_hc[m]
        t_1[m] = t_ca[m] + t_hc[m]

        # Distance traveled by the ray before hitting the optic
        distance[m] = np.where(t_0[m] > t_1[m], t_0[m], t_1[m])
        return distance

    def generate_normals(self, X, rays):
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        normals[m] = self.normalize(self.param['center'] - X[m])
        return normals

