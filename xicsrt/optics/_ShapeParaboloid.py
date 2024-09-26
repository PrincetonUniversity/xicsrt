# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

   Define the :class:`ShapeParaboloid` class.
"""
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject

@dochelper
class ShapeParaboloid(ShapeObject):
    """
    A Paraboloid shape.
    """

    def default_config(self):
        """
        multiplier:
            Multiplier of the parabola equation as x = a*z^2, a is given by this configuration
        convex:
            If True the surface will be oriented to be convex, otherwise
            the surface will be oriented to be concave.
        """
        config = super().default_config()

        config['multiplier']  = 1.0
        config['convex'] = False
        
        return config

    def initialize(self):
        super().initialize()

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])

        self.param['center'] = self.param['origin'] #- self.param['zaxis'] * self.param['multiplier']

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calculate the distance to the intersection of the rays with the optic.
        """

        Omod = rays['origin'] - self.param['center']
        dMod = rays['direction']
        m    = rays['mask']
        
        mul = self.param['multiplier']

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])

        distances = np.zeros(m.shape, dtype=np.float64)
        
        t1        = np.zeros(m.shape, dtype=np.float64)
        t2        = np.zeros(m.shape, dtype=np.float64)
        
        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        Roxy2     = np.zeros(m.shape, dtype=np.float64)

        Roxy2[m] = mul * (np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], yaxis) ** 2)
        Ocompd   = 2 * mul * (np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis'])  + np.einsum('ij,j->i', Omod, yaxis) * np.einsum('ij,j->i', dMod, yaxis))

        a[m]  = mul * (np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', dMod[m], yaxis) ** 2)
        b[m]  = Ocompd[m] - np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        c[m]  = Roxy2[m] - np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m1 = abs(a) > 0.0
        t1[m & m1], t2[m & m1] = xq.multi_quadratic(a[m & m1], b[m & m1], c[m & m1])   

        t2[m & ~m1] = np.abs(c[m & ~m1] / b[m & ~m1])
        distances[m] = np.where((t1[m] > t2[m]) if not self.param['convex'] else (t1[m] < t2[m]), t1[m], t2[m])
        m[m] &= distances[m] > 0.0

        return distances, m

    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])

        pt1 = xloc[m] - self.param['origin']
        
        px = np.einsum('ij,j->i', pt1, self.param['xaxis'])
        py = np.einsum('ij,j->i', pt1, yaxis)
        pt = -2 * self.param['multiplier'] * (np.einsum('i,j->ij', px, self.param['xaxis']) + np.einsum('i,j->ij', py, yaxis)) + self.param['zaxis']
        
        pt = xm.normalize(pt)
        
        normals[m] = pt
        
        return normals
