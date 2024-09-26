# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>


   Define the :class:`ShapeCuboid` class.
"""
import sys
import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject


@dochelper
class ShapeCuboid(ShapeObject):
    """
    A Cuboid shape defined by z=a*x^3 + a*y^3
    """

    def default_config(self):
        """
        multiplier:
            Multiplier of the cuboid equation.
        convex:
            If True the surface will be oriented to be convex, otherwise
            the surface will be oriented to be convex.
        """
        config = super().default_config()

        config['multiplier']  = 1.0
        config['convex'] = True
        
        return config

    def initialize(self):
        super().initialize()

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calculate the distance to the intersection of the rays with the optic.
        """

        Omod = rays['origin'] - self.param['origin']
        dMod = rays['direction']
        m    = rays['mask']

        mul = self.param['multiplier']

        distances = np.zeros(m.shape, dtype=np.float64)
        
        r1        = np.zeros(m.shape, dtype=complex)
        r2        = np.zeros(m.shape, dtype=complex)
        r3        = np.zeros(m.shape, dtype=complex)
        
        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        d         = np.zeros(m.shape, dtype=np.float64)

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        Roxy2     = np.zeros(m.shape, dtype=np.float64)

        Roxy2[m] = mul * (np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], yaxis) ** 2)
        Ocompd   = 2 * mul * (np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis'])  + np.einsum('ij,j->i', Omod, yaxis) * np.einsum('ij,j->i', dMod, yaxis))

        r0 = np.sqrt(np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], yaxis) ** 2)
        dr = np.sqrt(np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', dMod[m], yaxis) ** 2)

        a[m]  = mul * dr ** 3
        b[m]  = 3 * mul *  r0 * dr ** 2
        c[m]  = 3 * mul *  r0 ** 2 * dr - np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        d[m]  = a * r0 ** 3 - np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m1 = abs(b) > 0.0
        r1[m & m1], r2[m & m1], r3[m & m1] = xq.multi_cubic(a[m & m1], b[m & m1], c[m & m1], d[m & m1])

        r1[r1.imag != 0.0] = -sys.float_info.max
        r2[r1.imag != 0.0] = -sys.float_info.max
        r3[r1.imag != 0.0] = -sys.float_info.max
        
        t1, t2, t3 = r1.real, r2.real, r3.real
        t2[m & ~m1] = np.abs(d[m & ~m1] / c[m & ~m1])
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
        pr = 1 / np.sqrt(px ** 2 + py ** 2)
        
        pt = -3 * self.param['multiplier'] * (np.einsum('i,j->ij', px, self.param['xaxis']) + np.einsum('i,j->ij', py, yaxis)) + np.einsum('i,j->ij', pr, self.param['zaxis'])
        
        pt = xm.normalize(pt)
        
        normals[m] = pt
        
        return normals
