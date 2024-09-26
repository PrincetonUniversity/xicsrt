# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

   Define the :class:`ShapeEllipticalCylinder` class.
"""
import sys
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject


@dochelper
class ShapeEllipsoidCylinder(ShapeObject):
    """
    A Elliptical Cylinder shape.
    This class defines intersections with a elliptical cylinder
    """

    def default_config(self):
        """
        radius_major:
            Major Radius of the Ellipse Mirror
        radius_minor:
            Minor Radius of the Ellipse Mirror
        convex:
            If True the surface will be oriented to be convex, otherwise
            the surface will be oriented to be concave.
        """
        config = super().default_config()
        config['radius_major']  = 0.5
        config['radius_minor']  = 0.2
        config['convex'] = False
        
        return config

    def initialize(self):        
        super().initialize()
        self.param['center'] = self.param['radius_major'] * self.param['zaxis'] + self.param['origin']

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calculate the distance to the intersection of the rays with the optic.
        """

        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        radius_major = self.param['radius_major']
        radius_minor = self.param['radius_minor']
        
        distances = np.zeros(m.shape, dtype=np.float64)
        dMod = np.zeros((len(m),3))
        Omod = np.zeros((len(m),3))
        
        t1        = np.zeros(m.shape, dtype=np.float64)
        t2        = np.zeros(m.shape, dtype=np.float64)
        y1        = np.zeros(m.shape, dtype=np.float64)
        y2        = np.zeros(m.shape, dtype=np.float64)
        y0        = np.zeros(m.shape, dtype=np.float64)
        m1        = np.zeros(m.shape, dtype=np.float64)
        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        Roxy2     = np.zeros(m.shape, dtype=np.float64)
        
        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        
        orig = O - self.param['center']
        sign = np.einsum('ij,ij->i', D, -orig)

        m[m] &= sign[m] > 0.0

        sin  = np.sqrt(1 - np.einsum('ij,j->i', D, yaxis) ** 2)
        
        dMod = D - np.einsum('i,j->ij', np.einsum('ij,j->i', D, yaxis), yaxis)
        
        
        m[m] &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0
        
        Omod  = orig - np.einsum('i,j->ij', np.einsum('ij,j->i', orig, yaxis), yaxis)
        
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))
        Roxy2[m] = np.einsum('ij,j->i', Omod[m], self.param['zaxis']) ** 2 * radius_minor ** 2 + np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 * radius_major ** 2
        Ocompd   = np.einsum('ij,j->i', Omod, self.param['zaxis']) * np.einsum('ij,j->i', dMod, self.param['zaxis']) * radius_minor ** 2 + np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis']) * radius_major ** 2

        a[m] = radius_minor ** 2 * np.einsum('ij,j->i', dMod[m], self.param['zaxis']) ** 2 
        a[m]+= radius_major ** 2 * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2
        b[m] = 2 * Ocompd[m]
        c[m] = Roxy2[m] - radius_minor ** 2 * radius_major ** 2

        t1[m], t2[m] = xq.multi_quadratic(a[m], b[m], c[m])

        t1[m] /= sin[m]
        t2[m] /= sin[m]

        y0[m] = np.einsum('ij,j->i', orig[m], yaxis)
        m1[m] = np.einsum('ij,j->i', D[m], yaxis)

        y1[m] = y0[m] + m1[m] * t1[m]
        y2[m] = y0[m] + m1[m] * t2[m]
        
        distances[m] = np.where((t1[m] > t2[m]) if not self.param['convex'] else (t1[m] < t2[m]), t1[m], t2[m])    
        m[m] &= distances[m] > 0.0
        m[m] &= distances[m] < sys.float_info.max

        return distances, m
    
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        pt = np.subtract(xloc[m], self.param['center'])
        pt -= np.einsum('i,j->ij',np.einsum('ij,j->i',pt, yaxis), yaxis)
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        normals[m] = pt
        
        normals[mask] = pt
        
        return normals
