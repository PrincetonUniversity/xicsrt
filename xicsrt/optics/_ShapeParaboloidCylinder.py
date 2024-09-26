# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

   Define the :class:`ShapeParaboloidCylinder` class.
"""
import sys
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject


@dochelper
class ShapeParaboloidCylinder(ShapeObject):
    """
    A Parabolic Cylinder shape.
    """

    def default_config(self):
        """
        multiplier:
            Multiplier of the parabola equation as z = a*x^2, a is given by this configuration
        convex:
            If True the surface will be oriented to be convex, otherwise
            the surface will be oriented to be convex.    
        """
        config = super().default_config()
    
        config['multiplier']  = 4.0
        config['convex'] = False
        
        return config

    def initialize(self):
        super().initialize()
        self.param['center'] = self.param['origin']

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask
    
    def intersect_distance(self, rays):
        """
        Calculate the distance to the intersection of the rays with the optic.
        """
        
         # setup
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        mul    = self.param['multiplier']
        Length = self.param['length']
        
        # variable setup

        distances = np.zeros(m.shape, dtype=np.float64)
        dMod      = np.zeros((len(m),3))
        Omod      = np.zeros((len(m),3))

        t1 = np.zeros(m.shape, dtype=np.float64)
        t2 = np.zeros(m.shape, dtype=np.float64)
        y1 = np.zeros(m.shape, dtype=np.float64)
        y2 = np.zeros(m.shape, dtype=np.float64)
        y0 = np.zeros(m.shape, dtype=np.float64)
        m1 = np.zeros(m.shape, dtype=np.float64)
        a  = np.zeros(m.shape, dtype=np.float64)
        b  = np.zeros(m.shape, dtype=np.float64)
        c  = np.zeros(m.shape, dtype=np.float64)

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        orig = O - self.param['center']
    
        sin  = np.sqrt(1 - np.einsum('ij,j->i', D, yaxis) ** 2)
        dMod = D - np.einsum('i,j->ij', np.einsum('ij,j->i', D, yaxis), yaxis)    

        m[m]   &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0    
        Omod    = orig - np.einsum('i,j->ij', np.einsum('ij,j->i', orig, yaxis), yaxis)    
        dMod[m] = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))

        a[m] = mul * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2
        b[m] = 2 * mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) * np.einsum('ij,j->i', dMod[m], self.param['xaxis'])
        b[m] -= np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        c[m] = mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2
        c[m] -= np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m[m] &= a[m] > 0.0

        t1[m], t2[m] = xq.multi_quadratic(a[m], b[m], c[m])
        t1[m] /= sin[m]
        t2[m] /= sin[m]

        y0[m] = np.einsum('ij,j->i', orig[m], yaxis)
        m1[m] = np.einsum('ij,j->i', D[m], yaxis)

        y1[m] = y0[m] + m1[m] * t1[m]
        y2[m] = y0[m] + m1[m] * t2[m]

        t1[np.abs(y1) > Length / 2] = -sys.float_info.max
        t2[np.abs(y2) > Length / 2] = sys.float_info.max

        distances[m] = np.where((t1[m] > t2[m]) if not self.param['convex'] else (t1[m] < t2[m]), t1[m], t2[m])    
        m[m] &= distances[m] > 0.0
        m[m] &= distances[m] < sys.float_info.max

        return distances, m

    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        pt = xloc[m] - self.param['center']
        pt -= np.einsum('i,j->ij', np.einsum('ij,j->i', pt, yaxis), yaxis)
        
        pt = np.einsum('i,j->ij', np.ones(len(pt)), self.param['zaxis']) - 2 * self.param['multiplier'] * np.einsum('i,j->ij', np.einsum('ij,j->i', pt, self.param['xaxis']), self.param['xaxis'])
        
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[m] = pt
        
        return normals
