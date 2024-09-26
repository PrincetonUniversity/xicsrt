# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

   Define the :class:`ShapeCuboidCylinder` class.
"""
import sys
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject

@dochelper
class ShapeCuboidCylinder(ShapeObject):
    """
    A Cuboid Cylindrical shape defined by z = a*x^3
    """

    def default_config(self):        
        """
        multiplier:
            multiplier of the cuboid equation. This sets the value of a in z = a*x^3.
        """
        config = super().default_config()
        
        config['multiplier']  = 1.0
        
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

        mul = self.param['multiplier']

        distances = np.zeros(m.shape, dtype=np.float64)
        dMod = np.zeros((len(m),3))
        Omod = np.zeros((len(m),3))

        t1        = np.zeros(m.shape, dtype=complex)
        t2        = np.zeros(m.shape, dtype=complex)
        t3        = np.zeros(m.shape, dtype=complex)

        y1        = np.zeros(m.shape, dtype=np.float64)
        y2        = np.zeros(m.shape, dtype=np.float64)
        y3        = np.zeros(m.shape, dtype=np.float64)
        y0        = np.zeros(m.shape, dtype=np.float64)
        m1        = np.zeros(m.shape, dtype=np.float64)

        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        d         = np.zeros(m.shape, dtype=np.float64)

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        orig = O - self.param['center']

        sin  = np.sqrt(1 - np.einsum('ij,j->i', D, yaxis) ** 2)
        dMod = D - np.einsum('i,j->ij', np.einsum('ij,j->i', D, yaxis), yaxis)    

        m[m] &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0    
        Omod  = orig - np.einsum('i,j->ij', np.einsum('ij,j->i', orig, yaxis), yaxis)    
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))

        a[m] = mul * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 3
        b[m] = 3 * mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2
        c[m] = 3 * mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) - np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        d[m] = mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 3 - np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m[m] &= abs(a[m]) > 0.0

        t1[m], t2[m], t3[m] = xq.multi_cubic(a[m], b[m], c[m], d[m])
        t1[m] /= sin[m]
        t2[m] /= sin[m]
        t3[m] /= sin[m]

        t1[t1.imag != 0] = -sys.float_info.max
        t2[t2.imag != 0] = -sys.float_info.max
        t3[t3.imag != 0] = -sys.float_info.max

        R1 = np.zeros((len(t1), 3), dtype=float)
        R1[:,0] = t1.real
        R1[:,1] = t2.real
        R1[:,2] = t3.real

        y0[m] = np.einsum('ij,j->i', orig[m], yaxis)
        m1[m] = np.einsum('ij,j->i', D[m], yaxis)

        y1[m] = y0[m] + m1[m] * R1[m,0]
        y2[m] = y0[m] + m1[m] * R1[m,1]
        y3[m] = y0[m] + m1[m] * R1[m,2] 

        R1 = np.sort(R1,axis=1)
        distances[a < 0] = R1[a < 0, 1]
        distances[a > 0] = R1[a > 0, 2]
        m &= distances > 0.0

        return distances, m
    
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        pt = xloc[m] - self.param['center']
        pt -= np.einsum('i,j->ij', np.einsum('ij,j->i', pt, yaxis), yaxis)
        
        pt = np.einsum('i,j->ij', np.ones(len(pt)), self.param['zaxis']) - 3 * self.param['multiplier'] * np.einsum('i,j->ij', np.einsum('ij,j->i', pt, self.param['xaxis']), self.param['xaxis']) ** 2
        
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[m] = pt
        
        return normals
