# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>
   James Kring <jdk0026@tigermail.auburn.edu>
   Yevgeniy Yakusevich <eugenethree@gmail.com>

   Define the :class:`ShapeToroidal` class.
"""
import sys
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._ShapeObject import ShapeObject
from xicsrt.optics._multiQuartic import multi_cubic          # Multiple Quartic Solver with Order Reduction Method

@dochelper
class ShapeCuboid(ShapeObject):
    """
    A Toroidal shape.
    This class defines intersections with a torus.
    """

    def default_config(self):
        config = super().default_config()
        
        """
        Rmajor:
                Major Radius of the Torus
        Rminor:
                Minor Radius of the Torus
        index:
                Highest intesection roots
        """

        config['multiplier']  = 1.0
        config['concave'] = True
        
        return config

    def initialize(self):
        super().initialize()
        
        """
            Here, we considered Torus to be buit around y-axis, so we construct torus axis from
            system axes given such as torus z-axis goes in yaxis of the system.
            And system z-axis & x-axis goes to torus x-axis and y-axis respectively
            
            Torus center is defined as,
                Torus_Center = Torus_Major_Radius * Torus_X_axis + System_Origin
                Here, Torus_X_axis = System_Z_axis
        """
        self.Yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])

        self.param['center'] = self.param['origin'] #- self.param['zaxis'] * self.param['multiplier']

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calulate the distance to the intersection of the rays with the
        Toroidal optic.
        """
        
        """
        Calulate the distance to the intersection of the rays with the
        Toroidal optic.
        """
        
        # setup
        Omod = rays['origin'] - self.param['center']
        dMod = rays['direction']
        m    = rays['mask']
        
        mul = self.param['multiplier']
        
        distances = np.zeros(m.shape, dtype=np.float64)
        
        t1        = np.zeros(m.shape, dtype=np.float64)
        t2        = np.zeros(m.shape, dtype=np.float64)
        t3        = np.zeros(m.shape, dtype=np.float64)
        
        r1        = np.zeros(m.shape, dtype=complex)
        r2        = np.zeros(m.shape, dtype=complex)
        r3        = np.zeros(m.shape, dtype=complex)
        
        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        d         = np.zeros(m.shape, dtype=np.float64)
        r0        = np.zeros(m.shape, dtype=np.float64)
        dr        = np.zeros(m.shape, dtype=np.float64)
        Roxy2     = np.zeros(m.shape, dtype=np.float64)
        
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1 / np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))

        Roxy2[m] = mul * (np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], self.Yaxis) ** 2)
        Ocompd   = 2 * mul * (np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis'])  + np.einsum('ij,j->i', Omod, self.Yaxis) * np.einsum('ij,j->i', dMod, self.Yaxis))

        r0 = np.sqrt(np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], self.Yaxis) ** 2)
        dr = np.sqrt(np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', dMod[m], self.Yaxis) ** 2)

        a[m]  = mul * dr ** 3
        b[m]  = 3 * mul *  r0 * dr ** 2
        c[m]  = 3 * mul *  r0 ** 2 * dr - np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        d[m]  = a * r0 ** 3 - np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m1 = abs(b) > 0.0
        r1[m & m1], r2[m & m1], r3[m & m1] = multi_cubic(a[m & m1], b[m & m1], c[m & m1], d[m & m1])   

        r1[r1.imag != 0.0] = -sys.float_info.max
        r2[r1.imag != 0.0] = -sys.float_info.max
        r3[r1.imag != 0.0] = -sys.float_info.max
        
        t1, t2, t3 = r1.real, r2.real, r3.real
        t2[m & ~m1] = np.abs(d[m & ~m1] / c[m & ~m1])
        distances[m] = np.where((t1[m] > t2[m]) if self.param['concave'] else (t1[m] < t2[m]), t1[m], t2[m])    
        m[m] &= distances[m] > 0.0

        return distances, m
    
    # Generates normals
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        pt1 = xloc[m] - self.param['center']
        
        px = np.einsum('ij,j->i', pt1, self.param['xaxis'])
        py = np.einsum('ij,j->i', pt1, self.Yaxis)
        pr = 1 / np.sqrt(px ** 2 + py ** 2)
        
        pt = -3 * self.param['multiplier'] * (np.einsum('i,j->ij', px, self.param['xaxis']) + np.einsum('i,j->ij', py, self.Yaxis)) + np.einsum('i,j->ij', pr, self.param['zaxis'])
        
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[m] = pt
        
        return normals
