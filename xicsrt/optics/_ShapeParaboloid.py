# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>
   James Kring <jdk0026@tigermail.auburn.edu>
   Yevgeniy Yakusevich <eugenethree@gmail.com>

   Define the :class:`ShapeToroidal` class.
"""
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._ShapeObject import ShapeObject
from xicsrt.optics._multiQuartic import multi_quadratic          # Multiple Quartic Solver with Order Reduction Method

@dochelper
class ShapeParaboloid(ShapeObject):
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
        
        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        Roxy2     = np.zeros(m.shape, dtype=np.float64)
        
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))

        Roxy2[m] = mul * (np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], self.Yaxis) ** 2)
        Ocompd   = 2 * mul * (np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis'])  + np.einsum('ij,j->i', Omod, self.Yaxis) * np.einsum('ij,j->i', dMod, self.Yaxis))

        a[m]  = mul * (np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', dMod[m], self.Yaxis) ** 2)
        b[m]  = Ocompd[m] - np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        c[m]  = Roxy2[m] - np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m1 = abs(a) > 0.0
        t1[m & m1], t2[m & m1] = multi_quadratic(a[m & m1], b[m & m1], c[m & m1])   

        t2[m & ~m1] = np.abs(c[m & ~m1] / b[m & ~m1])
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
        pt = -2 * self.param['multiplier'] * (np.einsum('i,j->ij', px, self.param['xaxis']) + np.einsum('i,j->ij', py, self.Yaxis)) + self.param['zaxis']
        
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[m] = pt
        
        return normals
