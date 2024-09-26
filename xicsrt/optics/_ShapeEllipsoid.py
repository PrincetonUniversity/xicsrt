# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>


   Define the :class:`ShapeEllipsoid` class.
"""
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject

@dochelper
class ShapeEllipsoid(ShapeObject):
    """
    A Ellipsoid shape.
    """

    def default_config(self):        
        """
        radius_major:
                Major Radius of the Ellipsoid
        radius_minor:
                Minor Radius of the Ellipsoid
        convex:
            If True the surface will be oriented to be convex, otherwise
            the surface will be oriented to be concave.
        """
        config = super().default_config()
        config['radius_major']  = 1.1
        config['radius_minor']  = 0.2
        config['convex'] = False
        
        return config

    def initialize(self):
        """
        Here, we considered Ellipse Mirror to be built around y-axis, so we construct ellipse Mirror axis from
        system axes given such as ellipse Mirror z-axis goes in yaxis of the system.
        And system z-axis & x-axis goes to ellipse Mirror x-axis and y-axis respectively

        Ellipse Mirror center is defined as,
            Ellipse_Mirror_Center = Ellipse_Mirror_Major_Radius * Ellipse_Mirror_Z_axis + System_Origin
        """
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
        dMod = rays['direction']
        m = rays['mask']

        radius_major = self.param['radius_major']
        radius_minor = self.param['radius_minor']
        
        # variable setup
        distances = np.zeros(m.shape, dtype=np.float64)

        t1        = np.zeros(m.shape, dtype=np.float64)
        t2        = np.zeros(m.shape, dtype=np.float64)
        a         = np.zeros(m.shape, dtype=np.float64)
        b         = np.zeros(m.shape, dtype=np.float64)
        c         = np.zeros(m.shape, dtype=np.float64)
        Roxy2     = np.zeros(m.shape, dtype=np.float64)

        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        Omod = O - self.param['center']

        m[m] &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0

        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))
        
        Roxy2[m] = (np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', Omod[m], yaxis) ** 2) * radius_major ** 2 + np.einsum('ij,j->i', Omod[m], self.param['zaxis']) ** 2 * radius_minor ** 2
        Ocompd   = (np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis'])  + np.einsum('ij,j->i', Omod, yaxis) * np.einsum('ij,j->i', dMod, yaxis) )* radius_major ** 2 + np.einsum('ij,j->i', Omod, self.param['zaxis']) * np.einsum('ij,j->i', dMod, self.param['zaxis']) * radius_minor ** 2

        a[m] = radius_major ** 2 * (np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2 + np.einsum('ij,j->i', dMod[m], yaxis) ** 2 )
        a[m]+= radius_minor ** 2 * np.einsum('ij,j->i', dMod[m], self.param['zaxis']) ** 2
        b[m] = 2 * Ocompd[m]
        c[m] = Roxy2[m] - radius_minor ** 2 * radius_major ** 2

        t1[m], t2[m] = xq.multi_quadratic(a[m], b[m], c[m])

        distances[m] = np.where((t1[m] > t2[m]) if not self.param['convex'] else (t1[m] < t2[m]), t1[m], t2[m])
        m[m] &= distances[m] >= 0.0

        return distances, m
    
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        pt = xloc[m] - self.param['center']
        z = np.einsum('ij,j->i',pt, self.param['zaxis'])

        pt -= np.einsum('i,j->ij', z, self.param['zaxis'])
        pt += self.param['radius_minor'] ** 2 / self.param['radius_major'] ** 2 * np.einsum('i,j->ij', z, self.param['zaxis'])
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[m] = pt
        
        return normals
