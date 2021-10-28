# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>
   James Kring <jdk0026@tigermail.auburn.edu>
   Yevgeniy Yakusevich <eugenethree@gmail.com>

   Define the :class:`ShapeElliptical` class.
"""
import sys
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._ShapeObject import ShapeObject
from xicsrt.optics._multiQuartic import multi_quadratic         # Multiple Quartic Solver with Order Reduction Method

@dochelper
class ShapeEllipsoidCylinder(ShapeObject):
    """
    A Elliptical shape.
    This class defines intersections with a ellipse
    """

    
    def default_config(self):
        config = super().default_config()
        
        """
        Rmajor:
                Major Radius of the Ellipse Mirror
        Rminor:
                Minor Radius of the Ellipse Mirror
        Length:
                Length of the Ellipse Mirror
        concave:
                If True it will consider intersection of Ellipse Mirror concave surface with Rays only, otherwise 
                it will consider intersection of Ellipse Mirror convex surface with Rays only
        """
        
        config['Rmajor']  = 0.5
        config['Rminor']  = 0.2
        config['length'] = 1.0
        config['concave'] = True
        
        return config

    def initialize(self):
        super().initialize()
        
        """
            Here, we considered Ellipse Mirror to be buit around y-axis, so we construct ellipse Mirror axis from
            system axes given such as ellipse Mirror z-axis goes in yaxis of the system.
            And system z-axis & x-axis goes to ellipse Mirror x-axis and y-axis respectively
            
            Ellipse Mirror center is defined as,
                Ellipse_Mirror_Center = Ellipse_Mirror_Major_Radius * Ellipse_Mirror_Z_axis + System_Origin
        """ 
        self.Yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])

        self.param['center'] = self.param['Rmajor'] * self.param['zaxis'] + self.param['origin']

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
        
        # setup
        # setup
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        Rmajor = self.param['Rmajor']
        Rminor = self.param['Rminor']
        Length = self.param['length']
        
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
        
        orig = O - self.param['center']
        sign = np.einsum('ij,ij->i', D, -orig)

        m[m] &= sign[m] > 0.0

        sin  = np.sqrt(1 - np.einsum('ij,j->i', D, self.Yaxis) ** 2)
        
        dMod = D - np.einsum('i,j->ij', np.einsum('ij,j->i', D, self.Yaxis), self.Yaxis)
        
        
        m[m] &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0
        
        Omod  = orig - np.einsum('i,j->ij', np.einsum('ij,j->i', orig, self.Yaxis), self.Yaxis)
        
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))
        Roxy2[m] = np.einsum('ij,j->i', Omod[m], self.param['zaxis']) ** 2 * Rminor ** 2 + np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 * Rmajor ** 2
        Ocompd   = np.einsum('ij,j->i', Omod, self.param['zaxis']) * np.einsum('ij,j->i', dMod, self.param['zaxis']) * Rminor ** 2 + np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis']) * Rmajor ** 2

        a[m] = Rminor ** 2 * np.einsum('ij,j->i', dMod[m], self.param['zaxis']) ** 2 
        a[m]+= Rmajor ** 2 * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2
        b[m] = 2 * Ocompd[m]
        c[m] = Roxy2[m] - Rminor ** 2 * Rmajor ** 2

        t1[m], t2[m] = multi_quadratic(a[m], b[m], c[m])
        #print(t1,t2)
        t1[m] /= sin[m]
        t2[m] /= sin[m]

        y0[m] = np.einsum('ij,j->i', orig[m], self.Yaxis)
        m1[m] = np.einsum('ij,j->i', D[m], self.Yaxis)

        y1[m] = y0[m] + m1[m] * t1[m]
        y2[m] = y0[m] + m1[m] * t2[m]
    
        t1[np.abs(y1) > Length / 2] = -sys.float_info.max
        t2[np.abs(y2) > Length / 2] = sys.float_info.max
        
        distances[m] = np.where((t1[m] > t2[m]) if self.param['concave'] else (t1[m] < t2[m]), t1[m], t2[m])    
        m[m] &= distances[m] > 0.0
        m[m] &= distances[m] < sys.float_info.max

        return distances, m
    
    # Generates normals
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        pt = np.subtract(xloc[m], self.param['center'])
        pt -= np.einsum('i,j->ij',np.einsum('ij,j->i',pt, self.Yaxis), self.Yaxis)
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        normals[m] = pt
        
        normals[mask] = pt
        
        return normals
