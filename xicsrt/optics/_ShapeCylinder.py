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

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._ShapeObject import ShapeObject
from xicsrt.optics._multiQuartic import multi_quadratic          # Multiple Quartic Solver with Order Reduction Method

@dochelper
class ShapeCylinder(ShapeObject):
    """
    A Cylindrical shape.
    This class defines intersections with a ellipse
    """

    
    def default_config(self):
        config = super().default_config()
        
        """
        Rmajor:
                Major Radius of the Cylinder
        Rminor:
                Minor Radius of the Cylinder
        concave:
                If True it will consider intersection of Cylinder concave surface with Rays only, otherwise 
                it will consider intersection of Cylinder convex surface with Rays only
        inside:
                If True it will consider intersection of Cylinder which will give reflection from inside of
                the Cylinder Tube otherwise it will consider intersection of Cylinder which will give reflection
                from outside of the Cylinder Tube
        """
        
        config['radius']  = 0.2
        config['concave'] = True
        
        return config

    def initialize(self):
        super().initialize()
        
        """
            Here, we considered Cylinder to be buit around y-axis, so we construct cylinder axis from
            system axes given such as cylinder z-axis goes in yaxis of the system.
            And system z-axis & x-axis goes to cylinder x-axis and y-axis respectively
            
            Cylinder center is defined as,
                Cylinder_Center = Cylinder_Major_Radius * Cylinder_Z_axis + System_Origin
        """
        
        self.Yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        
        # Making Y-axis and Z-axis orthogonal by altering Y-axis
        dot = np.dot(self.Yaxis, self.param['zaxis'])
        if dot > 0.0:
            self.Yaxis -= dot * self.param['zaxis']
        self.Yaxis /= np.linalg.norm(self.Yaxis)

        dist = self.param['radius']

        if not self.param['concave']: dist = -dist

        self.param['center'] = self.param['origin'] + self.param['zaxis'] * dist

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
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        Radius = self.param['radius']
        
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
#         sign = np.einsum('ij,ij->i', D, -orig)

#         m[m] &= sign[m] > 0.0

        sin  = np.sqrt(1 - np.einsum('ij,j->i', D, self.Yaxis) ** 2)
        
        dMod = D - np.einsum('i,j->ij', np.einsum('ij,j->i', D, self.Yaxis), self.Yaxis)
        
        
        m[m] &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0
        
        Omod  = orig - np.einsum('i,j->ij', np.einsum('ij,j->i', orig, self.Yaxis), self.Yaxis)
        
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))
        Roxy2[m] = np.einsum('ij,ij->i', Omod[m], Omod[m])
        Ocompd   = np.einsum('ij,j->i', Omod, self.param['zaxis']) * np.einsum('ij,j->i', dMod, self.param['zaxis']) + np.einsum('ij,j->i', Omod, self.param['xaxis']) * np.einsum('ij,j->i', dMod, self.param['xaxis'])
        
        # TODO : this code will be replaced with geometrical method
        a[m] = np.einsum('ij,j->i', dMod[m], self.param['zaxis']) ** 2 
        a[m]+= np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2
        b[m] = 2 * Ocompd[m]
        c[m] = Roxy2[m] - Radius ** 2

        t1[m], t2[m] = multi_quadratic(a[m], b[m], c[m])
#         di2 = Roxy2[m]
#         t02 = Ocompd[m]
#         t0 = np.sqrt(t02)
#         h  = np.sqrt(Radius ** 2 + t02 - di2)
#         t1[m] = t0 - h
#         t2[m] = t0 + h
        #print(t1,t2)
        t1[m] /= sin[m]
        t2[m] /= sin[m]
        
        distances[m] = np.where((t1[m] > t2[m]) if self.param['concave'] else (t1[m] < t2[m]), t1[m], t2[m])    
        m[m] &= distances[m] > 0.0

        return distances, m
    
    # Generates normals
    def intersect_normal(self, xloc, mask):
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        pt = xloc[mask] - self.param['center']
        pt -= np.einsum('i,j->ij', np.einsum('ij,j->i', pt, self.Yaxis), self.Yaxis)
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[mask] = pt
        
        return normals
