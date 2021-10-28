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
from xicsrt.optics._multiQuartic import multi_cubic          # Multiple Quartic Solver with Order Reduction Method

@dochelper
class ShapeCuboidCylinder(ShapeObject):
    """
    A Parabolical shape.
    This class defines intersections with a Parabola
    """

    
    def default_config(self):
        config = super().default_config()
        
        """
        mul:
                mul of the parabola equation as x = a*z^2, a is given by this configuration
        Length:
                Length of the Ellipse Mirror
        concave:
                If True it will consider intersection of Ellipse Mirror concave surface with Rays only, otherwise 
                it will consider intersection of Ellipse Mirror convex surface with Rays only
        """
        
        config['multiplier']  = 1.0
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

        self.param['center'] = self.param['origin']

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

        mul = self.param['multiplier']
        Length = self.param['length']

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

        orig = O - self.param['center']

        sin  = np.sqrt(1 - np.einsum('ij,j->i', D, self.Yaxis) ** 2)
        dMod = D - np.einsum('i,j->ij', np.einsum('ij,j->i', D, self.Yaxis), self.Yaxis)    

        m[m] &= np.einsum('ij,ij->i', dMod[m], dMod[m]) > 0.0    
        Omod  = orig - np.einsum('i,j->ij', np.einsum('ij,j->i', orig, self.Yaxis), self.Yaxis)    
        dMod[m]  = np.einsum('ij,i->ij', dMod[m], 1/ np.sqrt(np.einsum('ij,ij->i',dMod[m],dMod[m])))

        a[m] = mul * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 3
        b[m] = 3 * mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) ** 2
        c[m] = 3 * mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 2 * np.einsum('ij,j->i', dMod[m], self.param['xaxis']) - np.einsum('ij,j->i', dMod[m], self.param['zaxis'])
        d[m] = mul * np.einsum('ij,j->i', Omod[m], self.param['xaxis']) ** 3 - np.einsum('ij,j->i', Omod[m], self.param['zaxis'])

        m[m] &= abs(a[m]) > 0.0

        t1[m], t2[m], t3[m] = multi_cubic(a[m], b[m], c[m], d[m])
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

        y0[m] = np.einsum('ij,j->i', orig[m], self.Yaxis)
        m1[m] = np.einsum('ij,j->i', D[m], self.Yaxis)

        y1[m] = y0[m] + m1[m] * R1[m,0]
        y2[m] = y0[m] + m1[m] * R1[m,1]
        y3[m] = y0[m] + m1[m] * R1[m,2] 

        R1[m,0][np.abs(y1) > Length / 2] = -sys.float_info.max
        R1[m,1][np.abs(y2) > Length / 2] = -sys.float_info.max
        R1[m,2][np.abs(y3) > Length / 2] = -sys.float_info.max    

        R1 = np.sort(R1,axis=1)
        distances[a < 0] = R1[a < 0, 1]
        distances[a > 0] = R1[a > 0, 2]
        m &= distances > 0.0

        return distances, m
    
    # Generates normals
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        pt = xloc[m] - self.param['center']
        pt -= np.einsum('i,j->ij', np.einsum('ij,j->i', pt, self.Yaxis), self.Yaxis)
        
        pt = np.einsum('i,j->ij', np.ones(len(pt)), self.param['zaxis']) - 3 * self.param['multiplier'] * np.einsum('i,j->ij', np.einsum('ij,j->i', pt, self.param['xaxis']), self.param['xaxis']) ** 2
        
        pt = np.einsum('ij,i->ij', pt, 1 / np.sqrt(np.einsum('ij,ij->i', pt, pt)))
        
        normals[m] = pt
        
        return normals
