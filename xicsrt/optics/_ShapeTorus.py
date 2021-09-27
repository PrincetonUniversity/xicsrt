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
from xicsrt.optics._multiQuartic import multi_quartic          # Multiple Quartic Solver with Order Reduction Method

@dochelper
class ShapeTorus(ShapeObject):
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
        
        config['Rmajor']  = 1.1
        config['Rminor']  = 0.2
        config['index'] = 3
        
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
        
        self.torusZaxis = self.param['zaxis']
        self.torusXaxis = self.param['xaxis']
        
        self.torusYaxis = np.cross(self.param['zaxis'], self.param['xaxis'])

        self.param['center'] = (self.param['Rmajor'] + self.param['Rminor']) * self.torusZaxis + self.param['origin']
        #self.param['center'] = self.param['origin']

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

        Rmajor = self.param['Rmajor']
        Rminor = self.param['Rminor']  
        
        # variable setup
        distances = np.zeros(m.shape, dtype=np.float64)
        
        orig = O - self.param['center']      

        # Calculaing Ray Direction components in Torus coordinate system (Transforming Ray)
        d = np.zeros((len(m),3), dtype= np.float64)
        d[:, 0] = np.dot(D, self.torusXaxis)
        d[:, 1] = np.dot(D, self.torusYaxis)
        d[:, 2] = np.dot(D, self.torusZaxis)

        # Calculaing Ray Origin components in Torus coordinate system (Transforming Ray Origin)
        dOrig = np.zeros((len(m),3), dtype= np.float64)
        dOrig[:, 0] = np.dot(orig, self.torusXaxis)
        dOrig[:, 1] = np.dot(orig, self.torusYaxis)
        dOrig[:, 2] = np.dot(orig, self.torusZaxis)
        
        # Calculaing Magnitude of Ray Direction
        dMag = np.sqrt(np.einsum('ij,ij->i', d, d))
        
        # defining resusable variables
        distRayOrigin2OriginSq = np.einsum('ij,ij->i', dOrig, dOrig)
        rayCompDirOnOrigin = np.einsum('ij,ij->i', dOrig, d)

        R1 = Rmajor ** 2 + Rminor ** 2 

        
        """
            The form of quartic equation to be solved is,
            c0 ** t ^ 4 + c1 ** t ^ 3 + c2 ** t ^ 2 + c3 ** t + c4 = 0
        """
        
        # defining co-efficients
        # Coefficient of t^4
        c0 = dMag ** 4                           # ok
        # Coefficient of t^3
        c1 = 4 * dMag ** 2 * rayCompDirOnOrigin  # ok
        # Coefficient of t^2
        c2 = 4 * rayCompDirOnOrigin ** 2 + 2 * distRayOrigin2OriginSq * dMag ** 2 - 2 * R1 * dMag ** 2 + 4 * Rmajor ** 2 * d[:, 1] ** 2
        # Coefficient of t^1
        c3 = 4 * rayCompDirOnOrigin * (distRayOrigin2OriginSq - R1) + 8 * Rmajor ** 2 * d[:, 1] * dOrig[:, 1]
        # Coefficient of t^0
        c4 = distRayOrigin2OriginSq ** 2 - 2 * R1 * distRayOrigin2OriginSq + 4 * Rmajor ** 2 * dOrig[:, 1] ** 2 + (Rmajor ** 2 - Rminor ** 2) ** 2
    
        roots_0, roots_1, roots_2, roots_3 = multi_quartic(c0, c1, c2, c3, c4)
        
        # neglecting complex & negative solution of the quartic equation    
        roots_0[roots_0.imag != 0] = -float_info.max
        roots_1[roots_1.imag != 0] = -float_info.max
        roots_2[roots_2.imag != 0] = -float_info.max
        roots_3[roots_3.imag != 0] = -float_info.max

        r1 = np.zeros((len(roots_0), 4), dtype=float)
        r1[:,0] = roots_0.real
        r1[:,1] = roots_1.real
        r1[:,2] = roots_2.real
        r1[:,3] = roots_3.real
        r1 = np.sort(r1,axis=1)

        distances = r1[:, self.param['index']]
        m[m] &= distances[m] > 0.0

        return distances, m
    
    # Generates normals
    def intersect_normal(self, xloc, mask):
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)
        
        """
            Here, we first translates torus to the origin, then translates the circle on which 
            the point lies to the origin and then radius vector at that point will give normal
            at that point
        """
        
        # Simulates the Translation of Torus to the Origin
        pt = np.subtract(xloc[m], self.param['center'])
        
        """
        Checks that if Ray is intersecting Torus from back-side
        
        If ray is from back-side, ray-direction will give (+ve) component along the vector 
        from the Center of torus to the given point, otherwise, it should give (-ve) component.
        
        And, If ray is intersecting from back-side then the normal at intersection point should be flipped
        
            fromback = np.einsum('ij,j->i', pt, rays['direction']) > 0
            
        NOTE:
            In order to match with the sphere case, this calculation is not performed
        """
    
        """
            Calculates the Center of the Circle on which the point lies,
            by subtracting which the circle goes to the origin 
        """
        pt1 = np.subtract(pt, np.einsum('i,j->ij',np.einsum('ij,j->i', pt, self.torusYaxis), self.torusYaxis))
        pt1 = self.param['Rmajor'] * np.einsum('ij,i->ij', pt1 , 1 / np.sqrt(np.einsum('ij,ij->i', pt1, pt1)))
        
        """
        Checks that if Ray will reflect from inside or outside of the Torus tube
        
        If ray is from inside, ray-direction will give (+ve) component along the vector 
        from the Center of circle to the given point, otherwise, it should give (-ve) component.
        
        And, If ray is intersecting from inside then the normal at intersection point should be flipped
        
            inside = np.einsum('ij,j->i', pt, pt - pt1) < 0
        
        NOTE:
            In order to match with the sphere case, this calculation is not performed
        """        
        
        # Simulates the circle having the intersection point at the origin And getting radius vector
        pt1 = np.subtract(pt, pt1)
        pt1 = np.einsum('ij,i->ij', pt1, 1 / np.sqrt(np.einsum('ij,ij->i', pt1, pt1)))
        
        """            
        pt1[fromback] = -pt1[fromback]      # flipping normal if ray is hitting from back-side
        pt1[inside] = -pt1[inside]          # flipping normal if point is on the inside ring of torus
                
        NOTE:
            In order to match with the spherical case, this calculation is not performed
        """
        normals[m] = pt1
        
        return normals
