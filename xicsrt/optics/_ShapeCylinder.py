
# -*- coding: utf-8 -*-
"""
.. Authors:
   Conor Perks <cjperks@psfc.mit.edu>
   Novimir Pablant <npablant@pppl.gov>
   James Kring <jdk0026@tigermail.auburn.edu>
   Yevgeniy Yakusevich <eugenethree@gmail.com>
   Define the :class:`ShapeSpherical` class.
"""
import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._ShapeObject import ShapeObject

@dochelper
class ShapeCylinder(ShapeObject):
    """
    A cylindrical shape.
    This class defines intersections with a cylinder.
    """

    def default_config(self):
        """
        radius : float (1.0)
          The radius of the sphere.
        """
        config = super().default_config()
        config['radius'] = 1.0
        return config

    def initialize(self):
        super().initialize()
        # Finds center location of the cylinder object wrt origin on surface
        self.param['center'] = self.param['radius'] * self.param['zaxis'] + self.param['origin']

    def intersect(self, rays):
        # Loads ray information
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calulate the distance to the intersection of the rays with the
        cylindrical optic.
        This calculation is copied from:
        https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf
        """
        #setup variables
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        # Initializes arrays to store intersection variables
        distance = np.full(m.shape, np.nan, dtype=np.float64)
        q        = np.empty(m.shape, dtype=np.float64)
        int_cap  = np.empty(m.shape, dtype=np.float64)
        int_edge = np.empty(m.shape, dtype=np.float64)

        # Loads quantities defining cylinder, axis parameterized pa + t*va
        pa = self.param['center'] # point on axis cylinder is centered on (global coordinates)
        va = self.param['xaxis'] # cylinder axis orientation (wrt global coor.)
        za = self.param['zaxis'] # normal vector of crystal surface (wrt global coor.)
        r = self.param['radius'] # radius
        xsize = self.param['xsize'] # Length of the cylinder
        ysize = self.param['ysize'] # Height of the cylinder (assumes H<2*r)


        # Checks orthogonality of input axes
        if np.dot(va,za) != 0:
            print('Cylinder axes are not orthogonally defined')

        # If we can proceed with calculations
        else:
            # Calculates coordinates of cylinder caps
            #p1 = pa - va*(xsize/2)
            #p2 = pa + va*(xsize/2)

            # Calculates location of edges of cylindrically-bent object
            ha = np.cross(va, za) # Vector in the crystal height direction
            ha = ha/np.linalg.norm(ha) # Ensures direction is normalized
            #theta = np.arcsin(ysize/2/r) # Angle between cylinder center and object edge
            #h1 = pa + ha*(ysize/2) - za*r*np.cos(theta) # Calculates edge coordinates
            #h2 = pa - ha*(ysize/2) - za*r*np.cos(theta)

            # Calculates quadratic formula coefficients for t
            dp = O-pa # Length between ray origin and cylinder center
            dot_Dva = np.einsum('ij,ij->i', D, va)#np.dot(D, va) 
            dot_dpva = np.einsum('ij,ij->i', dp, va)#np.dot(dp, va)
            A1 = D - dot_Dva*va
            B1 = dp-dot_dpva*va
            A = np.einsum('ij,ij->i', A1, A1)#np.dot(A1,A1) # Coefficients for f(t) = A*t**2 + B*t + C = 0
            B = 2*np.einsum('ij,ij->i', A1, B1)#np.dot(A1,B1)
            C = np.einsum('ij,ij->i', B1, B1)-r**2#np.dot(B1,B1)-r**2

            # Solves the quadratic formula for t
            dis = B**2-4*A*C
            t1 = (-B - np.sqrt(dis))/(2*A)
            t2 = (-B + np.sqrt(dis))/(2*A)

            # Assumes that the object is bent such that there is only one real intersection
            # If the reflection surface is oriented towards the ray source (like it always should be)
            t = np.maximum(t1,t2).reshape(m.shape, 1)

            # Calculates ray intersection point if infinite cylinder
            q[m] = O[m] + D[m]*t[m]

            # Checks if the intersection point is within the end caps and object edge
            int_cap[m] = np.abs(np.einsum('ij,ij->i',va ,q[m]-pa))#np.abs(np.dot(va, q-pa)) # cap
            int_edge[m] = np.abs(np.einsum('ij,ij->i',ha ,q[m]-pa))#np.abs(np.dot(ha, q-pa)) # edge

            # If the intersection point is larger than the cystal, the ray misses the cylinder
            m[m] &= (int_cap[m] < xsize/2 and int_edge[m] < ysize/2)

            # Distance traveled by the ray before hitting the optic
            distance[m] = t[m]

            return distance, m

    # Calculates the normal vector from the interaction with the cylinder surface
    def intersect_normal(self, xloc, mask):
        # Initializes qunatities
        m = mask
        norm = np.full(xloc.shape, np.nan, dtype=np.float64)
        pa_proj = np.full(xloc.shape, np.nan, dtype=np.float64)
        pa = self.param['center']
        va = self.param['xaxis']

        # Note that a normal vector on a cylinder is orthogonal to the cylinder axis
        # As such, we need to project the 'center' of the cylinder onto the same
        # plane as the normal vector
        pa_proj[m] = pa - np.einsum('ij,ij->i',va ,pa-xloc[m])*va

        # Calculates the normal vector from intersection point to cylinder center
        norm[m] = xm.normalize(pa_proj[m] - xloc[m])
        return norm