
# -*- coding: utf-8 -*-
"""
.. Authors:
   Conor Perks <cjperks@psfc.mit.edu>
   Novimir Pablant <npablant@pppl.gov>

Define the :class:`ShapeCylinder` class.
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
        # Finds center location on the surface of the cylinder
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
        #d        = np.empty(m.shape, dtype=np.float64)
        #t_hc     = np.empty(m.shape, dtype=np.float64)
        #t_0      = np.empty(m.shape, dtype=np.float64)
        #t_1      = np.empty(m.shape, dtype=np.float64)

        # Loads quantities defining cylinder, axis parameterized pa + t*va
        pa = self.param['center'] # point on axis cylinder is centered on (global coordinates)
        va = self.param['cylaxis'] # cylinder axis orientation
        za = self.param['zaxis'] # normal vector of crystal surface
        r = self.param['radius'] # radius
        xsize = self.param['xsize'] # Length of the cylinder
        ysize = self.param['ysize'] # Height of the cylinder (assumes H<2*r)


        # Checks orthogonality of input axes
        if np.dot(va,za) != 0:
            print('Cylinder axes are not orthogonally defined')

        # If we can proceed with calculations
        else:
            # Calculates coordinates of cylinder caps
            p1 = pa - va*(xsize/2)
            p2 = pa + va*(xsize/2)

            # Calculates location of edges of cylindrically-bent object
            ha = np.cross(va, za) # Vector in the crystal height direction
            ha = ha/np.linalg.norm(ha) # Ensures direction is normalized
            theta = np.arcsin(ysize/2/r) # Angle between cylinder center and object edge
            h1 = pa + ha*(ysize/2) - za*r*np.cos(theta) # Calculates edge coordinates
            h2 = pa - ha*(ysize/2) - za*r*np.cos(theta)

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

            #if np.dot(za, D) < 0:
            #    t = np.max([t1,t2]) # Saves the longest intersection distance
            # If the reflection surface is oriented away from the ray source
            #elif np.dot(za, D) > 0:
            #    t = np.min([t1,t2]) # Saves the shortest intersection distance

            # Error check
            #if t<0:
            #    print('Error in intersection time')
            #else:

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
                # If we have a valid intersection with our cylindrically-bent finite object
                #if int_cap < xsize/2 and int_edge < ysize/2:
                #    success = True

                    # Outputs distance traveled by the ray before hitting the optic
                #    distance = t

                #else:
                #    success = False
                #    distance = np.nan

        ############# Original spherical crystal code #############################
        # L is the destance from the ray origin to the center of the sphere.
        # t_ca is the projection of this distance along the ray direction.
        #L     = self.param['center'] - O
        #t_ca  = np.einsum('ij,ij->i', L, D)
        
        # If t_ca is less than zero, then there is no intersection in the
        # the direction of the ray (there might be an intersection behind.)
        # Use mask to only perform calculations on rays that hit the crystal
        # m[m] &= (t_ca[m] >= 0)
        
        # d is the minimum distance between a ray and center of curvature.
        #d[m] = np.sqrt(np.einsum('ij,ij->i',L[m] ,L[m]) - t_ca[m]**2)

        # If d is larger than the radius, the ray misses the sphere.
        #m[m] &= (d[m] <= self.param['radius'])
        
        # t_hc is the distance from d to the intersection points.
        #t_hc[m] = np.sqrt(self.param['radius']**2 - d[m]**2)
        
        #t_0[m] = t_ca[m] - t_hc[m]
        #t_1[m] = t_ca[m] + t_hc[m]

        # Distance traveled by the ray before hitting the optic
        #distance[m] = np.where(t_0[m] > t_1[m], t_0[m], t_1[m])

        #return distance, m

    def intersect_normal(self, xloc, mask):
        m = mask
        norm = np.full(xloc.shape, np.nan, dtype=np.float64)
        norm[m] = xm.normalize(self.param['center'] - xloc[m])
        return norm