# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import numpy as np

from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

class XicsrtOpticMosaicGraphite(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['mosaic_spread'] = 0.0
        return config

    def intersect(self, rays):
        """
        Calulate the distance to the intersection of the rays with an 
        infinite plane.
        """
        
        #test to see if a ray intersects the mirror plane
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m]  = np.dot((self.param['origin'] - O[m]), self.param['zaxis'])
        distance[m] /= np.dot(D[m], self.param['zaxis'])

        # Update the mask to count only intersections with the plain.
        m[m] &= (distance[m] > 0)
        
        return distance
    """#Uniformly distributed crystallite normals
    def f(self, theta, number):
        output = np.empty((number, 3))
        
        z   = np.random.uniform(np.cos(theta/2),1, number)
        phi = np.random.uniform(0, 2 * np.pi, number)
        
        output[:,0]   = np.sqrt(1-z**2) * np.cos(phi)
        output[:,1]   = np.sqrt(1-z**2) * np.sin(phi)
        output[:,2]   = z
        return output
    """
    
    #Gaussian distributed crystallite normals
    def f(self, FWHM, number):
        output = np.empty((number, 3))
        
        #convert the angular FWHM into a linear-displacement-from-vertical FWHM
        disp = 1 - np.cos(FWHM / 2)
        
        #convert from linear-displacement FWHM to standard deviation
        sigma = disp / (2 * np.sqrt(2 * np.log(2)))
        
        #create the half-normal distribution of off-vertical vectors
        z = 1
        if sigma > 0:
            z -= np.abs(np.random.normal(0, sigma, number))
        phi = np.random.uniform(0, 2 * np.pi, number)
        
        output[:,0]   = np.sqrt(1-z**2) * np.cos(phi)
        output[:,1]   = np.sqrt(1-z**2) * np.sin(phi)
        output[:,2]   = z
        
        return output
    
    
    def generate_optic_normals(self, X, rays):
        # Pulled from Novi's FocusedExtendedSource
        # Generates a list of crystallite norms normally distributed around the
        # average graphite mirror norm       
        O = rays['origin']
        m = rays['mask']
        
        normals = np.zeros(O.shape)
        rad_spread = np.radians(self.param['mosaic_spread'])
        dir_local = self.f(rad_spread, len(m))

        # Create two vectors perpendicular to the surface normal,
        # it doesn't matter how they are oriented otherwise.
        norm_surf = np.ones(O.shape) * self.param['zaxis']
        o_1     = np.zeros(O.shape)
        o_1[m]  = np.cross(norm_surf[m], [0,0,1])
        o_1[m] /= np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
        o_2     = np.zeros(O.shape)
        o_2[m]  = np.cross(norm_surf[m], o_1[m])
        o_2[m] /= np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
        
        R = np.empty((len(m), 3, 3))
        # We could mask this with m, but I don't know if that will
        # improve speed or actually make it worse.
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = norm_surf

        normals[m] = np.einsum('ij,ijk->ik', dir_local[m], R[m])
        return normals
    
    def mesh_generate_optic_normals(self, X, rays, hits):
        # Pulled from Novi's FocusedExtendedSource
        # Generates a list of crystallite norms normally distributed around the
        # average graphite mirror norm
        O = rays['origin']
        m = rays['mask']
        
        normals = np.zeros(O.shape)
        rad_spread = np.radians(self.param['mosaic_spread'])
        dir_local = self.f(rad_spread, len(m))

        for ii in range(len(self.param['mesh_faces'])):
            tri   = self.mesh_triangulate(ii)
            test  = np.equal(ii, (hits - 1))
            test &= m
            
            norm_surf  = np.ones(O.shape) * tri['normal']
            o_1        = np.zeros(O.shape)
            o_1[test]  = np.cross(norm_surf[test], [0,0,1])
            o_1[test] /= np.linalg.norm(o_1[test], axis=1)[:, np.newaxis]
            o_2        = np.zeros(O.shape)
            o_2[test]  = np.cross(norm_surf[test], o_1[test])
            o_2[test] /= np.linalg.norm(o_2[test], axis=1)[:, np.newaxis]
            
            R = np.empty((len(m), 3, 3))
            # We could mask this with test, but I don't know if that will
            # improve speed or actually make it worse.
            R[:,0,:] = o_1
            R[:,1,:] = o_2
            R[:,2,:] = norm_surf
            
            normals[test] = np.einsum('ij,ijk->ik', dir_local[test], R[test])
            
        return normals
