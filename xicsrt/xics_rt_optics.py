# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:05:57 2017
Edited on Fri Sep 06 11:06:00 2019

@author: James
@editor: Eugene

Description
-----------
The spherical quartz crystal and Highly Oriented Pyrolytic Graphite film 
that reflect X-rays that satisfy the Bragg condition. Optical elements have a
position and rotation in 3D space, optical properties such as crystal spacing,
rocking curve, and reflectivity, as well as a height and width.
"""
from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
         
class GenericOptic:
    def __init__(self, optic_input, general_input):
        self.position       = optic_input['position']
        self.normal         = optic_input['normal']
        self.xorientation   = optic_input['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.spacing        = optic_input['spacing']
        self.rocking_curve  = optic_input['rocking_curve']
        self.reflectivity   = optic_input['reflectivity']
        self.width          = optic_input['width']
        self.height         = optic_input['height']
        self.pixel_size     = self.width / optic_input['pixel_scaling']
        self.pixel_width    = int(round(self.width / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))     
        np.random.seed(general_input['random_seed'])
        
        def pixel_center(row, column):
            row_center = self.pixel_height / 2 - .5
            column_center = self.pixel_width / 2 - .5
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.position + xstep * self.xorientation 
                                    + ystep * self.yorientation)
            
            return center
        
        def create_center_array():
            center_array = []
            for ii in range(0, self.pixel_height):
                for jj in range(0, self.pixel_width):
                    point = pixel_center(ii, jj)
                    center_array.append(point)
            
            return center_array
        
        self.pixel_array = np.zeros((self.pixel_height, self.pixel_width))
        self.center_tree = cKDTree(create_center_array())
        
    def normalize(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        vector_norm = vector / magnitude[:, np.newaxis]
        return vector_norm
        
    def norm(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        return magnitude 

    def rocking_curve_filter(self, incident_angle, bragg_angle):
        # Convert from FWHM to sigma.
        sigma = self.rocking_curve/np.sqrt(2 * np.log(2)) / 2
        
        # Normalized Gaussian.
        p = np.exp(-np.power(incident_angle - bragg_angle, 2.) / (2 * sigma**2))
        
        test = np.random.uniform(0.0, 1.0, len(incident_angle))
        mask = p.flatten() > test
        return mask
    
    def intersect_check(self, rays, distance):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X = np.zeros(O.shape, dtype=np.float64)
        xproj = np.zeros(m.shape, dtype=np.float64)
        yproj = np.zeros(m.shape, dtype=np.float64)
        
        #X is the 3D point where the ray intersects the optic
        X[m] = O[m] + D[m] * distance[m,np.newaxis]
        
        #find which rays hit the optic, update mask to remove misses   
        xproj[m] = abs(np.dot(X[m] - self.position, self.xorientation))
        yproj[m] = abs(np.dot(X[m] - self.position, self.yorientation))
        m[m] &= ((xproj[m] <= self.width / 2) & (yproj[m] <= self.height / 2))
        return X, rays
    
    def angle_check(self, X, rays, norm):
        D = rays['direction']
        W = rays['wavelength']
#       w = rays['weight']
        m = rays['mask']
        
        bragg_angle = np.zeros(m.shape, dtype=np.float64)
        dot = np.zeros(m.shape, dtype=np.float64)
        incident_angle = np.zeros(m.shape, dtype=np.float64)
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the optic
        bragg_angle[m] = np.arcsin( W[m] / (2 * self.crystal_spacing))
        dot[m] = np.einsum('ij,ij->i',D[m], -1 * norm[m])
        incident_angle[m] = (np.pi / 2) - np.arccos(dot[m] / self.norm(D[m]))
        #check which rays satisfy bragg, update mask to remove those that don't
        m[m] &= self.rocking_curve_filter(bragg_angle[m], incident_angle[m])
        return rays, norm
    
    def reflect_vectors(self, X, rays, norm):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        # Check which vectors meet the Bragg condition (with rocking curve)
        rays, norm = self.angle_check(X, rays, norm)
        
        # Perform reflection around normal vector, creating new rays with new
        # origin O = X and new direction D
        O[m]  = X[m]
        D[m] -= 2 * np.einsum('ij,ij->i', D[m], norm[m])[:, np.newaxis] * norm[m]
        
        return rays
    
    def pixel_row_column(self, pixel_number):
        row = int(pixel_number // self.pixel_width)
        column = pixel_number - (row * self.pixel_width)
        return row, column

    def collect_rays(self, rays):
        X = rays['origin']
        m = rays['mask']
        index = self.center_tree.query(X[m])[1]
        self.photon_count = len(m[m])
        
        for number in index:
            row, column = self.pixel_row_column(number)
            self.pixel_array[row, column] += 1
        return
    
    def output_image(self, image_name, rotate=None):
        if rotate:
            out_array = np.rot90(self.pixel_array)
        else:
            out_array = self.pixel_array
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)

      
class SphericalCrystal(GenericOptic):
    def __init__(self, crystal_input, general_input):
        super().__init__(crystal_input, general_input)
        
        self.__name__       = 'SphericalCrystal'
        self.radius         = crystal_input['curvature']
        self.position       = crystal_input['position']
        self.normal         = crystal_input['normal']
        self.xorientation   = crystal_input['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.crystal_spacing= crystal_input['spacing']
        self.rocking_curve  = crystal_input['rocking_curve']
        self.reflectivity   = crystal_input['reflectivity']
        self.width          = crystal_input['width']
        self.height         = crystal_input['height']
        self.center         = self.radius * self.normal + self.position
        self.pixel_size     = self.width / crystal_input['pixel_scaling']
        self.pixel_width    = int(round(self.width / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))     
        np.random.seed(general_input['random_seed'])
        
        def pixel_center(row, column):
            row_center = self.pixel_height / 2 - .5
            column_center = self.pixel_width / 2 - .5
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.position + xstep * self.xorientation 
                                    + ystep * self.yorientation)
            
            return center
            
        def create_center_array():
            center_array = []
            for ii in range(0, self.pixel_height):
                for jj in range(0, self.pixel_width):
                    point = pixel_center(ii, jj)
                    center_array.append(point)
            
            return center_array
            
        self.pixel_array = np.zeros((self.pixel_height, self.pixel_width))
        self.center_tree = cKDTree(create_center_array())      
        
    def spherical_intersect(self, rays):
        """
        This calculation is copied from:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/
                minimal-ray-tracer-rendering-simple-shapes/
                ray-sphere-intersection
        """
        #setup variables
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        d        = np.zeros(m.shape, dtype=np.float64)
        t_hc     = np.zeros(m.shape, dtype=np.float64)
        t_0      = np.zeros(m.shape, dtype=np.float64)
        t_1      = np.zeros(m.shape, dtype=np.float64)
        
        #distance traveled by the ray before hitting the optic
        #this calculation is performed for all rays, mask regardless
        L     = self.center - O
        t_ca  = np.einsum('ij,ij->i', L, D)
        mag_L = np.linalg.norm(L, axis=1)
        
        # If t_ca is less than zero, then there is no intersection
        # Use mask to only perform calculations on rays that hit the crystal
#        m[m] &= ((t_ca[m] >= 0) & (mag_L[m] >= self.radius))
        
        #d is the impact parameter between a ray and center of curvature
        d[m]    = np.sqrt(np.einsum('ij,ij->i',L[m] ,L[m]) - t_ca[m]**2)
        t_hc[m] = np.sqrt(self.radius**2 - d[m]**2)
        
        t_0[m] = t_ca[m] - t_hc[m]
        t_1[m] = t_ca[m] + t_hc[m]
        
        distance[m] = np.where(t_0[m] > t_1[m], t_0[m], t_1[m])
        return distance
    
    def spherical_norm_generate(self, X, rays):
        m = rays['mask']
        norm = np.zeros(X.shape, dtype=np.float64)
        norm[m] = self.normalize(self.center - X[m])
        
        return norm

    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.spherical_intersect(rays))
        print(' Rays on Crystal:   {:6.4e}'.format(D[m].shape[0]))        
        rays = self.reflect_vectors(X, rays, self.spherical_norm_generate(X, rays))
        print(' Rays from Crystal: {:6.4e}'.format(D[m].shape[0]))     
        return rays      

class MosaicGraphite(GenericOptic):
    def __init__(self, graphite_input, general_input):
        super().__init__(graphite_input, general_input)
        
        self.__name__       = 'MosaicGraphite'
        self.position       = graphite_input['position']
        self.normal         = graphite_input['normal']
        self.xorientation   = graphite_input['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.crystal_spacing= graphite_input['spacing']
        self.rocking_curve  = graphite_input['rocking_curve']
        self.reflectivity   = graphite_input['reflectivity']
        self.mosaic_spread  = graphite_input['mosaic_spread']
        self.width          = graphite_input['width']
        self.height         = graphite_input['height']
        self.pixel_size     = self.width / graphite_input['pixel_scaling'] 
        self.pixel_width    = int(round(self.width / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))
        self.center         = self.position
        np.random.seed(general_input['random_seed'])
        
        def pixel_center(row, column):
            row_center = self.pixel_height / 2 - .5
            column_center = self.pixel_width / 2 - .5
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.position + xstep * self.xorientation 
                                    + ystep * self.yorientation)
            
            return center
            
        def create_center_array():
            center_array = []
            for ii in range(0, self.pixel_height):
                for jj in range(0, self.pixel_width):
                    point = pixel_center(ii, jj)
                    center_array.append(point)
                
            return center_array
            
        self.pixel_array = np.zeros((self.pixel_height, self.pixel_width))
        self.center_tree = cKDTree(create_center_array())

    def planar_intersect(self, rays):
        #test to see if a ray intersects the mirror plane
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m] = np.dot((self.position - O[m]), self.normal)/np.dot(D[m], self.normal)
        
        test = (distance > 0) & (distance < 10)
        distance = np.where(test, distance, 0)
        return distance
    
    def mosaic_norm_generate(self, rays):
        # Pulled from Novi's FocusedExtendedSource
        # Generates a list of crystallite norms normally distributed around the
        # average graphite mirror norm
        def f(theta, number):
            output = np.empty((number, 3))
            
            z   = np.random.uniform(np.cos(theta),1, number)
            phi = np.random.uniform(0, np.pi * 2, number)
            
            output[:,0]   = np.sqrt(1-z**2) * np.cos(phi)
            output[:,1]   = np.sqrt(1-z**2) * np.sin(phi)
            output[:,2]   = z
            return output
        
        O = rays['origin']
        m = rays['mask']
        normal = np.ones(O.shape) * self.normal
        length = len(m)
        norm = np.empty(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, length)
        
        o_1 = np.cross(normal, [0,0,1])
        o_1[m] /=  np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
        o_2 = np.cross(normal, o_1)
        o_2[m] /=  np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
        
        R = np.empty((length, 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal
        
        norm = np.einsum('ij,ijk->ik', dir_local, R)
        return norm
    
    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.planar_intersect(rays))
        print(' Rays on Graphite:  {:6.4e}'.format(D[m].shape[0]))        
        rays = self.reflect_vectors(X, rays, self.mosaic_norm_generate(rays))
        print(' Rays from Graphite:{:6.4e}'.format(D[m].shape[0]))     
        return rays