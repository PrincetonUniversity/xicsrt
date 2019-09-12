# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:05:57 2017
Edited on Fri Sep 06 11:06:00 2019

@author: James
@editor: Eugene

Description
-----------
The spherical quartz crystal and Highly Oriented Pyrolytic Graphite film (NYI) 
that reflect X-rays that satisfy the Bragg condition. Optical elements have a
location and rotation in 3D space, optical properties such as crystal spacing,
rocking curve, and reflectivity, as well as a height and width.
"""
from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
from scipy import signal
          
class SphericalCrystal:
    def __init__(
            self
            ,location
            ,normal
            ,orientation
            ,radius_of_curvature 
            ,crystal_spacing
            ,rocking_curve
            ,reflectivity
            ,width
            ,height
            ,pixel_scaling):
        
        self.radius = radius_of_curvature
        self.location = location
        self.normal = normal
        self.xorientation = orientation
        self.yorientation = (np.cross(normal, orientation) / 
                             np.linalg.norm(np.cross(normal, orientation)))
        self.center = self.radius * self.normal + self.location
        self.crystal_spacing = crystal_spacing
        self.rocking_curve = rocking_curve
        self.reflectivity = reflectivity
        self.width = width
        self.height = height
        self.pixel_size = self.width / pixel_scaling
        self.pixel_width = int(round(self.width / self.pixel_size))
        self.pixel_height = int(round(self.height / self.pixel_size))        
        
        def pixel_center(row, column):
            row_center = self.pixel_height / 2 - .5
            column_center = self.pixel_width / 2 - .5
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.location + xstep * self.xorientation 
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
        
    def pixel_center(self, row, column):
        row_center = self.pixel_height / 2 - .5
        column_center = self.pixel_width / 2 - .5
            
        xstep = (column - column_center) * self.pixel_size
        ystep = (row_center - row) * self.pixel_size
        center = (self.location + xstep * self.xorientation 
                                    + ystep * self.yorientation)
            
        return center        
        
    def create_center_array_new(self):
        center_array = [] 
        for ii in range(int(self.pixel_height/4), int(3/4*self.pixel_height),1):
            for jj in range(int(self.pixel_width/4), int(3/4*self.pixel_width),1):
                point = self.pixel_center(ii, jj)
                point = point.tolist()
                center_array.append(point)
            
        return np.array(center_array)   
        
    def normalize(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        vector_norm = vector / magnitude[:, np.newaxis]
        return vector_norm
        
    def norm(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        return magnitude      
        
    def intersect(self, rays):
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
        L     = (self.center - O)
        t_ca  = np.einsum('ij,ij->i', L, D)
        mag_L = np.linalg.norm(L, axis=1)
        
        # If t_ca is less than zero, then there is no intersection.
        # Use mask to only perform calculations on rays that hit the crystal
        # Update: James 9/3/2019 
        #   If O is inside of radius of curvature, t_ca can be zero
        #   added simple statement to allow source in sphere of curvature    
        m[m] &= ((t_ca > 0) & (mag_L > self.radius))
        
        #d is the line-of-sight distance between ray and center of curvature
        d[m]    = np.sqrt(np.einsum('ij,ij->i',L[m] ,L[m]) - t_ca[m]**2)
        t_hc[m] = np.sqrt(self.radius**2 - d[m]**2)
        
        t_0[m] = t_ca[m] - t_hc[m]
        t_1[m] = t_ca[m] + t_hc[m]

        distance[m] = np.where(t_0 > t_1, t_0, t_1)
        return distance
        
    def intersect_check(self, rays, distance):
        """
        Check if ray intesects the optic within the geometrical bounds.

        Programming Notes
        -----------------
        I am not sure why we need the normal projection check here.
        The only time that I could see this being useful is if the
        intersect calculator returned the wrong intersect solution
        (the otheside of the crystal sphere from the optic).
        For now I have disabled it -- Novimir 2019-04-01
        """
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X = np.zeros(O.shape, dtype=np.float64)
        xproj = np.zeros(m.shape, dtype=np.float64)
        yproj = np.zeros(m.shape, dtype=np.float64)
        
        #X is the 3D point where the ray intersects the crystal
        X[m] = O[m] + D[m] * distance[m,np.newaxis]
        
        #find which rays hit crystal, update mask to remove those that don't    
        xproj[m] = abs(np.dot(X[m] - self.location, self.xorientation))
        yproj[m] = abs(np.dot(X[m] - self.location, self.yorientation))
        m[m] &= ((xproj[m] <= self.width / 2) & (yproj[m] <= self.height / 2))
        return X, rays
    
    """
    def rocking_curve_adjust_weight(self, rays, bragg_angle, angle):
        
        Weight each ray according to the rocking curve.

        Using this type of weighting method will produce an image much faster
        than the filter method. There is an issue however that the image
        no longer follows Poisson statistics.  If the final image is to be used
        in fitting applications, this method is not reccomented.

        In principal it would be possible to track the statistics separately
        and generate an images of sigmas. This is a bit complicated, and would
        require changes in any fitting software to accomodate this extra
        information. 
        
        w = rays['weight']

        # Convert from FWHM to sigma.
        sigma = self.rocking_curve/np.sqrt(2*np.log(2))/2

        # Normalized Gaussian.
        w = np.exp(-np.power(angle - bragg_angle, 2.) / (2 * sigma**2))
        return w
    """
    def rocking_curve_filter(self, bragg_angle, angle):
        """ 
        Treat the rocking curve as a probability distribution and
        generate a filter mask.
        
        This method is much less efficent than using weighting, but
        ensures that the final image has normal poisson statistics.
        """
        
        # Convert from FWHM to sigma.
        sigma = self.rocking_curve / np.sqrt(2 * np.log(2)) / 2

        # Normalized Gaussian.
        p = np.exp(-np.power(angle - bragg_angle, 2) / (2 * sigma**2))

        test = np.random.uniform(0.0, 1.0, len(angle))
        mask = p.flatten() > test
        
        return mask
        
    def angle_check(self, X, rays, norm):
        D = rays['direction']
        W = rays['wavelength']
#       w = rays['weight']
        m = rays['mask']
        
        bragg_angle = np.zeros(m.shape, dtype=np.float64)
        dot = np.zeros(m.shape, dtype=np.float64)
        angle = np.zeros(m.shape, dtype=np.float64)
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the crystal
        bragg_angle[m] = np.arcsin( W[m] / (2 * self.crystal_spacing))
        dot[m] = np.einsum('ij,ij->i',D[m], -1 * norm[m])
        angle[m] = (np.pi / 2) - np.arccos(dot[m]/self.norm(D[m]))
        """
        # For a discription of these options see the headers for the
        # following methods:
        #  - rocking_curve_adjust_weight
        #  - rocking_curve_filter
        use_weight = False
        use_filter = True
         if use_weight:
            # Decide where to cutoff reflections.
            # In this case once the reflectivity gets below 0.1%.
            # (This assumes that the rocking curve is gaussian.
             fraction = 0.001
             rocking_hwfm = self.rocking_curve/2 * np.sqrt(np.log(1.0/fraction)/np.log(2))
            
             print('rocking_hwfw:', rocking_hwfm)
             angle_min = bragg_angle - rocking_hwfm
             angle_max = bragg_angle + rocking_hwfm
            
             clause = (angle <= angle_max) & (angle >= angle_min)
             clause = np.ndarray.flatten(clause)
         
             w = self.rocking_curve_adjust_weight(w[clause], bragg_angle[clause], angle[clause])
        if use_filter:
        """
        #check which rays satisfy bragg, update mask to remove those that don't
        m[m] &= self.rocking_curve_filter(bragg_angle[m], angle[m])
        return rays, norm

    def reflect_vectors(self, X, rays):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        norm = np.zeros(X.shape, dtype=np.float64)
        norm[m] = self.normalize(self.center - X[m])
        
        # Check which vectors meet the Bragg condition (with rocking curve)
        rays, norm = self.angle_check(X, rays, norm)
        
        # Perform reflection around normal vector, creating new rays with new
        # origin O = X and new direction D
        
        O[m]  = X[m]
        D[m] -= 2 * np.einsum('ij,ij->i', D[m], norm[m])[:, np.newaxis] * norm[m]
        
        return rays 

    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.intersect(rays))
        print(' Rays on Crystal:   {:6.4e}'.format(D[m].shape[0]))        
        rays = self.reflect_vectors(X, rays) #new origin and direction
        print(' Rays from Crystal: {:6.4e}'.format(D[m].shape[0]))     
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
        

class GraphiteMirror:
    def __init__(
            self
            ,location
            ,normal
            ,orientation
            ,crystal_spacing
            ,rocking_curve
            ,reflectivity
            ,mosaic_spread
            ,width
            ,height
            ,pixel_scaling):
        
        self.location = location
        self.normal = normal
        self.xorientation = orientation
        self.yorientation = (np.cross(normal, orientation) / 
                             np.linalg.norm(np.cross(normal, orientation)))
        self.center = self.location
        self.crystal_spacing = crystal_spacing
        self.rocking_curve = rocking_curve
        self.reflectivity = reflectivity
        self.width = width
        self.height = height
        self.pixel_size = self.width / pixel_scaling
        self.pixel_width = int(round(self.width / self.pixel_size))
        self.pixel_height = int(round(self.height / self.pixel_size))      
        
        def pixel_center(row, column):
            row_center = self.pixel_height / 2 - .5
            column_center = self.pixel_width / 2 - .5
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.location + xstep * self.xorientation 
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
    
    def pixel_center(self, row, column):
        row_center = self.pixel_height / 2 - .5
        column_center = self.pixel_width / 2 - .5
            
        xstep = (column - column_center) * self.pixel_size
        ystep = (row_center - row) * self.pixel_size
        center = (self.location + xstep * self.xorientation 
                                    + ystep * self.yorientation)
            
        return center        
        
    def create_center_array_new(self):
        center_array = [] 
        for ii in range(int(self.pixel_height/4), int(3/4*self.pixel_height),1):
            for jj in range(int(self.pixel_width/4), int(3/4*self.pixel_width),1):
                point = self.pixel_center(ii, jj)
                point = point.tolist()
                center_array.append(point)
            
        return np.array(center_array)   
        
    def normalize(self, vector):
        value = np.einsum('ij,ij->i', vector, vector) ** .5
        vector_norm = vector / value[:, np.newaxis]
        return vector_norm
    
    def norm(self, vector):
        value = np.einsum('ij,ij->i', vector, vector) ** .5
        return value[:, np.newaxis]        

    def wavelength_to_bragg_angle(self, wavelength):
        # wavelength in angstroms
        # angle in radians
        angle = np.arcsin(wavelength / (2 * self.crystal_spacing))
        return angle

    def intersect(self, rays):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(O.shape[0])
        distance[m] = np.dot((self.position - O[m]), self.normal)/np.dot(D[m], self.normal)
        
        test = (distance > 0) & (distance < 10)
        distance = np.where(test, distance, 0)
        return distance
    
    def intersect_check(self, rays, distance):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        test = (distance != 0)
        m &= test
        
        X = np.zeros(O.shape, dtype=np.float64)
        X[m] = O[m] + D[m] * distance[m,np.newaxis]
        
        yproj = abs(np.dot(X[m] - self.position, self.yorientation))
        xproj = abs(np.dot(X[m] - self.position, self.xorientation))
        
        clause = ((xproj <= self.width * .5) & (yproj <= self.height * .5))
        m[m] &= clause
        return X, m
    
    """
    NOTE: Mosaic spreading needs to be implemented somewhere here. Without
    mosaic spreading, the graphite mirror will behave like a regular crystal
    plane mirror.
    """
    
    def rocking_curve_filter(self, bragg_angle, angle):  
        # Convert from FWHM to sigma.
        sigma = self.rocking_curve/np.sqrt(2*np.log(2))/2

        # Normalized Gaussian.
        p = np.exp(-np.power(angle - bragg_angle, 2.) / (2 * sigma**2))

        test = np.random.uniform(0.0, 1.0, len(angle))
        mask = p.flatten() > test
        
        return mask
    
    def angle_check(self, X, rays, norm):
        D = rays['direction']
        W = rays['wavelength']
        #w = rays['weight']
        m = rays['mask']
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the crystal
        bragg_angle = np.arcsin( W[m] / (2 * self.crystal_spacing))
        dot = np.einsum('ij,ij->i',D[m], -1 * norm[m])[:, np.newaxis]
        angle = np.arccos(dot/self.norm(D[m]))
        angle = (np.pi * .5) - angle

        """
        # For a discription of these options see the headers for the
        # following methods:
        #  - rocking_curve_adjust_weight
        #  - rocking_curve_filter
        use_weight = False
        use_filter = True
        if use_weight:
            # Decide where to cutoff reflections.
            # In this case once the reflectivity gets below 0.1%.
            # (This assumes that the rocking curve is gaussian.
             fraction = 0.001
             rocking_hwfm = self.rocking_curve/2 * np.sqrt(np.log(1.0/fraction)/np.log(2))
             
             print('rocking_hwfw:', rocking_hwfm)
             angle_min = bragg_angle - rocking_hwfm
             angle_max = bragg_angle + rocking_hwfm
             
             clause = (angle <= angle_max) & (angle >= angle_min)
             clause = np.ndarray.flatten(clause)
         
             w = self.rocking_curve_adjust_weight(w[clause], bragg_angle[clause], angle[clause])
        if use_filter:
        """
        clause = self.rocking_curve_filter(bragg_angle, angle)
        m[m] &= clause
        return rays, norm
    
    def reflect_vectors(self, X, rays):
        norm = self.normalize(self.center - X)

        # Check which vectors meet the Bragg condition (with rocking curve)
        rays, norm = self.angle_check(X, rays, norm)

        # Perform reflection around normal vector. These lines can be optimized
        # more, as they are generating a new array D which takes up mem/speed
        D = rays['direction']
        D[:] = D - 2 * np.einsum('ij,ij->i', D, norm)[:, np.newaxis] * norm
        
        return rays 
    
    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.intersect(rays))
        print(' Rays on Graphite:   {:6.4e}'.format(D[m].shape[0]))        
        rays = self.reflect_vectors(X, rays) #new origin and direction
        print(' Rays from Graphite: {:6.4e}'.format(D[m].shape[0]))     
        return rays

    def pixel_row_column(self, pixel_number):
        row = int(pixel_number // self.pixel_width)
        column = pixel_number - (row * self.pixel_width)
        return row, column        

    def collect_rays(self, rays):
        X = rays['origin']
        m = rays['mask']
        index = self.center_tree.query(np.array(X[m]))[1]
        self.photon_count = len(X[m])

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
        