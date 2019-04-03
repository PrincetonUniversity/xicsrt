# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:05:57 2017

@author: James
"""
from PIL import Image
import numpy as np
from scipy.spatial import cKDTree
from scipy import signal
import matplotlib.pyplot as plt
          
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

        # Generate a gaussian rocking curve.
        # For additional accuracy an arbitrary curve can be loaded here.
        self.rocking_gaussian = signal.gaussian(100, std=22)

        
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
        
        i = 0
        for i in range(int(self.pixel_height/4), int(3/4*self.pixel_height)):
            j = 0
            for j in range(int(self.pixel_width/4), int(3/4*self.pixel_width)):
                point = self.pixel_center(i, j)
                point = point.tolist()
                
                center_array.append(point)
                j += 1
                
            i += 1
            
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
        
        
    def intersect(self, O, D):
        """
        This calculation is copied from:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/
                minimal-ray-tracer-rendering-simple-shapes/
                ray-sphere-intersection
        """

        # TEMPORARY:
        # Check to see if ray crosses through origin.
        #distance_center = np.zeros(len(O[:,0]))
        #for ii in range(len(O)):
        #    d = (self.location - O[ii])/D[ii]
        #    distance_center[ii] = d[0]

        distance = np.zeros(O.shape[0])

        L = (self.center - O)
        t_ca = np.einsum('ij,ij->i', L, D)

        # If t_ca is less than zero, then there is no intersection.
        mask = t_ca > 0

        d = np.sqrt(np.einsum('ij,ij->i',L[mask,:] ,L[mask,:]) - t_ca[mask]**2)

        t_hc = np.sqrt(self.radius**2 - d**2)

        t_0 = t_ca - t_hc
        t_1 = t_ca + t_hc

        distance[mask] = np.where(t_0 > t_1, t_0, t_1)

        return distance
        
        
    def intersect_check(self, O, D, W, w, distance):
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
        
        test = None
        test = (distance != 0) #finds which rays intersect, removes others
        O = O[test]
        D = D[test]
        W = W[test]
        w = w[test]
        
        distance = distance[test]
        
        X = O + D * distance[:,np.newaxis]
        yproj = abs(np.dot(X - self.location, self.yorientation))
        xproj = abs(np.dot(X - self.location, self.xorientation))

        clause = ((xproj <= self.width * .5) & (yproj <= self.height * .5))
        
        # nproj = abs(np.dot(X - self.location, self.normal))        
        # n_value = (self.radius - (self.radius**2 - (self.height * .5) ** 2) **.5)
        # clause &= (nproj <= n_value)
        
        return X[clause], D[clause], W[clause], w[clause]
    

    def point_on_crystal_check(self, point):
        max_dist = np.sqrt((self.width *.5)**2 + (self.height * .5) ** 2)
        
        displace = point - self.location
        distance = np.linalg.norm(displace)
        
        yproj = abs(np.dot(displace, self.yorientation))
        xproj = abs(np.dot(displace, self.xorientation))
        clause = None
        clause = (distance <= max_dist) & (xproj <= self.width * .5) & (yproj <= self.height * .5)
        
        return clause
        
    
    def rocking_curve_width(self, bragg_angle, angle):

        rocking_window  =  self.rocking_curve
        
        window_range    = np.linspace(bragg_angle - rocking_window, 
                                      bragg_angle + rocking_window,
                                      100)
        
        index           = (np.abs(window_range - angle)).argmin()

        
        return index


    def adjust_weight(self, w, bragg_angle, angle):
        w_new = []
        for ii in range(0, len(w)):
            index = self.rocking_curve_width(bragg_angle[ii][0], angle[ii][0])
            w_new.append(self.rocking_gaussian[index] * w[ii])       
            
        w = np.array(w_new)       
        return w
    
    
    def angle_check(self, X, D, W, w, norm):
        # returns vectors that satisfy the bragg condition
        clause = None
        bragg_angle = np.arcsin( W / (2 * self.crystal_spacing))

        dot = np.einsum('ij,ij->i',D, -1 * norm)[:, np.newaxis]
        
        angle = np.arccos(dot/self.norm(D))

        angle = (np.pi * .5) - angle

        angle_min = bragg_angle - self.rocking_curve
        angle_max = bragg_angle + self.rocking_curve


        clause = (angle <= angle_max) & (angle >= angle_min)

        clause = np.ndarray.flatten(clause)
        #print(angle[clause])
        #print(bragg_angle[clause])
        
        w = self.adjust_weight(w[clause], bragg_angle[clause], angle[clause])

        return X[clause], D[clause], W[clause], w, norm[clause]


    def reflect_vectors(self, X, D, W, w):
        
        norm = self.normalize(self.center - X)

        # Check which vectors meet the Bragg condition (with rocking curve)
        X, D, W, w, norm = self.angle_check(X, D, W, w, norm)

        # Perform reflection around nermal vector.
        D = D - 2 * np.einsum('ij,ij->i', D, norm)[:, np.newaxis] * norm
        
        return X, D, W, w  

    
    def light(self, O, D, W, w):

        X, D, W, w = self.intersect_check(O, D, W, w, self.intersect(O, D))
        
        print('Rays that hit Crystal: ' + str(len(D)))
        
        O, D, W, w = self.reflect_vectors(X, D, W, w) #new origin and direction
        
        #print('Rays from Crystal: ' + str(len(D)))

        return O, D, W, w


    def pixel_row_column(self, pixel_number):
        row = int(pixel_number // self.pixel_width)
        column = pixel_number - (row * self.pixel_width)

        return row, column
        
        
    def collect_rays(self, O, D, W, w):
        X = O
        index = self.center_tree.query(np.array(X))[1]

        for number in index:
            row, column = self.pixel_row_column(number)
            self.pixel_array[row, column] += 1
            
        return

        
    def output_image(self, image_name):
        generated_image = Image.fromarray(self.pixel_array)
        generated_image.save(image_name)          

        
    def height_illuminated(self, X):
        vert = np.dot((X - self.location), self.yorientation)
        
        return X, vert
        
