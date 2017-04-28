# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:05:57 2017

@author: James
"""

from PIL import Image
import numpy as np
from scipy.spatial import KDTree


class Detector:

    
    def __init__(self, position, normal, orientation, horizontal_pixels, 
                 vertical_pixels, pixel_size):
        self.position = position
        self.xorientation = orientation
        self.yorientation = (np.cross(normal, orientation) / 
                             np.linalg.norm(np.cross(normal, orientation)))
        self.normal = normal
        self.width = horizontal_pixels
        self.height = vertical_pixels
        self.pixel_size = pixel_size
        self.clause = 0
        
        def pixel_center(row, column):
            row_center = self.height / 2 - .5
            column_center = self.width / 2 - .5
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.position + xstep * self.xorientation 
                                    + ystep * self.yorientation)  
        
            return center
                        
            
        def create_center_array():
            center_array = []
            i = 0
            for i in range(0, self.height):
                j = 0
                for j in range(0, self.width):
                    point = pixel_center(i, j)
                    center_array.append(point)
                    j += 1
            
                i += 1
            
            return center_array
    
        self.pixel_array = np.zeros((self.height, self.width))
        self.center_tree = KDTree(create_center_array())
        
        
    def pixel_corner(self, row, column, corner):
        row_center = self.height / 2 - .5
        column_center = self.width / 2 - .5
        
        xstep = (column - column_center) * self.pixel_size
        ystep = (row_center - row) * self.pixel_size
        center = (self.position + xstep * self.xorientation 
                                    + ystep * self.yorientation)  
        half = self.pixel_size * .5
            
        corner0 = (center - half * self.xorientation 
                                    + half * self.yorientation)   
        corner1 = (center + half * self.xorientation 
                                    + half * self.yorientation)   
        corner2 = (center + half * self.xorientation 
                                    - half * self.yorientation)   
        corner3 = (center - half * self.xorientation 
                                    - half * self.yorientation)   
        corners = [corner0, corner1, corner2, corner3]
        return corners[corner]

  
    def corner_points(self):
        corners = np.array([self.pixel_corner(0, 0, 0), 
                            self.pixel_corner(0, 194, 1),
                            self.pixel_corner(1474, 194, 2),
                            self.pixel_corner(1474, 0, 3)])
        
        return corners        
        
        
    def intersect(self, O, D):
        distance = np.dot((self.position - O), self.normal)/np.dot(D, self.normal)
        test = (distance > 0) & (distance < 10)
        return np.where(test, distance, 0)
      
    def intersect_check(self, O, D, distance):
        test = (distance != 0)
        O = O[test]
        D = D[test]
        distance = distance[test]

        X = O + D * distance[:,np.newaxis]
        yproj = abs(np.dot(X - self.position, self.yorientation))
        xproj = abs(np.dot(X - self.position, self.xorientation)) 
        
        clause = (xproj <= .5 * self.width * self.pixel_size) & (yproj <= .5 * self.height * self.pixel_size)        

        return X[clause], clause

  
    def pixel_row_column(self, pixel_number):
        row = int(pixel_number // self.width)
        column = pixel_number - (row * self.width)
        
        return row, column  
        
        
    def collect_rays(self, O, D, W):
        X, clause = self.intersect_check(O, D, self.intersect(O, D))
        self.clause = clause
        index = self.center_tree.query(np.array(X))[1]
        numbers = [71510, 287430]
        np.delete(index, numbers)
        print('Rays Collected: ' + str(len(X)))
        for number in index:
            row, column = self.pixel_row_column(number)
            self.pixel_array[row][column] += 1  
        return
        
    def output_image(self, image_name):
        generated_image = Image.fromarray(self.pixel_array)
        generated_image.save(image_name)        

        
  

          
class SphericalCrystal:
    def __init__(self, location, normal, orientation, radius_of_curvature, 
                 crystal_spacing, rocking_curve, reflectivity, width, height):
        
        self.radius = radius_of_curvature
        self.location = location
        self.normal = normal
        self.xorientation = orientation
        self.yorientation = (np.cross(normal, orientation) / 
                             np.linalg.norm(np.cross(normal, orientation)))
        self.center = self.radius * self.normal + self.location
        self.crystal_spacing = crystal_spacing
        #self.bragg_angle = bragg_angle
        self.rocking_curve = rocking_curve
        self.reflectivity = reflectivity
        self.width = width
        self.height = height
        self.pixel_size = self.width / 1001
        self.pixel_width = round(self.width / self.pixel_size)
        self.pixel_height = round(self.height / self.pixel_size)        
        
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
            i = 0
            for i in range(0, self.pixel_height):
                j = 0
                for j in range(0, self.pixel_width):
                    point = pixel_center(i, j)
                    center_array.append(point)
                    j += 1
                
                i += 1
                
            return center_array
            
        self.pixel_array = np.zeros((self.pixel_height, self.pixel_width))
        self.center_tree = KDTree(create_center_array())
            
        
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
        # intersection of ray having origin O and direction D
        vector = (O - self.center)

        
        test = ((np.dot(D, vector[0])) **2 
                - np.linalg.norm(vector[0]) ** 2 + self.radius ** 2)
        sq = np.sqrt(np.maximum(0, test))
        
        a = - np.dot(D, vector[0]) + sq
        b = - np.dot(D, vector[0]) - sq

        distance = np.where((a > 0) & (a > b), a, b)
    
        hit = (test > 0) & (distance > 0)  
        
        #print(hit)
        return np.where(hit, distance, 0)
        
    def intersect_check(self, O, D, W, distance):
        test = None
        test = (distance != 0) #finds which rays intersect, removes others
        O = O[test]
        D = D[test]
        W = W[test]
        distance = distance[test]
        
        X = O + D * distance[:,np.newaxis]
        yproj = abs(np.dot(X - self.location, self.yorientation))
        xproj = abs(np.dot(X - self.location, self.xorientation))
        clause = None
        clause = (xproj <= self.width) & (yproj <= self.height)
        return X[clause], D[clause], W[clause]
    
    def angle_check(self, X, D, norm, W):
        
        # returns vectors that satisfy the bragg condition
        clause = None
        #print(W)
        bragg_angle = np.arcsin( W / (2 * self.crystal_spacing))
        
        dot = np.einsum('ij,ij->i',D, -1 * norm)[:, np.newaxis]

        angle = np.arccos(dot/self.norm(D))
        angle = (np.pi * .5) - angle
        angle_min = bragg_angle - self.rocking_curve
        angle_max = bragg_angle + self.rocking_curve

        clause = (angle <= angle_max) & (angle >= angle_min)
        clause = np.ndarray.flatten(clause)
        
        #return X[clause], D[clause], norm[clause], W[clause]
        return X, D, norm, W


    def reflect_vectors(self, X, D, W):
        
        norm = self.normalize(self.center - X)
        
        X, D, norm, W = self.angle_check(X, D, norm, W) #returns the sufficient vectors        
       
        D = D - 2 * np.einsum('ij,ij->i', D, norm)[:, np.newaxis] * norm
        
        return X, D, W    
            
    def light(self, O, D, W):

        X, D, W = self.intersect_check(O, D, W, self.intersect(O, D))
        
        print('Rays that hit Crystal: ' + str(len(D)))
        
        O, D, W = self.reflect_vectors(X, D, W) #new origin and direction
        
        #print('Rays from Crystal: ' + str(len(D)))

        return O, D, W


    def pixel_row_column(self, pixel_number):
        row = int(pixel_number // self.pixel_width)
        column = pixel_number - (row * self.pixel_width)

        return row, column
        
        
    def collect_rays(self, O, D, W):
        X = O
        index = self.center_tree.query(np.array(X))[1]

        for number in index:
            row, column = self.pixel_row_column(number)
            self.pixel_array[row][column] += 1  
        return

        
    def output_image(self, image_name):
        generated_image = Image.fromarray(self.pixel_array)
        generated_image.save(image_name)           


        

class PointSource(object):
    
    """Create a point source object that emitts light of a certain wavelength.
    Emits in four pi steradians.
    Wavelength in angstroms.
    """
    
    def __init__(self, position, intensity, wavelength, temperature):
        self.position = position
        self.wavelength = wavelength
        self.intensity = intensity #photons per second
        self.temp = temperature
        
    def random_direction(self):
        direction = np.random.uniform(-1, 1, 3)
        direction = direction/np.linalg.norm(direction)
        return direction.tolist()
        
        
    def random_wavelength(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
        
        
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave
        
    
    def generate_rays(self,duration):

        number_of_rays = self.intensity * duration 
        origin = [[self.position[0], self.position[1], self.position[2]]] * number_of_rays
        O = np.array(origin) 
        
        ray_list = None
        ray_list = []

        wavelength_list = None
        wavelength_list = []

        i = 0
        for i in range(0, number_of_rays):
            ray = self.random_direction()
            ray_list.append(ray)
            
            wavelength = self.random_wavelength()
            wavelength_list.append(wavelength)
            
            i += 1
            
        D = np.array(ray_list)
        W = np.array(wavelength_list)
        
        return O, D, W




        
class DirectedSource(object):
    
    """Create a directed source object that emitts light of a certain 
    wavelength in a defined cone. Wavelength in angstroms.
    """
    
    def __init__(self, position, direction, spread, intensity, wavelength, 
                 temperature, mass_number):
        self.position = position
        self.direction = direction
        self.spread = spread             # in degrees
        self.wavelength = wavelength     # in angstroms
        self.intensity = intensity       # photons per second

        self.xorientation = (np.cross(self.direction, self.position)/ 
                             np.linalg.norm(np.cross(self.direction, self.position)))
        self.yorientation = (np.cross(self.xorientation, self.direction) / 
                             np.linalg.norm(np.cross(self.xorientation,
                                                     self.direction)))    
        self.temp = temperature          # in eV
        self.mass_number = mass_number   # mass number of source material
        
        
    def random_direction(self):
        # in spread
        # spread converted to radians
        spread_rad = self.spread / 180 * np.pi
        theta = np.random.uniform(-spread_rad, spread_rad, 1)
        phi = np.random.uniform(0, 2*np.pi, 1)
        direction = [np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)]
        
        R = ([self.xorientation, self.yorientation, self.direction])
        R = np.transpose(R)
        direction = np.dot(R, direction)
        direction = np.ndarray.flatten(direction)      
        direction = direction/np.linalg.norm(direction)
        
        return direction.tolist()
    
     
    def random_direction_new(self):
        direction = np.random.uniform(-1, 1, 3)
        direction = direction/np.linalg.norm(direction)
        
        def theta(direction):
            angle = np.arccos(direction[0]/np.linalg.norm(direction))
            return angle
            
        while theta(direction) >= self.spread:
            direction = np.random.uniform(-1, 1, 3)
            direction = direction/np.linalg.norm(direction)        
            
        
        return direction.tolist()
           
    def random_wavelength(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
        
        
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave
        
    
    def generate_rays(self,duration):

        number_of_rays = self.intensity * duration 
        origin = [[self.position[0], self.position[1], self.position[2]]] * number_of_rays
        O = np.array(origin) 
        
        ray_list = None
        ray_list = []

        wavelength_list = None
        wavelength_list = []

        i = 0
        for i in range(0, number_of_rays):
            ray = self.random_direction_new()
            ray_list.append(ray)
            
            wavelength = self.random_wavelength()
            wavelength_list.append(wavelength)
            
            i += 1
            
        D = np.array(ray_list)
        W = np.array(wavelength_list)
        
        return O, D, W

                
def raytrace(duration, source, detector, *optics):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), and wavelength (W).
    """
    
    O, D, W = source.generate_rays(duration)
    print('Rays Generated: ' + str(len(D)))
    
    
    for optic in optics:
        O, D, W = optic.light(O, D, W)
        optic.collect_rays(O, D, W)
        print('Rays from Crystal: ' + str(len(D)))
    
    detector.collect_rays(O, D, W)
    
    return      
        
        
def raytrace_special(duration, source, detector, crystal):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), and wavelength (W).
    """
    
    O, D, W = source.generate_rays(duration)
    print('Rays Generated: ' + str(len(D)))
    
    

    O, D, W = crystal.light(O, D, W)
    print('Rays from Crystal: ' + str(len(D)))
    
    detector.collect_rays(O, D, W)
    clause = detector.clause
    
    #O1, D1, W1 = O[clause], D[clause], W[clause]
    O1, D1, W1 = O, D, W
    crystal.collect_rays(O1, D1, W1)
    
    return 