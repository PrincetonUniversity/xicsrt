# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:09:40 2017
Edited on Fri Sep 06 10:41:00 2019

@author: James
@editor: Eugene

Description
-----------
The detector object collects rays and compiles them into a .tif image. It has
a position and rotation in 3D space, as well as a height and width.
"""
from PIL import Image
import numpy as np
from scipy.spatial import cKDTree


class Detector:

    def __init__(
            self
            ,position
            ,normal
            ,orientation
            ,horizontal_pixels 
            ,vertical_pixels
            ,pixel_size):
        
        self.position = position
        self.xorientation = orientation
        self.yorientation = (np.cross(normal, orientation) / 
                             np.linalg.norm(np.cross(normal, orientation)))
        self.normal = normal
        self.width  = horizontal_pixels
        self.height = vertical_pixels
        self.pixel_size = pixel_size
        self.clause = 0
        self.photon_count = None
        
        def pixel_center(row, column):
            # These variables are labled wrong, but the calculaiton is correct.
            row_center = (self.height - 1) / 2
            column_center = (self.width - 1) / 2
            
            xstep = (column - column_center) * self.pixel_size
            ystep = (row_center - row) * self.pixel_size
            center = (self.position + xstep * self.xorientation 
                                    + ystep * self.yorientation)          
            return center

        def create_center_array():
            center_array = []
            for i in range(0, self.height):
                for j in range(0, self.width):
                    point = pixel_center(i, j)
                    center_array.append(point)
            return center_array
    
        self.pixel_array = np.zeros((self.height, self.width))
        self.center_tree = cKDTree(create_center_array())
                
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
        
    def intersect(self, rays):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m] = np.dot((self.position - O[m]), self.normal) / np.dot(D[m], self.normal)
        
        test = (distance > 0) & (distance < 10)
        distance = np.where(test, distance, 0)
        return distance
        
    def intersect_check(self, rays, distance):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X = np.zeros(O.shape, dtype=np.float64)
        xproj = np.zeros(m.shape, dtype=np.float64)
        yproj = np.zeros(m.shape, dtype=np.float64)
        
        #X is the 3D point where the ray intersects the crystal
        X[m] = O[m] + D[m] * distance[m,np.newaxis]
        
        #find which rays hit detector, update mask to remove those that don't    
        xproj[m] = abs(np.dot(X[m] - self.position, self.xorientation))
        yproj[m] = abs(np.dot(X[m] - self.position, self.yorientation))
        m[m] &= ((xproj[m] <= self.width * self.pixel_size / 2) & (
                yproj[m] <= self.height * self.pixel_size / 2))
        return X

    def pixel_row_column(self, pixel_number):
        row = int(pixel_number // self.width)
        column = pixel_number - (row * self.width)
        return row, column
        
    def collect_rays(self, rays):
        m = rays['mask']
        X = self.intersect_check(rays, self.intersect(rays))
        index = self.center_tree.query(X[m])[1]
        self.photon_count = len(m[m])

        w = rays['weight']
        for ii in range(0, len(index)):
            row, column = self.pixel_row_column(index[ii])
            self.pixel_array[row, column] += w[ii]
            
        return self.pixel_array
        
    def output_image(self, image_name, rotate=None):
        if rotate:
            out_array = np.rot90(self.pixel_array)
        else:
            out_array = self.pixel_array
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)       
