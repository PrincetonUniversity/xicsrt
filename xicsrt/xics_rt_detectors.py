# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:09:40 2017

@author: James
"""
from PIL import Image
import numpy as np
from scipy.spatial import cKDTree


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
        self.photon_count = None
        
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
        
         
    def collect_rays(self, O, D, W, w):
        X, clause = self.intersect_check(O, D, self.intersect(O, D))
        self.clause = clause
        index = self.center_tree.query(np.array(X))[1]
        self.photon_count = len(X)

        for ii in range(0, len(index)):
            row, column = self.pixel_row_column(index[ii])
            self.pixel_array[row, column] += w[ii]
            
        return self.pixel_array
        
        
    def output_image(self, image_name):
        
        generated_image = Image.fromarray(self.pixel_array)
        generated_image.save(image_name)        

    
    def height_illuminated(self, O, D, W):
        X, clause = self.intersect_check(O, D, self.intersect(O, D))
        
        vert = np.dot((X - self.position), self.yorientation)

        
        return X, vert
        
