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

from xicsrt.xics_rt_objects import TraceObject


class Detector(TraceObject):
    def __init__(self, detector_input):
        super().__init__(
            detector_input['position']
            ,detector_input['normal']
            ,detector_input['orientation'])
        
        self.__name__       = 'PilatusDetector'
        self.position       = detector_input['position']
        self.normal         = detector_input['normal']
        self.xorientation   = detector_input['orientation']
        self.yorientation   = np.cross(self.normal, self.xorientation)
        self.yorientation  /= np.linalg.norm(self.yorientation)
        self.width          = detector_input['width']
        self.height         = detector_input['height']
        self.depth          = 0
        self.pixel_size     = detector_input['pixel_size']
        self.pixel_width    = detector_input['horizontal_pixels']
        self.pixel_height   = detector_input['vertical_pixels']
        self.pixel_array    = np.zeros((self.pixel_width, self.pixel_height))
        self.miss_checks    = detector_input['do_miss_checks']
        self.photon_count   = None
        self.pixel_array_size_check()
        
    def pixel_array_size_check(self):
        ## Before loading anything up, check if the pixel array is mishapen
        failure  = False
        failure |= (self.pixel_width  != int(round(self.width  / self.pixel_size)))
        failure |= (self.pixel_height != int(round(self.height / self.pixel_size)))
            
        if failure:
            print('{} pixel array is mishapen'.format(self.__name__))
            print('Pixel array width/height and detector width/height are disproportionate')
            print('Please check {} code'.format(self.__name__))
            
            print('{} width  = {}'.format(self.__name__, self.width))
            print('{} height = {}'.format(self.__name__, self.height))
            print('{} pixel width  = {}'.format(self.__name__, self.pixel_width))
            print('{} pixel height = {}'.format(self.__name__, self.pixel_height))
            raise Exception
        
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
        
        #X is the 3D point where the ray intersects the detector
        X[m] = O[m] + D[m] * distance[m,np.newaxis]
        
        #find which rays hit detector, update mask to remove those that don't    
        xproj[m] = abs(np.dot(X[m] - self.position, self.xorientation))
        yproj[m] = abs(np.dot(X[m] - self.position, self.yorientation))
        if self.miss_checks is True:
            m[m] &= ((xproj[m] <= self.width / 2) & (yproj[m] <= self.height/ 2))
        return X, rays
    
    def light(self, rays):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.intersect(rays))
        print(' Rays on Detector:  {:6.4e}'.format(D[m].shape[0]))
        O[:] = X[:]
        return rays
    
    def collect_rays(self, rays):
        X = rays['origin']
        m = rays['mask'].copy()
        
        num_lines = np.sum(m)
        if num_lines > 0:
            # Transform the intersection coordinates from external coordinates
            # to local optical coordinates.
            point_loc = self.point_to_local(X[m])
            
            # Bin the intersections into pixels using integer math.
            pix = np.zeros([num_lines, 3], dtype = int)
            pix = np.round(point_loc / self.pixel_size).astype(int)
            
            # Check to ascertain if origin pixel is even or odd
            if (self.pixel_width % 2) == 0:
                pix_min_x = self.pixel_width//2
            else:
                pix_min_x = (self.pixel_width + 1)//2
                
            if (self.pixel_height % 2) == 0:
                pix_min_y = self.pixel_height//2
            else:
                pix_min_y = (self.pixel_height + 1)//2
            
            pix_min = np.array([pix_min_x, pix_min_y, 0], dtype = int)
            
            # Convert from pixels, which are centered around the origin, to
            # channels, which start from the corner of the optic.
            channel    = np.zeros([num_lines, 3], dtype = int)
            channel[:] = pix[:] - pix_min
            
            # I feel like there must be a faster way to do this than to loop over
            # every intersection.  This could be slow for large arrays.
            for ii in range(len(channel)):
                self.pixel_array[channel[ii,0], channel[ii,1]] += 1
        
        self.photon_count = len(m[m])
        return self.pixel_array   
        
    def output_image(self, image_name, rotate=None):
        if rotate:
            out_array = np.rot90(self.pixel_array)
        else:
            out_array = self.pixel_array
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)
        
