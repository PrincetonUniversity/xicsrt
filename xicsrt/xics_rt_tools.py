# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:30:38 2017

@author: James
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import peakutils
from scipy.optimize import leastsq


class OneGaussianFit():
    
    def __init__(self, x, y):
        self.x                  = x
        self.y                  = y
        self.y_fit, self.plsq   = self.one_gaussian_fit() 

        
    def find_peaks(self):
        x               = self.x
        y               = self.y
        index           = peakutils.indexes(y)
        return x, y, index
    
    
    def norm(self, x, mean, sd):
        norm = []
        for i in range(x.size):
            norm += [1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] - mean)**2/(2*sd**2))]
        return np.array(norm)
        
    
    def res(self, p, y, x): 
        m1, a1, sd1, off = p
    
        y_fit = a1 * self.norm(x, m1, sd1) + off
        err = y - y_fit
        return err


    def one_gaussian_fit(self):
        x, y, index = self.find_peaks()

        # Starting Values
        center1= x[index[0]]
        scale1 = y[index[0]]
        offset = y[0]
        stand_dev1 = 1
        parameters = [center1, scale1, stand_dev1, offset]
        p = parameters
        y_real = y
        #y_init = p[2] * norm(x, p[0], p[4]) + p[3] * norm(x, p[1], p[5]) + p[6]
    
        plsq = leastsq(self.res, p, args = (y_real, x))
    
        y_fit = (plsq[0][1] * self.norm(x, plsq[0][0], plsq[0][2]) + plsq[0][3])
    
        return y_fit, plsq


    def get_individual_gaussians(self):
        x, y, y_fit, plsq = self.x, self.y, self.y_fit, self.plsq
        
        g1 = plsq[0][1] * self.norm(x, plsq[0][0], plsq[0][2]) + plsq[0][3]
        
        return x, y, y_fit, g1
    
    
    def integrate_gaussians(self):
        x, plsq = self.x, self.plsq
        
        g1 = plsq[0][1] * self.norm(x, plsq[0][0], plsq[0][2])
 
                
        int1 = sum(g1)

        time_step = x[1] - x[0]
        
        return int1, time_step         
    
    
    
    
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def source_location(distance
                    ,vert_displace
                    ,crystal_location
                    ,crystal_normal
                    ,crystal_curvature
                    ,crystal_spacing
                    ,detector_location
                    ,wavelength):
    # Returns the source location that satisfies the bragg condition met by the
    # detector. Allows for a vertical displacement above and below the 
    # meridional plane
    
    crystal_center = crystal_location + crystal_curvature * crystal_normal
    meridional_normal = np.cross(crystal_location - crystal_center,
                                 detector_location - crystal_center)
    
    meridional_normal = meridional_normal / np.linalg.norm(meridional_normal)
    
    det_to_crys = crystal_location - detector_location
    sol_direction = (det_to_crys
                     - 2 * np.dot(det_to_crys, crystal_normal) * crystal_normal)
    
    sol_direction = sol_direction / np.linalg.norm(sol_direction)
    
    source_location1 = sol_direction * distance + crystal_location
    
    source_location = source_location1 + vert_displace * meridional_normal
    
    return source_location


def source_location_bragg(distance
                          ,vert_displace
                          ,horiz_displace
                          ,crystal_location
                          ,crystal_normal
                          ,crystal_curvature
                          ,crystal_spacing
                          ,detector_location
                          ,wavelength):
    """
    Returns the source on the meridional plain that meets the Bragg condition
    for the given wavelength. Allows for a vertical displacement above and 
    below the meridional plane.
    """

    bragg_angle = np.arcsin( wavelength / (2 * crystal_spacing))
    norm_angle = np.pi/2.0 - bragg_angle

    crystal_center = crystal_location + crystal_curvature * crystal_normal
    
    meridional_normal = np.cross(crystal_location - crystal_center,
                                 detector_location - crystal_center)

    meridional_normal = meridional_normal / np.linalg.norm(meridional_normal)

    rot_mat = rotation_matrix(meridional_normal, norm_angle)
    sol_direction = np.dot(rot_mat, crystal_normal)

    sol_direction = sol_direction / np.linalg.norm(sol_direction)

    source_location1 = sol_direction * distance + crystal_location

    source_location = source_location1 + vert_displace * meridional_normal
    
    sagittal_normal = np.cross(sol_direction, meridional_normal)
    sagittal_normal = sagittal_normal/np.linalg.norm(sagittal_normal)
    
    source_location = source_location + horiz_displace * sagittal_normal
    #from IPython import embed
    #embed()

    return source_location    
    
    
def plot_rows(file_name, row, bin):
    image_array = np.array(Image.open(file_name))
    
    min = int(row - bin // 2)
    max = int(row + bin // 2)
    
    row_array = np.array(image_array[min],dtype=np.int32)
    
    i = 0
    for i in range(min + 1, max +1):
        row_array += np.array(image_array[i], dtype=np.int32)
        i += 1
        
    #plt.figure()
    #print(row_array.T)
    plt.plot(row_array, 'k')
    plt.xlabel('Horizontal Pixels')
    plt.ylabel('Pixel Counts')
    plt.title('Line Intensity (row ' + str(row) + ', bin ' +str(bin) + ')')
    #plt.show()
    return


def plot_rows_data(file_name, row, bint, color):
    image_array = np.array(Image.open(file_name))
    
    mint = int(row - bint // 2)
    maxt = int(row + bint // 2)
    
    row_array = (np.array(image_array[mint].T[0],dtype= np.int32) +
                 np.array(image_array[mint].T[1],dtype= np.int32))

    
    i = 0
    for i in range(mint + 1, maxt +1):
        sample_row0 = np.array(image_array[i].T[0], dtype=np.int32)
        sample_row1 = np.array(image_array[i].T[1], dtype=np.int32)

        row_array = row_array + sample_row0 + sample_row1

        i += 1
        
    #plt.figure()

    #row_new = row_array.T[1]  +  row_array.T[0] 

    #plt.plot(row_new)
    plt.plot(row_array, 'b')

    plt.xlabel('Horizontal Pixels')
    plt.ylabel('Pixel Counts')
    plt.title('Line Intensity (row ' + str(row) + ', bin ' +str(bint) + ')')
    #plt.show()
    return


def get_rows(file_name, row, bin):
    image_array = np.array(Image.open(file_name))

    min = int(row - bin // 2)
    max = int(row + bin // 2)
    
    row_array = np.array(image_array[min],dtype=np.int32)
    
    i = 0
    for i in range(min + 1, max +1):
        row_array += np.array(image_array[i], dtype=np.int32)
        i += 1
        
    
    return row_array


def get_rows_data(file_name, row, bin):
    image_array = np.array(Image.open(file_name))
    image_array = image_array.T
    print(len(image_array[260]))
    min = int(row - bin // 2)
    max = int(row + bin // 2)
    
    row_array = np.array(image_array[min],dtype=np.int32)
    
    i = 0
    for i in range(min + 1, max +1):
        row_array += np.array(image_array[i], dtype=np.int32)
        i += 1
        
    
    return row_array

    
def image_height(file_name):
    # returns percentage of vertical illumination
    image_array = np.array(Image.open(file_name))
    test = np.where(image_array > 0)
    length = len(image_array)
    percent = (length - 2 * test[0][0]) / length
    return round(percent, 3) *100


def tif_to_gray_scale(file_name, scale):
    image = Image.open(file_name).convert("L")
    arr = np.asarray(image) * scale
    plt.figure()
    plt.imshow(arr, cmap='gray_r')
    plt.show()
    return

