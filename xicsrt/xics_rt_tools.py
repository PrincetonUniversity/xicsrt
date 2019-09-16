# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:30:38 2017

@author: James
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import leastsq

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
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

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
    
    meridional_normal /=  np.linalg.norm(meridional_normal)
    
    det_to_crys = crystal_location - detector_location
    sol_direction = (det_to_crys
                     - 2 * np.dot(det_to_crys, crystal_normal) * crystal_normal)
    
    sol_direction /= np.linalg.norm(sol_direction)
    
    source_location = sol_direction * distance + crystal_location
    
    source_location += vert_displace * meridional_normal
    
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
    Returns the source on the meridional plane that meets the Bragg condition
    for the given wavelength. Allows for a vertical displacement above and 
    below the meridional plane.
    """

    bragg_angle = np.arcsin( wavelength / (2 * crystal_spacing))
    norm_angle = np.pi/2.0 - bragg_angle
    crystal_center = crystal_location + crystal_curvature * crystal_normal
    
    meridional_normal = np.cross(crystal_location - crystal_center,
                                 detector_location - crystal_center)
    meridional_normal /= np.linalg.norm(meridional_normal)

    rot_mat = rotation_matrix(meridional_normal, norm_angle)
    sol_direction =  np.dot(rot_mat, crystal_normal)
    sol_direction /= np.linalg.norm(sol_direction)

    source_location =  sol_direction * distance + crystal_location
    source_location += vert_displace * meridional_normal
    
    sagittal_normal =  np.cross(sol_direction, meridional_normal)
    sagittal_normal /= np.linalg.norm(sagittal_normal)
    
    source_location += horiz_displace * sagittal_normal
    return source_location    

def plot_rows(file_name, row, bin):
    import matplotlib.pyplot as plt
    
    image_array = np.array(Image.open(file_name))
    
    min = int(row - bin // 2)
    max = int(row + bin // 2)
    
    row_array = np.array(image_array[min],dtype=np.int32)
    
    for i in range(min + 1, max +1, 1):
        row_array += np.array(image_array[i], dtype=np.int32)
        
    plt.plot(row_array, 'k')
    plt.xlabel('Horizontal Pixels')
    plt.ylabel('Pixel Counts')
    plt.title('Line Intensity (row ' + str(row) + ', bin ' +str(bin) + ')')
    #plt.show()
    return

def plot_rows_data(file_name, row, bint, color):
    import matplotlib.pyplot as plt
    
    image_array = np.array(Image.open(file_name))
    
    mint = int(row - bint // 2)
    maxt = int(row + bint // 2)
    
    row_array = (np.array(image_array[mint].T[0],dtype= np.int32) +
                 np.array(image_array[mint].T[1],dtype= np.int32))

    for i in range(mint + 1, maxt +1, 1):
        sample_row0 = np.array(image_array[i].T[0], dtype=np.int32)
        sample_row1 = np.array(image_array[i].T[1], dtype=np.int32)
        row_array = row_array + sample_row0 + sample_row1
        
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
    
    for i in range(min + 1, max +1, 1):
        row_array += np.array(image_array[i], dtype=np.int32)        
    
    return row_array

def get_rows_data(file_name, row, bin):
    image_array = np.array(Image.open(file_name))
    image_array = image_array.T
    print(len(image_array[260]))
    min = int(row - bin // 2)
    max = int(row + bin // 2)
    
    row_array = np.array(image_array[min],dtype=np.int32)
    
    for i in range(min + 1, max +1, 1):
        row_array += np.array(image_array[i], dtype=np.int32)        
    
    return row_array

def image_height(file_name):
    # returns percentage of vertical illumination
    image_array = np.array(Image.open(file_name))
    test = np.where(image_array > 0)
    length = len(image_array)
    percent = (length - 2 * test[0][0]) / length
    return round(percent, 3) * 100

def tif_to_gray_scale(file_name, scale):    
    image = Image.open(file_name).convert("L")
    arr = np.asarray(image) * scale
    plt.figure()
    plt.imshow(arr, cmap='gray_r')
    plt.show()
    return
