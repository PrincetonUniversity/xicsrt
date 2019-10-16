# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import wofz
import matplotlib.pyplot as plt
from PIL import Image

def voigt(x, y):
    """
    The Voigt function is also the real part of  w(z) = exp(-z^2) erfc(iz), 
    the complex probability function, which is also known as the Faddeeva 
    function. Scipy has implemented this function under the name wofz()
    """
    z = x + 1j*y
    I = wofz(z).real
    return I


def voigt_physical(intensity, location, sigma, gamma):
    """
    The voigt function in physical parameters.
    """
    u = (x - location) / np.sqrt(2) / sigma
    a = gamma / np.sqrt(2) / sigma
    y = voigt(a, u) / np.sqrt(2 * np.pi) / sigma * intensity
    return y

def bragg_angle(wavelength, crystal_spacing):
    """
    The Bragg angle calculation is used so often that it deserves its own funct
    """
    bragg_angle = np.arcsin(wavelength / (2 * crystal_spacing))
    return bragg_angle

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
    
def vector_rotate(a, b, theta):
    ## Rotate vector a around vector b by an angle theta (radians)
    #project a onto b, return parallel and perpendicular component vectors
    proj_para = b * np.dot(a, b) / np.dot(b, b)
    proj_perp = a - proj_para
    
    #define and normalize the unit vector w, perpendicular to a and b
    w  = np.cross(b, proj_perp)
    if (np.linalg.norm(w) != 0): 
        w /= np.linalg.norm(w)
    
    #return the final rotated vector c
    c = proj_para + (proj_perp * np.cos(theta)) + (
            np.linalg.norm(proj_perp) * w * np.sin(theta))
    return c

def plot_rows(file_name, row, bin):
    
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
