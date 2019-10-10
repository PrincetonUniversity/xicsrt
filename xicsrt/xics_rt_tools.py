# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:30:38 2017

@author: James
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq

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

def source_location(distance
                    ,vert_displace
                    ,crystal_location
                    ,crystal_normal
                    ,crystal_curvature
                    ,crystal_spacing
                    ,detector_location
                    ,wavelength):
    """
    Returns the source location that satisfies the bragg condition met by the
    detector. Allows for a vertical displacement above and below the 
    meridional plane
    """
    
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

    bragg_c = bragg_angle(wavelength, crystal_spacing)
    norm_angle = np.pi/2.0 - bragg_c
    crystal_center = crystal_location + crystal_curvature * crystal_normal
    
    meridional_normal = np.cross(-crystal_normal, detector_location - crystal_center)
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

# s = source / g = graphite / c = crystal / d = detector

def setup_beam_scenario(c_spacing ,g_spacing ,
                        distance_s_g ,distance_g_c, distance_c_d,
                        wavelength, backwards_raytrace,
                        g_offset, g_tilt,
                        c_offset, c_tilt,
                        d_offset, d_tilt):
    ## An idealized scenario with a source, an HOPG, a crystal, and a detector
    
    bragg_g = bragg_angle(wavelength, g_spacing)
    bragg_c = bragg_angle(wavelength, c_spacing)
    
    ## Source Placement
    #souce is placed at origin by default and aimed along the X axis
    s_position  = np.array([0, 0, 0], dtype = np.float64)
    s_normal    = np.array([1, 0, 0], dtype = np.float64)
    s_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector = np.array([1, 0, 0], dtype = np.float64)
    path_vector/= np.linalg.norm(path_vector)
    
    ## Graphite Placement
    #define graphite position, normal, and basis relative to source
    g_position  = s_position + (path_vector * distance_s_g)
    g_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    g_y_vector  = np.cross(g_z_vector, path_vector)  
    g_y_vector /= np.linalg.norm(g_y_vector)
    
    g_z_vector  = np.cross(path_vector, g_y_vector)
    g_z_vector /= np.linalg.norm(g_z_vector)
    
    g_x_vector  = np.cross(g_y_vector, g_z_vector)
    g_x_vector /= np.linalg.norm(g_x_vector)
    
    #graphite normal is positioned to best reflect X-Rays according to Bragg
    g_normal    = (g_y_vector * np.cos(bragg_g)) - (g_x_vector * np.sin(bragg_g))
    g_normal   /= np.linalg.norm(g_normal)
    
    #reflect the path vector off of the graphite
    path_vector-= 2 * np.dot(path_vector, g_normal) * g_normal
    path_vector/= np.linalg.norm(path_vector)
    ## Crystal Placement
    #define crystal position, normal, and basis relative to graphite
    c_position  = g_position + (path_vector * distance_g_c)
    c_z_vector  = np.array([0, 0, 1], dtype = np.float64)

    c_y_vector  = np.cross(c_z_vector, path_vector)  
    c_y_vector /= np.linalg.norm(c_y_vector)
    
    c_z_vector  = np.cross(path_vector, c_y_vector)
    c_z_vector /= np.linalg.norm(c_z_vector)
    
    c_x_vector  = np.cross(c_y_vector, c_z_vector)
    c_x_vector /= np.linalg.norm(c_x_vector)
    
    #crystal normal is positioned to best reflect X-Rays according to Bragg
    c_normal    = (c_y_vector * np.cos(bragg_c)) - (c_x_vector * np.sin(bragg_c))
    c_normal   /= np.linalg.norm(c_normal)
    
    #reflect the path vector off of the crystal
    path_vector-= 2 * np.dot(path_vector, c_normal) * c_normal
    path_vector/= np.linalg.norm(path_vector)
    
    ## Detector Placement
    #define detector position, normal, and basis relative to crystal
    d_position  = c_position + (path_vector * distance_c_d)
    d_z_vector  = np.array([0, 1, 0], dtype = np.float64)
    
    d_y_vector  = np.cross(d_z_vector, path_vector)  
    d_y_vector /= np.linalg.norm(d_y_vector)
    
    d_z_vector  = np.cross(path_vector, d_y_vector)
    d_z_vector /= np.linalg.norm(d_z_vector)
    
    d_x_vector  = np.cross(d_y_vector, d_z_vector)
    d_x_vector /= np.linalg.norm(d_x_vector)
    
    #detector normal faces crystal
    d_normal    = -path_vector
    d_normal   /= np.linalg.norm(d_normal)
    
    """
    Optical elements' x_vectors, y_vectors, z_vectors form 3D bases
    Use the bases to create transformation matrices that use vector math to
    convert g_offset, c_offset, d_offset to XYZ offsets
    """
    ## Offset and Tilt Vector Math
    #create bases
    g_basis     = np.transpose(np.array([g_x_vector, g_y_vector, g_z_vector]))
    c_basis     = np.transpose(np.array([c_x_vector, c_y_vector, c_z_vector]))
    d_basis     = np.transpose(np.array([d_x_vector, d_y_vector, d_z_vector]))
    
    #offset using vector transformation matrix
    g_position += g_basis.dot(np.transpose(g_offset))
    c_position += c_basis.dot(np.transpose(c_offset))
    d_position += d_basis.dot(np.transpose(d_offset))
    
    #tilt using lots of vector rotations (WARNING! Non-commutative operations)
    g_normal    = vector_rotate(g_normal, g_x_vector, g_tilt[0])
    g_normal    = vector_rotate(g_normal, g_y_vector, g_tilt[1])
    g_normal    = vector_rotate(g_normal, g_z_vector, g_tilt[2])

    c_normal    = vector_rotate(c_normal, c_x_vector, c_tilt[0])
    c_normal    = vector_rotate(c_normal, c_y_vector, c_tilt[1])
    c_normal    = vector_rotate(c_normal, c_z_vector, c_tilt[2])
    
    d_normal    = vector_rotate(d_normal, d_x_vector, d_tilt[0])
    d_normal    = vector_rotate(d_normal, d_y_vector, d_tilt[1])
    d_normal    = vector_rotate(d_normal, d_z_vector, d_tilt[2])
    
    if   backwards_raytrace is False:
        s_target = g_position
    elif backwards_raytrace is True:
        s_target = c_position
    
    scenario_output = [s_position, s_normal, s_z_vector,
                       g_position, g_normal, g_z_vector,
                       c_position, c_normal, c_z_vector,
                       d_position, d_normal, d_z_vector,
                       s_target]
    return  scenario_output

def setup_crystal_test(c_spacing, distance_s_c, distance_c_d,
                       wavelength, backwards_raytrace,
                       c_offset, c_tilt):
    """
    An idealized scenario involving a source, crystal, and detector
    Designed to probe the crystal's properties and check for bugs
    """
    bragg_c = bragg_angle(wavelength, c_spacing)
    
    s_position      = np.array([0, 0, 0], dtype = np.float64)
    s_normal        = np.array([1, 0, 0], dtype = np.float64)
    s_orientation   = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector     = np.array([1, 0, 0], dtype = np.float64)
    
    #define crystal position, normal, and basis relative to graphite
    c_position  = s_position + (path_vector * distance_s_c)
    c_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    c_y_vector  = np.cross(c_z_vector, path_vector)  
    c_y_vector /= np.linalg.norm(c_y_vector)
    
    c_z_vector  = np.cross(path_vector, c_y_vector)
    c_z_vector /= np.linalg.norm(c_z_vector)
    
    c_x_vector  = np.cross(c_y_vector, c_z_vector)
    c_x_vector /= np.linalg.norm(c_x_vector)
    
    #crystal normal is positioned to best reflect X-Rays according to Bragg
    c_normal    = (c_y_vector * np.cos(bragg_c)) - (c_x_vector * np.sin(bragg_c))
    c_normal   /= np.linalg.norm(c_normal)
    
    #for focused extended sources, target them towards the crystal position    
    #s_target = c_position
    
    #alternatively, target them towards where the graphite would be in the beam
    s_target = path_vector * 1
    
    #reflect the path vector off of the crystal
    path_vector    -= 2 * np.dot(path_vector, c_normal) * c_normal

    #define detector position, normal, and basis relative to crystal
    d_position  = c_position + (path_vector * distance_c_d)
    d_z_vector  = np.array([0, 1, 0], dtype = np.float64)
    
    d_y_vector  = np.cross(d_z_vector, path_vector)  
    d_y_vector /= np.linalg.norm(d_y_vector)
    
    d_z_vector  = np.cross(path_vector, d_y_vector)
    d_z_vector /= np.linalg.norm(d_z_vector)
    
    d_x_vector  = np.cross(d_y_vector, d_z_vector)
    d_x_vector /= np.linalg.norm(d_x_vector)
    
    #detector normal faces crystal
    d_normal    = -path_vector
    d_normal   /= np.linalg.norm(d_normal)
    
    #offset vector math
    c_basis     = np.transpose(np.array([c_x_vector, c_y_vector, c_z_vector]))
    c_position += c_basis.dot(np.transpose(c_offset))
    c_z_vector    = vector_rotate(c_z_vector, c_normal, c_tilt)
    
    scenario_output = [s_position, s_normal, s_orientation,
                       c_position, c_normal, c_z_vector,
                       d_position, d_normal, d_z_vector,
                       s_target]
    return  scenario_output

def setup_graphite_test(g_spacing, distance_s_g, distance_g_d, wavelength):
    """
    An idealized scenario involving a source, HOPG, and detector
    Designed to probe the crystal's properties and check for bugs
    """
    bragg_g = bragg_angle(wavelength, g_spacing)
    
    s_position      = np.array([0, 0, 0], dtype = np.float64)
    s_normal        = np.array([1, 0, 0], dtype = np.float64)
    s_orientation   = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector     = np.array([1, 0, 0], dtype = np.float64)
    
    #define crystal position and normal relative to source
    g_position      = s_position + (path_vector * distance_s_g)
    g_z_vector      = np.array([0, 0, 1], dtype = np.float64)
    g_y_vector      = np.cross(g_z_vector, path_vector)  
    g_normal        = (g_y_vector * np.cos(bragg_g)) - (path_vector * np.sin(bragg_g))
    
    #for focused extended sources, target them towards the crystal position    
    s_target = g_position
    
    #reflect the path vector off of the crystal
    path_vector    -= 2 * np.dot(path_vector, g_normal) * g_normal

    #define detector position and normal relative to crystal
    d_position      = g_position + (path_vector * distance_g_d)
    d_z_vector      = np.array([0, 1, 0], dtype = np.float64)
    d_normal        = -path_vector
    
    scenario_output = [s_position, s_normal, s_orientation,
                       g_position, g_normal, g_z_vector,
                       d_position, d_normal, d_z_vector,
                       s_target]
    return  scenario_output

def setup_source_test(distance_s_d):
    """
    A source and a detector, nothing else. Useful for debugging sources
    """
    s_position      = np.array([0, 0, 0], dtype = np.float64)
    s_normal        = np.array([1, 0, 0], dtype = np.float64)
    s_z_vector      = np.array([0, 0, 1], dtype = np.float64)
        
    d_position      = s_position + (s_normal * distance_s_d)
    d_z_vector      = np.array([0, 0, 1], dtype = np.float64)
    d_normal        = -s_normal
    
    s_target = d_position
    
    scenario_output = [s_position, s_normal, s_z_vector,
                       d_position, d_normal, d_z_vector,
                       s_target]
    return  scenario_output

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
