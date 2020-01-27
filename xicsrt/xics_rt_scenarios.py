# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:30:38 2017

@author: James
"""
from xicsrt.xics_rt_math import bragg_angle, rotation_matrix, vector_rotate
import numpy as np

def source_location(distance, vert_displace, config):
    """
    Returns the source location that satisfies the bragg condition met by the
    detector. Allows for a vertical displacement above and below the 
    meridional plane
    """
    crystal_location    = config['crystal_input']['position']
    crystal_normal      = config['crystal_input']['normal']
    crystal_curvature   = config['crystal_input']['curvature']
    crystal_spacing     = config['crystal_input']['spacing']
    detector_location   = config['detector_input']['position']
    wavelength          = config['source_input']['wavelength']
    
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


def source_location_bragg(config, distance ,vert_displace ,horiz_displace):
    """
    Returns the source on the meridional plane that meets the Bragg condition
    for the given wavelength. Allows for a vertical displacement above and 
    below the meridional plane.
    """

    crystal_location    = config['crystal_input']['position']
    crystal_normal      = config['crystal_input']['normal']
    crystal_curvature   = config['crystal_input']['curvature']
    crystal_bragg       = config['crystal_input']['bragg']
    detector_location   = config['detector_input']['position']

    norm_angle = np.pi/2.0 - crystal_bragg
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
    
    config['source_input']['position'] = source_location
    return config

def setup_real_scenario(config):
    """
    Rather than generating the entire scenario from scratch, this scenario
    generator takes the information provided by the ITER XICS team and fills in
    all the blanks to produce a complete description of the spectrometer layout
    """
    
    ## Unpack variables and convert to meters
    chord  = config['scenario_input']['chord']
    g_corners  = config['scenario_input']['graphite_corners'][chord] / 1000
    c_corners  = config['scenario_input']['crystal_corners']         / 1000
    d_corners  = config['scenario_input']['detector_corners']        / 1000
    
    #calculate geometric properties of all meshes
    g_position = np.mean(g_corners, axis = 0)
    c_position = np.mean(c_corners, axis = 0)
    d_position = np.mean(d_corners, axis = 0)
    
    g_width    = np.linalg.norm(g_corners[0] - g_corners[3])
    c_width    = np.linalg.norm(c_corners[0] - c_corners[3])
    d_width    = np.linalg.norm(d_corners[0] - d_corners[3])
    
    g_height   = np.linalg.norm(g_corners[0] - g_corners[1])
    c_height   = np.linalg.norm(c_corners[0] - c_corners[1])
    d_height   = np.linalg.norm(d_corners[0] - d_corners[1])
    
    g_x_vector = g_corners[0] - g_corners[3]
    c_x_vector = c_corners[0] - c_corners[3]
    d_x_vector = d_corners[0] - d_corners[3]
    
    g_x_vector/= g_width
    c_x_vector/= c_width
    d_x_vector/= d_width
    
    g_y_vector = g_corners[0] - g_corners[1]
    c_y_vector = c_corners[0] - c_corners[1]
    d_y_vector = d_corners[0] - d_corners[1]
    
    g_y_vector/= g_height
    c_y_vector/= c_height
    d_y_vector/= d_height
    
    g_normal   = np.cross(g_x_vector, g_y_vector)
    c_normal   = np.cross(c_x_vector, c_y_vector)
    d_normal   = np.cross(d_x_vector, d_y_vector)
    
    g_normal  /= np.linalg.norm(g_normal)
    c_normal  /= np.linalg.norm(c_normal)
    d_normal  /= np.linalg.norm(d_normal)
    
    #calculate the graphite pre-reflector's sightline of the plasma
    #start with the crystal-graphite vector, normalize, and reflect it
    sightline  = g_position - c_position
    sightline /= np.linalg.norm(sightline)
    sightline -= 2 * np.dot(sightline, g_normal) * g_normal
    
    #triangulate the graphite
    config['graphite_input']['mesh_points'] = g_corners
    config['graphite_input']['mesh_faces']  = np.array([[2,1,0],[0,3,2]])
    
    ## Repack variables
    config['plasma_input']['target']          = g_position
    config['plasma_input']['sight_position']  = g_position
    config['plasma_input']['sight_direction'] = sightline
    config['plasma_input']['sight_thickness'] = 0.1

    config['graphite_input']['position']      = g_position
    config['graphite_input']['normal']        = g_normal
    config['graphite_input']['orientation']   = g_x_vector
    config['graphite_input']['width']         = g_width
    config['graphite_input']['height']        = g_height
    
    config['crystal_input']['position']       = c_position
    config['crystal_input']['normal']         = c_normal
    config['crystal_input']['orientation']    = c_x_vector
    config['crystal_input']['width']          = c_width
    config['crystal_input']['height']         = c_height
    
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_x_vector
    config['detector_input']['width']         = d_width
    config['detector_input']['height']        = d_height    
    
    return config
    
def setup_plasma_scenario(config):
    """
    An idealized scenario with a plasma source, an HOPG Pre-refelector,
    a crystal, and a detector.

    This is meant to model a basic XRCS spectrometer with a pre-refelctor,
    such as is planned for the ITER XRCS-Core spectrometer.
    """

    #unpack variables
    distance_p_g    = config['scenario_input']['source_graphite_dist']
    distance_g_c    = config['scenario_input']['graphite_crystal_dist']
    distance_c_d    = config['scenario_input']['crystal_detector_dist']

    bragg_c = bragg_angle(config['source_input']['wavelength'], config['crystal_input']['spacing'])
    bragg_g = bragg_angle(config['source_input']['wavelength'], config['graphite_input']['spacing'])

    meridi_focus = config['crystal_input']['curvature'] * np.sin(bragg_c)
    sagitt_focus = - meridi_focus / np.cos(2 * bragg_c)

    if distance_c_d is None:
        distance_c_d = meridi_focus

    ## Source Placement
    #souce is placed at origin by default and aimed along the X axis
    p_position  = np.array([0, 0, 0], dtype = np.float64)
    p_normal    = np.array([1, 0, 0], dtype = np.float64)
    p_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector = np.array([1, 0, 0], dtype = np.float64)
    path_vector/= np.linalg.norm(path_vector)
    
    ## Graphite Placement
    #define graphite position, normal, and basis relative to source
    g_position  = p_position + (path_vector * distance_p_g)
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

    if config['general_input']['backwards_raytrace']:
        p_target = c_position
    else:
        p_target = g_position
    
    ## Repack variables
    config['plasma_input']['position']        = p_position
    config['plasma_input']['normal']          = p_normal
    config['plasma_input']['orientation']     = p_z_vector
    config['plasma_input']['target']          = p_target

    config['graphite_input']['position']      = g_position
    config['graphite_input']['normal']        = g_normal
    config['graphite_input']['orientation']   = g_z_vector
    config['crystal_input']['position']       = c_position
    config['crystal_input']['normal']         = c_normal
    config['crystal_input']['orientation']    = c_z_vector
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_z_vector

    return config


def setup_throughput_scenario(config):
    """
    An idealized scenario with a plasma source, a crystal, and a detector.

    This is useful for comparing the ray throughput between a spectrometer with
    and without an HOPG pre-reflector. The plasma aims at where the graphite
    normally would be, but the rays continue on towards the crystal.
    """

    #unpack variables
    distance_p_g    = config['scenario_input']['source_graphite_dist']
    distance_g_c    = config['scenario_input']['graphite_crystal_dist']
    distance_c_d    = config['scenario_input']['crystal_detector_dist']

    bragg_c = bragg_angle(config['source_input']['wavelength'], config['crystal_input']['spacing'])

    meridi_focus = config['crystal_input']['curvature'] * np.sin(bragg_c)
    sagitt_focus = - meridi_focus / np.cos(2 * bragg_c)

    if distance_c_d is None:
        distance_c_d = meridi_focus

    ## Source Placement
    #souce is placed at origin by default and aimed along the X axis
    p_position  = np.array([0, 0, 0], dtype = np.float64)
    p_normal    = np.array([1, 0, 0], dtype = np.float64)
    p_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector = np.array([1, 0, 0], dtype = np.float64)
    path_vector/= np.linalg.norm(path_vector)

    ## Crystal Placement
    #define crystal position, normal, and basis relative to graphite
    p_target    = p_position + (path_vector *  distance_p_g)
    c_position  = p_position + (path_vector * (distance_p_g + distance_g_c))
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

    
    ## Repack variables
    config['plasma_input']['position']        = p_position
    config['plasma_input']['normal']          = p_normal
    config['plasma_input']['orientation']     = p_z_vector
    config['plasma_input']['target']          = p_target
    
    config['graphite_input']['position']      = c_position
    config['graphite_input']['normal']        = c_normal
    config['graphite_input']['orientation']   = c_z_vector
    config['crystal_input']['position']       = c_position
    config['crystal_input']['normal']         = c_normal
    config['crystal_input']['orientation']    = c_z_vector
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_z_vector

    return config


def setup_beam_scenario(config):
    """
    An idealized scenario with a source, an HOPG Pre-refelector, a crystal,
    and a detector.

    This is meant to model a basic XRCS spectrometer with a pre-refelctor,
    such as is planned for the ITER XRCS-Core spectrometer.
    """

    #unpack variables
    distance_s_g    = config['scenario_input']['source_graphite_dist']
    distance_g_c    = config['scenario_input']['graphite_crystal_dist']
    distance_c_d    = config['scenario_input']['crystal_detector_dist']

    bragg_c = bragg_angle(config['source_input']['wavelength'], config['crystal_input']['spacing'])
    bragg_g = bragg_angle(config['source_input']['wavelength'], config['graphite_input']['spacing'])

    meridi_focus = config['crystal_input']['curvature'] * np.sin(bragg_c)
    sagitt_focus = - meridi_focus / np.cos(2 * bragg_c)

    if distance_c_d is None:
        distance_c_d = meridi_focus

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

    if config['general_input']['backwards_raytrace']:
        s_target = c_position
    else:
        s_target = g_position
    
    ## Repack variables
    config['source_input']['position']        = s_position
    config['source_input']['normal']          = s_normal
    config['source_input']['orientation']     = s_z_vector
    config['source_input']['target']          = s_target

    config['graphite_input']['position']      = g_position
    config['graphite_input']['normal']        = g_normal
    config['graphite_input']['orientation']   = g_z_vector
    config['crystal_input']['position']       = c_position
    config['crystal_input']['normal']         = c_normal
    config['crystal_input']['orientation']    = c_z_vector
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_z_vector

    return config


def setup_crystal_test(config):
    """
    An idealized scenario with a source, a crystal, and a detector

    This is meant to be used in conjunction with the beam scenario.
    Using the same configuration file this will generate a scenario where
    the crystal is at the position where the pre-reflector used to be.
    """

    distance_s_c    = config['scenario_input']['source_graphite_dist']
    distance_c_d    = config['scenario_input']['crystal_detector_dist']
    
    bragg_c = bragg_angle(config['source_input']['wavelength'], config['crystal_input']['spacing'])

    meridi_focus = config['crystal_input']['curvature'] * np.sin(bragg_c)
    sagitt_focus = - meridi_focus / np.cos(2 * bragg_c)

    if distance_c_d is None:
        distance_c_d = meridi_focus

    ## Source Placement
    #souce is placed at origin by default and aimed along the X axis    
    s_position      = np.array([0, 0, 0], dtype = np.float64)
    s_normal        = np.array([1, 0, 0], dtype = np.float64)
    s_z_vector      = np.array([0, 0, 1], dtype = np.float64)
    
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
    s_target = c_position
    
    #alternatively, target them towards where the graphite would be in the beam
    #s_target = path_vector * 1
    
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
    
    # repack variables
    config['source_input']['position']        = s_position
    config['source_input']['normal']          = s_normal
    config['source_input']['orientation']     = s_z_vector
    config['graphite_input']['position']      = c_position
    config['graphite_input']['normal']        = c_normal
    config['graphite_input']['orientation']   = c_z_vector
    config['crystal_input']['position']       = c_position
    config['crystal_input']['normal']         = c_normal
    config['crystal_input']['orientation']    = c_z_vector
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_z_vector
    config['source_input']['target']          = s_target

    return config


def setup_graphite_test(config):
    """
    An idealized scenario with a source, a crystal, and a detector

    This is meant to be used in conjunction with the beam scenario.
    Using the same configuration file this will generate a scenario where
    the crystal is removed and the detector is placed at the
    crystal-detector distance.
    """
    distance_s_g    = config['scenario_input']['source_graphite_dist']
    distance_g_d    = config['scenario_input']['crystal_detector_dist']

    bragg_g = bragg_angle(config['source_input']['wavelength'], config['graphite_input']['spacing'])

    s_position  = np.array([0, 0, 0], dtype = np.float64)
    s_normal    = np.array([1, 0, 0], dtype = np.float64)
    s_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector= np.array([1, 0, 0], dtype = np.float64)
    
    #define graphite position and normal relative to source
    g_position  = s_position + (path_vector * distance_s_g)
    g_z_vector  = np.array([0, 0, 1], dtype = np.float64)
    
    g_y_vector  = np.cross(g_z_vector, path_vector)
    g_y_vector /= np.linalg.norm(g_y_vector)
    
    g_normal    = (g_y_vector * np.cos(bragg_g)) - (path_vector * np.sin(bragg_g))
    g_normal   /= np.linalg.norm(g_normal)
    
    #for focused extended sources, target them towards the graphite position    
    s_target = g_position
    
    #reflect the path vector off of the graohite
    path_vector    -= 2 * np.dot(path_vector, g_normal) * g_normal

    #define detector position and normal relative to graphite
    d_position  = g_position + (path_vector * distance_g_d)
    d_z_vector  = np.array([0, 1, 0], dtype = np.float64)
    d_normal    = -path_vector
    
    # repack variables
    config['source_input']['position']        = s_position
    config['source_input']['normal']          = s_normal
    config['source_input']['orientation']     = s_z_vector
    config['graphite_input']['position']      = g_position
    config['graphite_input']['normal']        = g_normal
    config['graphite_input']['orientation']   = g_z_vector
    config['crystal_input']['position']       = g_position
    config['crystal_input']['normal']         = g_normal
    config['crystal_input']['orientation']    = g_z_vector
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_z_vector
    config['source_input']['target']          = s_target

    return config


def setup_source_test(config):
    """
    A source and a detector, nothing else. Useful for debugging sources

    This is meant to be used in conjunction with the beam scenario.
    Using the same configuration file this will generate a scenario with
    only a source and a detector. The detector will be placed at the
    source-graphite distance.
    """
    distance_s_d    = config['scenario_input']['source_graphite_dist']
    
    s_position      = np.array([0, 0, 0], dtype = np.float64)
    s_normal        = np.array([1, 0, 0], dtype = np.float64)
    s_z_vector      = np.array([0, 0, 1], dtype = np.float64)
        
    d_position      = s_position + (s_normal * distance_s_d)
    d_z_vector      = np.array([0, 0, 1], dtype = np.float64)
    d_normal        = -s_normal
    
    s_target = d_position
    
    #repack vectors
    config['source_input']['position']        = s_position
    config['source_input']['normal']          = s_normal
    config['source_input']['orientation']     = s_z_vector
    config['detector_input']['position']      = d_position
    config['detector_input']['normal']        = d_normal
    config['detector_input']['orientation']   = d_z_vector
    config['source_input']['target']          = s_target

    return config
