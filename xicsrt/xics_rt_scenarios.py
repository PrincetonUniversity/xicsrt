# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:30:38 2017

@author: James
"""
from xicsrt.xics_rt_math import bragg_angle, vector_rotate, rotation_matrix
import numpy as np

def source_location(distance, vert_displace, config):
    """
    Returns the source location that satisfies the bragg condition met by the
    detector. Allows for a vertical displacement above and below the 
    meridional plane
    """
    crystal_location = config['crystal_input']['position']
    crystal_normal = config['crystal_input']['normal']
    crystal_curvature = config['crystal_input']['curvature']
    crystal_spacing = config['crystal_input']['spacing']
    detector_location = config['detector_input']['position']
    wavelength = config['source_input']['wavelength']
    
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

    crystal_location = config['crystal_input']['position']
    crystal_normal = config['crystal_input']['normal']
    crystal_curvature = config['crystal_input']['curvature']
    crystal_bragg = config['crystal_input']['bragg']
    detector_location = config['detector_input']['position']

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

    g_offset        = config['scenario_input']['graphite_offset']
    c_offset        = config['scenario_input']['crystal_offset']
    d_offset        = config['scenario_input']['detector_offset']
    g_tilt          = config['scenario_input']['graphite_tilt']
    c_tilt          = config['scenario_input']['crystal_tilt']
    d_tilt          = config['scenario_input']['detector_tilt']

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

    g_offset        = config['scenario_input']['graphite_offset']
    c_offset        = config['scenario_input']['crystal_offset']
    d_offset        = config['scenario_input']['detector_offset']
    g_tilt          = config['scenario_input']['graphite_tilt']
    c_tilt          = config['scenario_input']['crystal_tilt']
    d_tilt          = config['scenario_input']['detector_tilt']

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
    
    c_offset        = config['scenario_input']['crystal_offset']
    c_tilt          = config['scenario_input']['crystal_tilt']

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
    
    #offset vector math
    c_basis     = np.transpose(np.array([c_x_vector, c_y_vector, c_z_vector]))
    c_position += c_basis.dot(np.transpose(c_offset))
    c_z_vector  = vector_rotate(c_z_vector, c_normal, c_tilt)
    
    # repack variables
    config['source_input']['position']        = s_position
    config['source_input']['normal']          = s_normal
    config['source_input']['orientation']     = s_z_vector
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
    
    g_offset        = config['scenario_input']['graphite_offset']
    g_tilt          = config['scenario_input']['graphite_tilt']

    bragg_g = bragg_angle(config['scenario_input']['wavelength'], config['graphite_input']['spacing'])

    s_position      = np.array([0, 0, 0], dtype = np.float64)
    s_normal        = np.array([1, 0, 0], dtype = np.float64)
    s_z_vector      = np.array([0, 0, 1], dtype = np.float64)
    
    #create a path vector that connects the centers of all optical elements
    path_vector     = np.array([1, 0, 0], dtype = np.float64)
    
    #define graphite position and normal relative to source
    g_position      = s_position + (path_vector * distance_s_g)
    g_z_vector      = np.array([0, 0, 1], dtype = np.float64)
    g_y_vector      = np.cross(g_z_vector, path_vector)  
    g_x_vector      = np.cross(g_y_vector, g_z_vector)
    g_normal        = (g_y_vector * np.cos(bragg_g)) - (path_vector * np.sin(bragg_g))
    
    #for focused extended sources, target them towards the graphite position    
    s_target = g_position
    
    #reflect the path vector off of the graohite
    path_vector    -= 2 * np.dot(path_vector, g_normal) * g_normal

    #define detector position and normal relative to graphite
    d_position      = g_position + (path_vector * distance_g_d)
    d_z_vector      = np.array([0, 1, 0], dtype = np.float64)
    d_normal        = -path_vector
    
    #offset vector math
    g_basis     = np.transpose(np.array([g_x_vector, g_y_vector, g_z_vector]))
    g_position += g_basis.dot(np.transpose(g_offset))
    g_z_vector  = vector_rotate(g_z_vector, g_normal, g_tilt)
    
    # repack variables
    config['source_input']['position']        = s_position
    config['source_input']['normal']          = s_normal
    config['source_input']['orientation']     = s_z_vector
    config['graphite_input']['position']      = g_position
    config['graphite_input']['normal']        = g_normal
    config['graphite_input']['orientation']   = g_z_vector
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
