# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:30:38 2017

@author: James
"""
from xicsrt.xicsrt_math import bragg_angle, rotation_matrix, vector_rotate
import numpy as np

def create_source_basis(distance):
    """
    sets up a source to begin each scenario
    requires a distance between the source and the first optic
    """
    origin  = np.array([0, 0, 0], dtype = np.float64)
    basis     = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype = np.float64)
    path_vector = np.array([1, 0, 0], dtype = np.float64)
    target    = origin + (path_vector * distance)
    
    return origin, basis, path_vector, target

def create_optic_basis(path_vector, orientation, bragg, width, height):
    """
    accepts two normalized vectors and a bragg angle, creates an optic basis
    path_vector is the incoming rays vector
    orientation is an arbitrary user-defined vector
    bragg is the optic's bragg angle
    """
    #Create a temporary scenario basis
    temp_basis = np.zeros([3,3], dtype = np.float64)
    
    temp_basis[2,:]  = orientation
    temp_basis[2,:] /= np.linalg.norm(temp_basis[2,:])
    
    temp_basis[0,:]  = path_vector
    temp_basis[0,:] /= np.linalg.norm(temp_basis[0,:])
    
    temp_basis[1,:]  = np.cross(temp_basis[2,:], temp_basis[0,:])
    temp_basis[1,:] /= np.linalg.norm(temp_basis[1,:])
    
    #Derive the normal vector from the temporary basis
    normal = temp_basis[1,:] * np.cos(bragg) - temp_basis[0,:] * np.sin(bragg)
    
    #Reflect the path vector
    path_vector -= 2 * np.dot(path_vector, normal) * normal
    path_vector /= np.linalg.norm(path_vector)
    
    #Create the optic basis
    optic_basis = np.zeros([3,3], dtype = np.float64)
    
    optic_basis[2,:]  = normal
    optic_basis[2,:] /= np.linalg.norm(optic_basis[2,:])
    
    optic_basis[0,:]  = orientation
    optic_basis[0,:] /= np.linalg.norm(optic_basis[0,:])
    
    optic_basis[1,:]  = np.cross(optic_basis[2,:], optic_basis[0,:])
    optic_basis[1,:] /= np.linalg.norm(optic_basis[1,:])
    
    #Use the basis to create a square mesh at the origin
    dx = width  * optic_basis[0,:] / 2
    dy = height * optic_basis[1,:] / 2
    
    p1 = + dx + dy
    p2 = - dx + dy
    p3 = - dx - dy
    p4 = + dx - dy
    
    mesh_points = np.array([p1, p2, p3, p4])
    mesh_faces  = np.array([[2,1,0],[0,3,2]])
    
    return path_vector, optic_basis, mesh_points, mesh_faces

def create_detector_basis(path_vector, orientation):
    """
    accepts two normalized vectors, creates a detector basis
    path_vector is the incoming rays vector
    orientation is an arbitrary user-defined vector
    """    
    normal = -path_vector
    
    #Create the detector basis
    optic_basis = np.zeros([3,3], dtype = np.float64)
    
    optic_basis[2,:]  = normal
    optic_basis[2,:] /= np.linalg.norm(optic_basis[2,:])
    
    optic_basis[0,:]  = orientation
    optic_basis[0,:] /= np.linalg.norm(optic_basis[0,:])
    
    optic_basis[1,:]  = np.cross(optic_basis[2,:], optic_basis[0,:])
    optic_basis[1,:] /= np.linalg.norm(optic_basis[1,:])
    
    return optic_basis

def source_location(distance, vert_displace, config):
    """
    Returns the source location that satisfies the bragg condition met by the
    detector. Allows for a vertical displacement above and below the 
    meridional plane
    """
    crystal_location    = config['crystal_input']['origin']
    crystal_normal      = config['crystal_input']['zaxis']
    crystal_curvature   = config['crystal_input']['radius']
    crystal_spacing     = config['crystal_input']['crystal_spacing']
    detector_location   = config['detector_input']['origin']
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

    crystal_location    = config['crystal_input']['origin']
    crystal_normal      = config['crystal_input']['zaxis']
    crystal_curvature   = config['crystal_input']['radius']
    crystal_bragg       = config['crystal_input']['bragg']
    detector_location   = config['detector_input']['origin']

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
    
    config['source_input']['origin'] = source_location
    return config

def setup_manfred_scenario(config):
    """
    Experimental sinusoidal spiral crystal designed by Dr. Manfred.
    A Multi-toroidal mirror designed for x-rays with energies 9.750-10.560 keV
    intended for Ge[400] crystal with inter-atomic distance 2d = 2.82868 A
    """
    from xicsrt.xics_rt_meshes import generate_sinusoidal_spiral
    
    manfred_input = {}
    manfred_input['horz_resolution'] = 11
    manfred_input['vert_resolution'] = 31
    manfred_input['base_radius']     = 0.300
    manfred_input['base_height']     = 0.01
    manfred_input['spiral_parameter']= 0.34
    manfred_input['crystal_spacing'] = 2.82868 / 2
    
    manfred_output = generate_sinusoidal_spiral(manfred_input)
    
    mesh_points     = manfred_output['mesh_points']
    mesh_faces      = manfred_output['mesh_faces']
    mesh_normals    = manfred_output['mesh_normals']
    detector_points = manfred_output['detector_points']
    c_width         = manfred_output['c_length']
    d_width         = manfred_output['d_length']
    
    #setup scenario
    s_origin = np.array([0.0, 0.0, 0.0])
    c_origin = np.mean(mesh_points, axis = 0)
    d_origin = np.mean(detector_points, axis = 0)
    
    s_x_vector = np.array([0.0, 0.0, 1.0])
    c_x_vector = np.array([0.0, 0.0, 1.0])
    d_x_vector = np.array([0.0, 0.0, 1.0])
    
    s_normal   = c_origin / np.linalg.norm(c_origin)
    c_normal   = np.mean(mesh_normals, axis = 0)
    d_normal   = np.cross(detector_points[-1,:] - detector_points[0,:], d_x_vector)
    s_target   = c_origin
    
    #repack variables
    config['crystal_input']['mesh_points']    = mesh_points
    config['crystal_input']['mesh_faces']     = mesh_faces
    config['source_input']['origin']          = s_origin
    config['source_input']['zaxis']           = s_normal
    config['source_input']['xaxis']           = s_x_vector
    config['source_input']['target']          = s_target
    config['source_input']['wavelength']      = 10.25 / 12.398425
    config['crystal_input']['origin']         = c_origin
    config['crystal_input']['zaxis']          = c_normal
    config['crystal_input']['xaxis']          = c_x_vector
    config['crystal_input']['height']         = c_width
    config['crystal_input']['width']          = manfred_input['base_height']
    config['crystal_input']['crystal_spacing']= manfred_input['crystal_spacing']
    config['detector_input']['origin']        = d_origin
    config['detector_input']['zaxis']         = d_normal
    config['detector_input']['xaxis']         = d_x_vector
    config['detector_input']['height']        = d_width
    config['detector_input']['vertical_pixels'] = int(round(d_width / config['detector_input']['pixel_size']))
    
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
    g_basis      = np.zeros([3,3], dtype = np.float64)
    c_basis      = np.zeros([3,3], dtype = np.float64)
    d_basis      = np.zeros([3,3], dtype = np.float64)
    
    g_origin     = np.mean(g_corners, axis = 0)
    c_origin     = np.mean(c_corners, axis = 0)
    d_origin     = np.mean(d_corners, axis = 0)
    
    g_width      = np.linalg.norm(g_corners[0] - g_corners[3])
    c_width      = np.linalg.norm(c_corners[0] - c_corners[3])
    d_width      = np.linalg.norm(d_corners[0] - d_corners[3])
    
    g_height     = np.linalg.norm(g_corners[0] - g_corners[1])
    c_height     = np.linalg.norm(c_corners[0] - c_corners[1])
    d_height     = np.linalg.norm(d_corners[0] - d_corners[1])
    
    g_basis[0,:] = g_corners[0] - g_corners[3]
    c_basis[0,:] = c_corners[0] - c_corners[3]
    d_basis[0,:] = d_corners[0] - d_corners[3]
    
    g_basis[0,:]/= g_width
    c_basis[0,:]/= c_width
    d_basis[0,:]/= d_width
    
    g_basis[1,:] = g_corners[0] - g_corners[1]
    c_basis[1,:] = c_corners[0] - c_corners[1]
    d_basis[1,:] = d_corners[0] - d_corners[1]
    
    g_basis[1,:]/= g_height
    c_basis[1,:]/= c_height
    d_basis[1,:]/= d_height
    
    g_basis[2,:] = np.cross(g_basis[0,:], g_basis[1,:])
    c_basis[2,:] = np.cross(c_basis[0,:], c_basis[1,:])
    d_basis[2,:] = np.cross(d_basis[0,:], d_basis[1,:])
    
    g_basis[2,:]/= np.linalg.norm(g_basis[2,:])
    c_basis[2,:]/= np.linalg.norm(c_basis[2,:])
    d_basis[2,:]/= np.linalg.norm(d_basis[2,:])
    
    #calculate the graphite pre-reflector's sightline of the plasma
    #start with the crystal-graphite vector, normalize, and reflect it
    sightline    = g_origin - c_origin
    sightline   /= np.linalg.norm(sightline)
    sightline   -= 2 * np.dot(sightline, g_basis[2,:]) * g_basis[2,:]
    
    #triangulate the graphite
    config['graphite_input']['mesh_points'] = g_corners
    config['graphite_input']['mesh_faces']  = np.array([[2,1,0],[0,3,2]])
    
    ## Repack variables
    config['plasma_input']['target']          = g_origin
    #config['plasma_input']['sight_origin']    = g_origin
    #config['plasma_input']['sight_direction'] = sightline
    #graphite dimensions override - remove these hashtags later
    config['graphite_input']['origin']        = g_origin
    config['graphite_input']['zaxis']         = g_basis[2,:]
    config['graphite_input']['xaxis']         = g_basis[0,:]
    #config['graphite_input']['width']        = g_width
    #config['graphite_input']['height']       = g_height
    
    config['crystal_input']['origin']         = c_origin
    config['crystal_input']['zaxis']          = c_basis[2,:]
    config['crystal_input']['xaxis']          = c_basis[0,:]
    config['crystal_input']['width']          = c_width
    config['crystal_input']['height']         = c_height
        
    config['detector_input']['origin']        = d_origin
    config['detector_input']['zaxis']         = d_basis[2,:]
    config['detector_input']['xaxis']         = d_basis[0,:]
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
    major_radius    = config['plasma_input']['major_radius']

    c_bragg = bragg_angle(config['source_input']['wavelength'], 
                          config['crystal_input']['crystal_spacing'])
    g_bragg = bragg_angle(config['source_input']['wavelength'], 
                          config['graphite_input']['crystal_spacing'])
    
    c_width = config['graphite_input']['width']
    c_height= config['graphite_input']['height']
    
    g_width = config['graphite_input']['width']
    g_height= config['graphite_input']['height']

    meridi_focus = config['crystal_input']['radius'] * np.sin(c_bragg)
    sagitt_focus = - meridi_focus / np.cos(2 * c_bragg)

    if distance_c_d is None:
        distance_c_d = meridi_focus

    ## Plasma Placement
    p_origin  = np.array([0, major_radius, 0], dtype = np.float64)
    p_basis     = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype = np.float64)
    path_vector = np.array([0, 1, 0], dtype = np.float64)
    p_target    = p_origin + (path_vector * distance_p_g)
    
    ## Graphite Placement
    g_origin  = p_origin + (path_vector * distance_p_g)
    g_orient    = np.array([0, 0, 1], dtype = np.float64)
    path_vector, g_basis, g_mesh_points, g_mesh_faces = create_optic_basis(
        path_vector, g_orient, g_bragg, g_width, g_height)

    ## Crystal Placement
    c_origin  = g_origin + (path_vector * distance_g_c)
    c_orient  = np.array([0, 0, 1], dtype = np.float64)
    path_vector, c_basis, c_mesh_points, c_mesh_faces = create_optic_basis(
        path_vector, c_orient, c_bragg, c_width, c_height)
    
    ## Detector Placement
    d_origin = c_origin + (path_vector * distance_c_d)
    d_orient   = np.array([0, 0, 1], dtype = np.float64)
    d_basis    = create_detector_basis(path_vector, d_orient)

    if config['general_input']['backwards_raytrace']:
        p_target = c_origin
    else:
        p_target = g_origin
    
    #define meshes
    config['crystal_input']['mesh_points']  = c_mesh_points + c_origin
    config['crystal_input']['mesh_faces']   = c_mesh_faces    
    config['graphite_input']['mesh_points'] = g_mesh_points + g_origin
    config['graphite_input']['mesh_faces']  = g_mesh_faces
    
    ## Repack variables
    config['plasma_input']['origin']        = p_origin
    config['plasma_input']['zaxis']         = p_basis[2,:]
    config['plasma_input']['xaxis']         = p_basis[0,:]
    config['graphite_input']['origin']      = g_origin
    config['graphite_input']['zaxis']       = g_basis[2,:]
    config['graphite_input']['xaxis']       = g_basis[0,:]
    config['crystal_input']['origin']       = c_origin
    config['crystal_input']['zaxis']        = c_basis[2,:]
    config['crystal_input']['xaxis']        = c_basis[0,:]
    config['detector_input']['origin']      = d_origin
    config['detector_input']['zaxis']       = d_basis[2,:]
    config['detector_input']['xaxis']       = d_basis[0,:]
    config['source_input']['target']        = p_target

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

    c_bragg = bragg_angle(config['source_input']['wavelength'], config['crystal_input']['crystal_spacing'])
    c_width = config['crystal_input']['width']
    c_height= config['crystal_input']['height']

    meridi_focus = config['crystal_input']['radius'] * np.sin(c_bragg)
    sagitt_focus = - meridi_focus / np.cos(2 * c_bragg)

    if distance_c_d is None:
        distance_c_d = meridi_focus

    ## Plasma Placement
    p_origin, p_basis, path_vector, p_target = create_source_basis(distance_p_g)

    ## Crystal Placement
    c_origin  = p_origin + (path_vector * (distance_p_g + distance_g_c))
    c_orient  = np.array([0, 0, 1], dtype = np.float64)
    path_vector, c_basis, mesh_points, mesh_faces = create_optic_basis(
        path_vector, c_orient, c_bragg, c_width, c_height)
    
    ## Detector Placement
    d_origin = c_origin + (path_vector * distance_c_d)
    d_orient   = np.array([0, 1, 0], dtype = np.float64)
    d_basis    = create_detector_basis(path_vector, d_orient)
    
    #define meshes
    config['crystal_input']['mesh_points']  = mesh_points + c_origin
    config['crystal_input']['mesh_faces']   = mesh_faces    
    config['graphite_input']['mesh_points'] = mesh_points + c_origin
    config['graphite_input']['mesh_faces']  = mesh_faces
    
    ## Repack variables
    config['plasma_input']['origin']        = p_origin
    config['plasma_input']['zaxis']         = p_basis[2,:]
    config['plasma_input']['xaxis']         = c_basis[0,:]
    config['plasma_input']['target']        = p_target
    config['graphite_input']['origin']      = c_origin
    config['graphite_input']['zaxis']       = c_basis[2,:]
    config['graphite_input']['xaxis']       = c_basis[0,:]
    config['crystal_input']['origin']       = c_origin
    config['crystal_input']['zaxis']        = c_basis[2,:]
    config['crystal_input']['xaxis']        = c_basis[0,:]
    config['detector_input']['origin']      = d_origin
    config['detector_input']['origin']      = d_origin
    config['detector_input']['zaxis']       = d_basis[2,:]
    config['detector_input']['xaxis']       = d_basis[0,:]

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
    
    c_bragg = bragg_angle(config['source_input']['wavelength'], 
                          config['crystal_input']['crystal_spacing'])
    g_bragg = bragg_angle(config['source_input']['wavelength'], 
                          config['graphite_input']['crystal_spacing'])
    
    c_width = config['crystal_input']['width']
    c_height= config['crystal_input']['height']
    
    g_width = config['graphite_input']['width']
    g_height= config['graphite_input']['height']
    
    meridi_focus = config['crystal_input']['radius'] * np.sin(c_bragg)
    sagitt_focus = - meridi_focus / np.cos(2 * c_bragg)

    if distance_c_d is None:
        distance_c_d = meridi_focus

    ## Source Placement
    s_origin, s_basis, path_vector, s_target = create_source_basis(distance_s_g)
    
    ## Graphite Placement
    g_origin  = s_origin + (path_vector * distance_s_g)
    g_orient  = np.array([0, 0, 1], dtype = np.float64)
    path_vector, g_basis, g_mesh_points, g_mesh_faces = create_optic_basis(
        path_vector, g_orient, g_bragg, g_width, g_height)

    ## Crystal Placement
    c_origin  = g_origin + (path_vector * distance_g_c)
    c_orient  = np.array([0, 0, 1], dtype = np.float64)
    path_vector, c_basis, c_mesh_points, c_mesh_faces = create_optic_basis(
        path_vector, c_orient, c_bragg, c_width, c_height)

    if config['general_input']['backwards_raytrace']:
        s_target = c_origin
    else:
        s_target = g_origin
    
    ## Detector Placement
    d_origin   = c_origin + (path_vector * distance_c_d)
    d_orient   = np.array([0, 0, 1], dtype = np.float64)
    d_basis    = create_detector_basis(path_vector, d_orient)
    
    #define meshes
    config['crystal_input']['mesh_points']  = c_mesh_points + c_origin
    config['crystal_input']['mesh_faces']   = c_mesh_faces    
    config['graphite_input']['mesh_points'] = g_mesh_points + g_origin
    config['graphite_input']['mesh_faces']  = g_mesh_faces
    
    ## Repack variables
    config['source_input']['origin']        = s_origin
    config['source_input']['zaxis']         = s_basis[2,:]
    config['source_input']['xaxis']         = s_basis[0,:]
    config['graphite_input']['origin']      = g_origin
    config['graphite_input']['zaxis']       = g_basis[2,:]
    config['graphite_input']['xaxis']       = g_basis[0,:]
    config['crystal_input']['origin']       = c_origin
    config['crystal_input']['zaxis']        = c_basis[2,:]
    config['crystal_input']['xaxis']        = c_basis[0,:]
    config['detector_input']['origin']      = d_origin
    config['detector_input']['zaxis']       = d_basis[2,:]
    config['detector_input']['xaxis']       = d_basis[0,:]
    config['source_input']['target']        = s_target

    return config


def setup_crystal_test(config):
    """
    An idealized scenario with a source, a crystal, and a detector

    This is meant to be used in conjunction with the beam scenario.
    Using the same configuration file this will generate a scenario where
    the crystal is at the position where the pre-reflector used to be.
    """

    distance_s_c = config['scenario_input']['source_graphite_dist']
    distance_c_d = config['scenario_input']['crystal_detector_dist']    
    c_bragg = bragg_angle(config['source_input']['wavelength'], 
                          config['crystal_input']['crystal_spacing'])
    c_width = config['graphite_input']['width']
    c_height= config['graphite_input']['height']
    
    meridi_focus = config['crystal_input']['radius'] * np.sin(c_bragg)
    sagitt_focus = - meridi_focus / np.cos(2 * c_bragg)

    if distance_c_d is None:
        distance_c_d = meridi_focus
        
    ## Source Placement
    s_origin, s_basis, path_vector, s_target = create_source_basis(distance_s_c)
    
    ## Crystal Placement
    c_origin  = s_origin + (path_vector * distance_s_c)
    c_orient  = np.array([0, 0, 1], dtype = np.float64)
    path_vector, c_basis, mesh_points, mesh_faces = create_optic_basis(
        path_vector, c_orient, c_bragg, c_width, c_height)

    ## Detector Placement
    d_origin = c_origin + (path_vector * distance_c_d)
    d_orient   = np.array([0, 0, 1], dtype = np.float64)
    d_basis    = create_detector_basis(path_vector, d_orient)
    
    #define meshes
    config['crystal_input']['mesh_points']  = mesh_points + c_origin
    config['crystal_input']['mesh_faces']   = mesh_faces    
    config['graphite_input']['mesh_points'] = mesh_points + c_origin
    config['graphite_input']['mesh_faces']  = mesh_faces
    
    ## Repack variables
    config['source_input']['origin']        = s_origin
    config['source_input']['zaxis']         = s_basis[2,:]
    config['source_input']['xaxis']         = s_basis[0,:]
    config['graphite_input']['origin']      = c_origin
    config['graphite_input']['zaxis']       = c_basis[2,:]
    config['graphite_input']['xaxis']       = c_basis[0,:]
    config['crystal_input']['origin']       = c_origin
    config['crystal_input']['zaxis']        = c_basis[2,:]
    config['crystal_input']['xaxis']        = c_basis[0,:]
    config['detector_input']['origin']      = d_origin
    config['detector_input']['zaxis']       = d_basis[2,:]
    config['detector_input']['xaxis']       = d_basis[0,:]
    config['source_input']['target']        = s_target
    
    return config


def setup_graphite_test(config):
    """
    An idealized scenario with a source, a crystal, and a detector

    This is meant to be used in conjunction with the beam scenario.
    Using the same configuration file this will generate a scenario where
    the crystal is removed and the detector is placed at the
    crystal-detector distance.
    """
    distance_s_g = config['scenario_input']['source_graphite_dist']
    distance_g_d = config['scenario_input']['crystal_detector_dist']
    g_bragg = bragg_angle(config['source_input']['wavelength'],
                          config['graphite_input']['crystal_spacing'])
    g_width = config['graphite_input']['width']
    g_height= config['graphite_input']['height']
    
    ## Source Placement
    s_origin, s_basis, path_vector, s_target = create_source_basis(distance_s_g)
    
    ## Graphite Placement
    g_origin  = s_origin + (path_vector * distance_s_g)
    g_orient    = np.array([0, 0, 1], dtype = np.float64)
    path_vector, g_basis, mesh_points, mesh_faces = create_optic_basis(
        path_vector, g_orient, g_bragg, g_width, g_height)
    
    ## Detector Placement
    d_origin = g_origin + (path_vector * distance_g_d)
    d_orient   = np.array([0, 0, 1], dtype = np.float64)
    d_basis    = create_detector_basis(path_vector, d_orient)
    
    #define meshes
    config['graphite_input']['mesh_points'] = mesh_points + g_origin
    config['graphite_input']['mesh_faces']  = mesh_faces
    config['crystal_input']['mesh_points']  = mesh_points + g_origin
    config['crystal_input']['mesh_faces']   = mesh_faces
    
    ## Repack variables
    config['source_input']['origin']        = s_origin
    config['source_input']['zaxis']         = s_basis[2,:]
    config['source_input']['xaxis']         = s_basis[0,:]
    config['graphite_input']['origin']      = g_origin
    config['graphite_input']['zaxis']       = g_basis[2,:]
    config['graphite_input']['xaxis']       = g_basis[0,:]
    config['crystal_input']['origin']       = g_origin
    config['crystal_input']['zaxis']        = g_basis[2,:]
    config['crystal_input']['xaxis']        = g_basis[0,:]
    config['detector_input']['origin']      = d_origin
    config['detector_input']['zaxis']       = d_basis[2,:]
    config['detector_input']['xaxis']       = d_basis[0,:]
    config['source_input']['target']        = s_target
    
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
    
    ## Source Placement
    s_origin, s_basis, path_vector, s_target = create_source_basis(distance_s_d)
        
    ## Detector Placement
    d_origin = s_origin + (path_vector * distance_s_d)
    d_orient   = np.array([0, 0, 1], dtype = np.float64)
    d_basis    = create_detector_basis(path_vector, d_orient)
    
    ## Repack variables
    config['source_input']['origin']        = s_origin
    config['source_input']['zaxis']         = s_basis[2,:]
    config['source_input']['xaxis']         = s_basis[0,:]
    config['detector_input']['origin']      = d_origin
    config['detector_input']['zaxis']       = d_basis[2,:]
    config['detector_input']['xaxis']       = d_basis[0,:]
    config['source_input']['target']        = s_target

    return config

