# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 09:48:30 2019

Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
This script creates a default configuration for a single XICS run of the ITER XRCS-Core.
"""

import numpy as np
import logging
from collections import OrderedDict

def get_config():
    config = OrderedDict()
    config['general']   = OrderedDict()
    config['filters']   = OrderedDict()
    config['sources']   = OrderedDict()
    config['optics']    = OrderedDict()
    
    config['sources']['focused']   = OrderedDict()
    config['optics']['graphite']   = OrderedDict()
    config['optics']['detector']   = OrderedDict()
    
    # -------------------------------------------------------------------------
    ## General raytracer properties
    """File and type settings
    """
    inpath = '/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/xrcscore_eugene/inputs/'
    outpath= '/Users/Eugene/PPPL_python_project1/xics_rt_code/results/'
    config['general']['input_path']         = inpath
    config['general']['output_path']        = outpath
    config['general']['number_of_iter']     = 1
    config['general']['number_of_runs']     = 1
    
    """Raytrace run settings
    set ideal_geometry to False to enable thermal expansion
    set backwards_raytrace to True to swap the detector and source
    set do_visualizations to toggle the visualizations on or off
    set do_savefiles to toggle whether the program saves .tif files
    change the random seed to alter the random numbers generated
    change xics_temp to test thermal expansion (kelvin)
    """
    config['general']['ideal_geometry']     = True
    config['general']['save_images']        = True
    config['general']['print_results']      = True
    config['general']['random_seed']        = 12345
    
    config['general']['keep_meta']          = True
    config['general']['keep_images']        = True
    config['general']['keep_history']       = True
    
    config['general']['save_meta']          = True
    config['general']['save_images']        = True
    config['general']['save_history']       = True
    config['general']['save_run_images']    = True
    
    # -------------------------------------------------------------------------
    ## Load focused extended source properties 
    config['sources']['focused']['class_name']      = 'XicsrtSourceFocused'
    config['sources']['focused']['use_poisson']     = False
    config['sources']['focused']['do_monochrome']   = True
    config['sources']['focused']['temperature']     = 1000
    config['sources']['focused']['intensity']       = 1e7
    config['sources']['focused']['velocity']        = np.array([0.0,0.0,0.0])

    config['sources']['focused']['origin']          = np.array([0.0, 6.2, 0.5])
    config['sources']['focused']['zaxis']           = np.array([0.0, 0.0, 1.0])
    config['sources']['focused']['xaxis']           = np.array([1.0, 0.0, 0.0])
    config['sources']['focused']['target']          = np.array([1.0, 0.0, 0.0])
    config['sources']['focused']['width']           = 0.1
    config['sources']['focused']['height']          = 0.0
    config['sources']['focused']['depth']           = 0.0
    
    config['sources']['focused']['spread']           = 0.001
    config['sources']['focused']['mass_number']      = 131.293
    config['sources']['focused']['wavelength']       = 12.398425 / 17.0
    config['sources']['focused']['linewidth']        = 1.129e+14
    
    # -------------------------------------------------------------------------
    ## Load mosaic graphite properties
    """Type and file settings
    setting 'do_bragg_checks' to False makes the graphite ignore Bragg conditions
    setting 'do_miss_checks'  to False prevents the graphite from masking missed rays
    setting 'use_trimesh' to False defaults to using simple rectangle geometry
    possible rocking curve types include 'STEP', 'GAUSS', and 'FILE'
    sigma and pi are polarized rocking curves. 'rocking_mix' interpolates between them.
    A 'rocking_mix' of 1.0 is 100% sigma curve, while 0.0 is 100% pi curve.
    """
    
    config['optics']['graphite']['class_name']        = 'XicsrtOpticMosaicGraphite'
    config['optics']['graphite']['do_bragg_check']    = True
    config['optics']['graphite']['do_miss_check']     = True
    config['optics']['graphite']['rocking_type']      = "GAUSS"
    config['optics']['graphite']['use_meshgrid']      = False
    config['optics']['graphite']['rocking_mix']       = 1.0
    config['optics']['graphite']['rocking_sigma_file']= inpath + 'rocking_curve_graphite_sigma.txt'
    config['optics']['graphite']['rocking_pi_file']   = inpath + 'rocking_curve_graphite_pi.txt'
    
    """Graphite settings
    'crystal_spacing'is the inter-atomic spacing (angstrom)
    'reflectivity'   is the maximum reflectivity used for 'STEP' and 'GAUSS'
    'mosaic_spread'  is the crystallite mosaic spread FWHM (degrees)
    'rocking_type'   is the rocking curve FWHM (rad) used for 'STEP' and 'GAUSS'
    'pixel_size'     is the size of the pixels used by the image generator
    'therm_expand'   is the thermal expansion coefficient (1/kelvin)
    """
    config['optics']['graphite']['crystal_spacing']   = 3.35
    config['optics']['graphite']['reflectivity']      = 0.96
    config['optics']['graphite']['mosaic_spread']     = 0.4
    config['optics']['graphite']['rocking_fwhm']      = 2620e-6
    config['optics']['graphite']['pixel_size']        = 0.00001
    #config['optics']['graphite']['therm_expand']      = 20e-6
    
    """Geometry Settings
    graphite 'width' and 'height' (meters) only matter when 'use_meshgrid' is False
    """
    config['optics']['graphite']['origin']            = np.array([1.0, 0.0, 0.0])
    config['optics']['graphite']['zaxis']             = np.array([0.0, 0.0, 0.0])
    config['optics']['graphite']['xaxis']             = np.array([0.0, 0.0, 1.0])
    
    config['optics']['graphite']['width']             = 0.0001
    config['optics']['graphite']['height']            = 0.0001
    
    """
    HOPG Crystallite Rocking Curve FWHM: 2620 urad (0.15 degrees)
    Taken from Ohler et al. “X-ray topographic determination of the granular
    structure in a graphite mosaic crystal: a three-dimensional reconstruction”
    """
    
    # -------------------------------------------------------------------------
    ## Load detector properties
    """
    setting 'do_miss_checks'  to False prevents the detector from masking missed rays
    'pixel_size' is in (meters)
    """
    config['optics']['detector']['class_name']        ='XicsrtOpticDetector'
    config['optics']['detector']['do_miss_check']     = True
    
    config['optics']['detector']['origin']            = np.array([0.0, 0.0, 0.0])
    config['optics']['detector']['zaxis']             = np.array([0.0, 0.0, 0.0])
    config['optics']['detector']['xaxis']             = np.array([0.0, 0.0, 0.0])
    
    config['optics']['detector']['pixel_size']        = 0.0001
    config['optics']['detector']['width']             = (200 * config['optics']['detector']['pixel_size'])
    config['optics']['detector']['height']            = (200 * config['optics']['detector']['pixel_size'])

    config = setup_fwhm_scenario(config)
    
    return config
    
def get_config_multi(configurations):
    config_multi = dict()
    for ii in range(configurations):
        config = get_config()
        config_multi[str(ii)] = config
        
    return config_multi

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

def setup_fwhm_scenario(config):
    """
    An idealized scenario with a source, a graphite, and a detector

    This is meant to be used in conjunction with the beam scenario.
    Using the same configuration file this will generate a scenario where
    the crystal is removed and the detector is placed at the
    crystal-detector distance.
    """
    from xicsrt.xicsrt_math import bragg_angle
    
    distance_s_g = 1
    distance_g_d = 1
    g_bragg = bragg_angle(config['sources']['focused']['wavelength'],
                          config['optics']['graphite']['crystal_spacing'])
    g_width = config['optics']['graphite']['width']
    g_height= config['optics']['graphite']['height']
    
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
    config['optics']['graphite']['mesh_points'] = mesh_points + g_origin
    config['optics']['graphite']['mesh_faces']  = mesh_faces
    
    ## Repack variables
    config['sources']['focused']['origin']      = s_origin
    config['sources']['focused']['zaxis']       = s_basis[2,:]
    config['sources']['focused']['xaxis']       = s_basis[0,:]
    config['optics']['graphite']['origin']      = g_origin
    config['optics']['graphite']['zaxis']       = g_basis[2,:]
    config['optics']['graphite']['xaxis']       = g_basis[0,:]
    config['optics']['detector']['origin']      = d_origin
    config['optics']['detector']['zaxis']       = d_basis[2,:]
    config['optics']['detector']['xaxis']       = d_basis[0,:]
    config['sources']['focused']['target']      = s_target
    
    return config

## Run the scripts in order (TEMPORARY - Find a better place to put this code)
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/xrcscore_eugene/')

from xicsrt import xicsrt_raytrace, xicsrt_input

visuals = False
runtype = 'single'
filepath= 'xicsrt_input.json'

logging.basicConfig(level=logging.DEBUG)

try:
    xicsrt_input.load_config(filepath)
except FileNotFoundError:
    print(filepath + ' not found!')

if runtype == 'single':
    config = get_config()
    output = xicsrt_raytrace.raytrace(config)

if runtype == 'multi':
    config = get_config_multi(5)
    output = xicsrt_raytrace.raytrace_multi(config)

if runtype == 'save':
    config = get_config()
    xicsrt_input.save_config(filepath, config)
        
if runtype == 'load':
    config = xicsrt_input.load_config(filepath)
    output = xicsrt_raytrace.raytrace(config)
    
if visuals:
    from xicsrt.visual import xicsrt_visualizer
    import matplotlib.pyplot as plt
    fig1 = xicsrt_visualizer.visualize_layout(config)
    fig2 = xicsrt_visualizer.visualize_bundles(config, output)
    fig3 = xicsrt_visualizer.visualize_vectors(config, output)
    
    plt.show(fig1)
    plt.show(fig2)
    plt.show(fig3)
    