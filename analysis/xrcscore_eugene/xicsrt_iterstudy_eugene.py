# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

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
from collections import OrderedDict

def get_config():
    config = OrderedDict()
    config['general_input']   = OrderedDict()
    config['scenario_input']  = OrderedDict()
    config['plasma_input']    = OrderedDict()
    config['source_input']    = OrderedDict()
    config['filter_input']    = OrderedDict()
    config['graphite_input']  = OrderedDict()
    config['crystal_input']   = OrderedDict()
    config['detector_input']  = OrderedDict()
    
    # -------------------------------------------------------------------------
    ## General raytracer properties
    """File and type settings
    possible scenarios include 'REAL', 'PLASMA', 'THROUGHPUT', 'BEAM',
    'CRYSTAL', 'GRAPHITE', 'SOURCE'
    """
    inpath = '/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/xrcscore_eugene/inputs/'
    outpath= '/Users/Eugene/PPPL_python_project1/xics_rt_code/results/'
    config['general_input']['input_path']         = inpath
    config['general_input']['output_path']        = outpath
    config['general_input']['output_suffix']      = '.tif'
    config['general_input']['scenario']           = 'MANFRED'
    config['general_input']['system']             = 'w7x_ar16'
    config['general_input']['shot']               = 180707017
    
    """Ray emission settings
    'number_of_rays' typically should not exceed 1e7 unless running on a cluster
    If more rays are necessary, increase 'number of runs'.
    """
    config['general_input']['number_of_rays']     = int(1e7)
    config['general_input']['number_of_runs']     = 1
    
    """Raytrace run settings
    set ideal_geometry to False to enable thermal expansion
    set backwards_raytrace to True to swap the detector and source
    set do_visualizations to toggle the visualizations on or off
    set do_savefiles to toggle whether the program saves .tif files
    change the random seed to alter the random numbers generated
    change xics_temp to test thermal expansion (kelvin)
    """
    config['general_input']['ideal_geometry']     = True
    config['general_input']['backwards_raytrace'] = False
    config['general_input']['do_visualizations']  = True
    config['general_input']['do_savefiles']       = True
    config['general_input']['random_seed']        = 12345
    config['general_input']['xics_temp']          = 273.0
    
    # -------------------------------------------------------------------------
    ## Load plasma properties
    """Type and file settings
    setting 'use_profiles' to True causes the code to read the temperature and
    emissivity profiles located at 'temperature_data' and 'emissivity_data' paths
    setting 'use_profiles' to False causes the code to use 'temperature' and
    'emissivity' instead, as flat step-function distributions
    All temperatures are in (eV) and emissivities are in (photons m^-3 s^-1)
    """
    #config['plasma_input']['use_profiles']        = True
    config['plasma_input']['use_poisson']         = True
    config['plasma_input']['temperature_file']    = inpath + 'plasma_temperature.txt'
    config['plasma_input']['emissivity_file']     = inpath + 'plasma_emissivity_xe44.txt'
    config['plasma_input']['velocity_file']       = inpath + 'plasma_velocity.txt'
    config['plasma_input']['wout_file']           = inpath + 'wout_iter.nc'
    config['plasma_input']['temperature']         = 1000
    config['plasma_input']['emissivity']          = 1e18
    config['plasma_input']['velocity']            = np.array([0.0,0.0,0.0])
    
    """Geometry settings
    The plasma is a torus with a 'major_radius' and 'minor_radius' (meters)
    Only a small cubic chunk of the plasma is rendered and emits rays
    This cubic chunk has a 'width', 'height', and 'depth' (meters)
    """
    config['plasma_input']['origin']              = np.array([0.0, 6.2, 0.5])
    config['plasma_input']['zaxis']               = np.array([0.0, 0.0, 1.0])
    config['plasma_input']['xaxis']               = np.array([1.0, 0.0, 0.0])
    config['plasma_input']['target']              = np.array([1.0, 0.0, 0.0])
    
    #config['plasma_input']['major_radius']        = 6.2
    #config['plasma_input']['minor_radius']        = 2.0
    config['plasma_input']['width']               = 4.0
    config['plasma_input']['height']              = 4.0
    config['plasma_input']['depth']               = 7.5
    
    """Bundle Settings
    The plasma works by emitting cubic ray bundles, which have their own settings
    NOTE: plasma volume, bundle volume, bundle count, and bundle_factor are 
    intrinsically linked. Setting 'bundle_type' to 'POINT' will calculate 
    bundle count from plasma volume / bundle volume. Setting 'bundle_type' to
    'VOXEL' will calculate bundle volume from plasma volume / bundle count.
    
    'max_rays' should equal config['general_input']['number_of_rays']
    'bundle_count' typically should not exceed 1e7 unless running on a cluster
    'bundle_factor' lets you interchange generating more bundles or more rays
    per bundle. Decreasing 'bundle_factor' results in better sampling, but slower.
    'bundle_volume'         is the volume of each bundle        (meters^3)
    'time_resolution'       is the emissivity integration time  (sec)
    'spread'                is the angular spread               (degrees)
    'mass_number'           is the impurity mass                (AMU)
    'wavelength'            is the x-ray emission line location (angstroms)
    'linewidth'             is the x-ray natural linewidth      (1/s)
    """
    config['plasma_input']['max_rays']            = config['general_input']['number_of_rays']
    config['plasma_input']['bundle_type']         = 'VOXEL'
    config['plasma_input']['bundle_count']        = int(1e5)
    config['plasma_input']['bundle_volume']       = 0.01 ** 3
    config['plasma_input']['time_resolution']     = 1e-7
    config['plasma_input']['spread']              = 1.0
    config['plasma_input']['mass_number']         = 131.293
    config['plasma_input']['wavelength']          = 2.7203
    config['plasma_input']['linewidth']           = 1.129e+14
    
    # -------------------------------------------------------------------------
    ## Load filter properties
    """Sightline Settings
    The plasma sightline is a vector that extends from the graphite 
    pre-reflector to the plasma. This improves rendering efficiency since the 
    plasma only needs to render bundles near the sightline.
    The sightline has a thickness (meters)
    """
    config['filter_input']['origin']    = np.array([0.0, 0.0, 0.0])
    config['filter_input']['direction'] = np.array([0.0, 1.0, 0.0])
    config['filter_input']['radius']    = 0.100
    
    # -------------------------------------------------------------------------
    ## Load source properties
    """Source Settings
    'intensity' should equal config['general_input']['number_of_rays']
    'spread'                is the angular spread               (degrees)
    'temperature'           is the ion temperature              (eV)
    'mass_number'           is the impurity mass                (AMU)
    'wavelength'            is the x-ray emission line location (angstroms)
    'linewidth'             is the x-ray natural linewidth      (1/s)
    'velocity'              is the impurity ion velocity vector (m/s)
    """
    config['source_input']['intensity']           = config['general_input']['number_of_rays']
    config['source_input']['spread']              = 2.0
    config['source_input']['temperature']         = 1000
    config['source_input']['mass_number']         = 131.293
    config['source_input']['wavelength']          = 2.7203
    config['source_input']['linewidth']           = 1.129e+14
    config['source_input']['velocity']            = np.array([0.0,0.0,0.0])
    
    """Geometry Settings
    'width', 'height', and 'depth' of source (meters)
    These values are arbitrary for now. Set to 0.0 for point source.
    """
    config['source_input']['origin']              = np.array([0.0, 0.0, 0.0])
    config['source_input']['zaxis']               = np.array([1.0, 0.0, 0.0])
    config['source_input']['xaxis']               = np.array([0.0, 0.0, 1.0])
    config['source_input']['target']              = np.array([1.0, 0.0, 0.0])
    
    config['source_input']['width']               = 0.0
    config['source_input']['height']              = 0.0
    config['source_input']['depth']               = 0.0
    
    config['source_input']['use_poisson']         = False
    
    # -------------------------------------------------------------------------
    ## Load spherical crystal properties
    """Type and file settings
    setting 'do_bragg_checks' to False makes the crystal ignore Bragg conditions
    setting 'do_miss_checks'  to False prevents the crystal from masking missed rays
    setting 'use_trimesh' to False defaults to using simple rectangle geometry
    possible rocking curve types include 'STEP', 'GAUSS', and 'FILE'
    sigma and pi are polarized rocking curves. 'rocking_mix' interpolates between them.
    A 'rocking_mix' of 1.0 is 100% sigma curve, while 0.0 is 100% pi curve.
    """
    config['crystal_input']['do_bragg_checks']    = False
    config['crystal_input']['do_miss_checks']     = True
    config['crystal_input']['rocking_type']       = 'FILE'
    config['crystal_input']['use_meshgrid']       = True
    config['crystal_input']['rocking_mix']        = 1.0
    config['crystal_input']['rocking_sigma_file'] = inpath + 'rocking_curve_germanium_sigma.txt'
    config['crystal_input']['rocking_pi_file']    = inpath + 'rocking_curve_germanium_pi.txt'
    
    """Crystal settings
    'crystal_spacing'is the inter-atomic spacing (angstrom)
    'reflectivity'   is the maximum reflectivity used for 'STEP' and 'GAUSS'
    'rocking_type'   is the rocking curve FWHM (rad) used for 'STEP' and 'GAUSS'
    'pixel_size'     is the size of the pixels used by the image generator
    'therm_expand'   is the thermal expansion coefficient (1/kelvin)
    """
    config['crystal_input']['crystal_spacing']    = 1.7059
    config['crystal_input']['reflectivity']       = 1
    config['crystal_input']['rocking_fwhm']       = 90.30e-6
    config['crystal_input']['pixel_size']         = 0.00001
    #config['crystal_input']['therm_expand']       = 5.9e-6
    
    """Geometry Settings
    crystal 'width' and 'height' (meters) only matter when 'use_trimesh' is False
    'radius' is the crystal's radius of curvature (meters)
    """
    config['crystal_input']['origin']             = np.array([0.0, 0.0, 0.0])
    config['crystal_input']['zaxis']              = np.array([0.0, 0.0, 0.0])
    config['crystal_input']['xaxis']              = np.array([0.0, 0.0, 0.0])
    
    config['crystal_input']['width']              = 0.040
    config['crystal_input']['height']             = 0.050
    config['crystal_input']['radius']             = 2.400
    
    """
    Rocking curve FWHM:  90.30 urad
    Darwin Curve, sigma: 48.070 urad
    Darwin Curve, pi:    14.043 urad
    Taken from XoP
    """
    bragg = np.arcsin(config['source_input']['wavelength'] / (2 * config['crystal_input']['crystal_spacing']))
    dx = config['crystal_input']['height'] * np.cos(bragg) / 2
    dy = config['crystal_input']['height'] * np.sin(bragg) / 2
    dz = config['crystal_input']['width']                  / 2
    
    p1 = config['crystal_input']['origin'] + np.array([ dx, dy, dz])
    p2 = config['crystal_input']['origin'] + np.array([-dx,-dy, dz])
    p3 = config['crystal_input']['origin'] + np.array([-dx,-dy,-dz])
    p4 = config['crystal_input']['origin'] + np.array([ dx, dy,-dz])
        
    config['crystal_input']['mesh_points'] = np.array([p1, p2, p3, p4])
    config['crystal_input']['mesh_faces']  = np.array([[0,1,2],[2,3,0]])
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
    config['graphite_input']['do_bragg_checks']   = True
    config['graphite_input']['do_miss_checks']    = True
    config['graphite_input']['rocking_type']      = "GAUSS"
    config['graphite_input']['use_meshgrid']      = True
    config['graphite_input']['rocking_mix']       = 1.0
    config['graphite_input']['rocking_sigma_file']= inpath + 'rocking_curve_graphite_sigma.txt'
    config['graphite_input']['rocking_pi_file']   = inpath + 'rocking_curve_graphite_pi.txt'
    
    """Graphite settings
    'crystal_spacing'is the inter-atomic spacing (angstrom)
    'reflectivity'   is the maximum reflectivity used for 'STEP' and 'GAUSS'
    'mosaic_spread'  is the crystallite mosaic spread FWHM (degrees)
    'rocking_type'   is the rocking curve FWHM (rad) used for 'STEP' and 'GAUSS'
    'pixel_size'     is the size of the pixels used by the image generator
    'therm_expand'   is the thermal expansion coefficient (1/kelvin)
    """
    config['graphite_input']['crystal_spacing']   = 3.35
    config['graphite_input']['reflectivity']      = 1
    config['graphite_input']['mosaic_spread']     = 0.5
    config['graphite_input']['rocking_fwhm']      = 2620e-6
    config['graphite_input']['pixel_size']        = 0.0001
    #config['graphite_input']['therm_expand']      = 20e-6
    
    """Geometry Settings
    graphite 'width' and 'height' (meters) only matter when 'use_meshgrid' is False
    """
    config['graphite_input']['origin']            = np.array([1.0, 0.0, 0.0])
    config['graphite_input']['zaxis']             = np.array([0.0, 0.0, 0.0])
    config['graphite_input']['xaxis']             = np.array([0.0, 0.0, 1.0])
    
    config['graphite_input']['width']             = 0.125
    config['graphite_input']['height']            = 0.040
    
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
    config['detector_input']['do_miss_checks']    = True
    
    config['detector_input']['origin']            = np.array([0.0, 0.0, 0.0])
    config['detector_input']['zaxis']             = np.array([0.0, 0.0, 0.0])
    config['detector_input']['xaxis']             = np.array([0.0, 0.0, 0.0])
    
    config['detector_input']['pixel_size']        = 0.000172
    #config['detector_input']['pixel_width']       = 195
    #config['detector_input']['pixel_height']      = 1475
    config['detector_input']['width']             = (195  * config['detector_input']['pixel_size'])
    config['detector_input']['height']            = (1475 * config['detector_input']['pixel_size'])

    # -------------------------------------------------------------------------
    ## Load scenario properties
    """
    Distances are in (meters). Set crystal_detector_dist to None so that the
    scenario generator defaults to placing the detector at the crystal's
    meridional focus.
    """
    config['scenario_input']['source_graphite_dist']  = 2
    config['scenario_input']['graphite_crystal_dist'] = 8.5
    config['scenario_input']['crystal_detector_dist'] = 2.4
    
    """
    Convert the numbers given in the XICS presentations into useful information.
    When copying values from the XICS presentations, please place them here.
    config['graphite_input'][chord number][corner number][3D coordinates]
    """
    config['scenario_input']['chord']               = 0
    config['scenario_input']['graphite_corners']    = np.array([[[240.59, 9180.83, -599.40],
                                                                 [212.04, 9141.38, -598.75],
                                                                 [209.38, 9214.92, -639.89],
                                                                 [238.10, 9254.18, -640.30]],
                                                                [[185.99, 9413.40, -602.60],
                                                                 [156.61, 9395.54, -602.32],
                                                                 [153.94, 9482.08, -643.57],
                                                                 [183.53, 9499.84, -643.75]],
                                                                [[135.88, 9539.44, -604.24],
                                                                 [106.26, 9527.76, -604.05],
                                                                 [104.04, 9615.99, -645.31],
                                                                 [133.87, 9627.61, -645.43]],
                                                                [[095.54, 9252.32, -600.37],
                                                                 [066.06, 9229.01, -599.99],
                                                                 [064.40, 9313.23, -641.23],
                                                                 [094.07, 9336.41, -641.48]],
                                                                [[050.36, 9068.45, -597.80],
                                                                 [020.78, 9043.69, -597.38],
                                                                 [019.58, 9127.13, -638.61],
                                                                 [049.35, 9151.75, -638.89]]])
    config['scenario_input']['crystal_corners']     =  np.array([[-105., 17876.47, -760.],
                                                                 [-105., 17907.65, -720.],
                                                                 [-055., 17908.85, -720.],
                                                                 [-055., 17877.69, -760.]])
    config['scenario_input']['detector_corners']    =  np.array([[-124.590, 17293.39, 1082.600],
                                                                 [ 074.544, 17311.94, 1082.603],
                                                                 [ 056.630, 17504.27, 1134.460],
                                                                 [-142.510, 17485.72, 1134.460]])
    
    return config

def get_config_multi(configurations):
    config_multi = dict()
    for ii in range(configurations):
        config = get_config()
        config_multi[str(ii)] = config
        
    return config_multi

## Run the scripts in order (TEMPORARY - Find a better place to put this code)
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/xrcscore_eugene/')

from xrcscore_eugene.xicsrt_initialize import initialize, initialize_multi
from xrcscore_eugene.xicsrt_run import run, run_multi
from xicsrt.xicsrt_input import save_config, load_config

runtype = 'single'
filepath= 'xicsrt_input.json'

try:
    load_config(filepath)
except FileNotFoundError:
    print(filepath + ' not found!')

if runtype == 'single':
    config = get_config()
    config = initialize(config)
    output, meta = run(config)

if runtype == 'multi':
    config_multi = get_config_multi(5)
    config_multi = initialize_multi(config_multi)
    output, meta = run_multi(config_multi)

if runtype == 'save':
    config_multi = get_config_multi(5)
    config_multi = initialize_multi(config_multi)
    save_config(filepath, config_multi)
        
if runtype == 'init_resave':
    config_multi = load_config(filepath)
    config_multi = initialize_multi(config_multi)    
    save_config(filepath, config)

if runtype == 'raw_load':
    config_multi = load_config(filepath)
    output, meta = run_multi(config_multi)
    
if runtype == 'init_load':
    config_multi = load_config(filepath)
    config_multi = initialize_multi(config_multi)
    output, meta = run_multi(config_multi)