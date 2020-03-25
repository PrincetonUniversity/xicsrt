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
    config['general_input']['scenario']           = 'REAL'
    config['general_input']['system']             = 'w7x_ar16'
    config['general_input']['shot']               = 180707017
    
    """Ray emission settings
    'number_of_rays' typically should not exceed 1e7 unless running on a cluster
    If more rays are necessary, increase 'number_of_runs'.
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
    config['plasma_input']['use_poisson']         = True
    config['plasma_input']['do_monochrome']       = False
    config['plasma_input']['temperature_file']    = inpath + 'plasma_temperature.txt'
    config['plasma_input']['emissivity_file']     = inpath + 'plasma_emissivity_xe51.txt'
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
    config['plasma_input']['time_resolution']     = 1e-5
    config['plasma_input']['spread']              = 1.0
    config['plasma_input']['mass_number']         = 131.293
    config['plasma_input']['wavelength']          = 2.19 #2.7203 
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
    config['source_input']['spread']              = 1.0
    config['source_input']['temperature']         = 1000
    config['source_input']['mass_number']         = 131.293
    config['source_input']['wavelength']          = 2.19 #2.7203 
    config['source_input']['linewidth']           = 1.129e+14
    config['source_input']['velocity']            = np.array([0.0,0.0,0.0])
    
    """Geometry Settings
    'width', 'height', and 'depth' of source (meters)
    These values are arbitrary for now. Set to 0.0 for point source.
    Setting 'do_monochrome' to True causes all rays to have the same wavelength
    Setting 'use_poisson' to True causes the source to use a random intensity
    poisson-distributed around its input intensity. This preserves ray statistics
    in situations with very low intensity.
    """
    config['source_input']['origin']              = np.array([0.0, 0.0, 0.0])
    config['source_input']['zaxis']               = np.array([1.0, 0.0, 0.0])
    config['source_input']['xaxis']               = np.array([0.0, 0.0, 1.0])
    config['source_input']['target']              = np.array([1.0, 0.0, 0.0])
    
    config['source_input']['width']               = 0.0
    config['source_input']['height']              = 0.0
    config['source_input']['depth']               = 0.0
    
    config['source_input']['do_monochrome']       = False
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
    config['crystal_input']['do_bragg_checks']    = True
    config['crystal_input']['do_miss_checks']     = True
    config['crystal_input']['rocking_type']       = 'FILE'
    config['crystal_input']['use_meshgrid']       = False
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
    config['crystal_input']['crystal_spacing']    = 1.42 #1.7059
    config['crystal_input']['reflectivity']       = 1.0
    config['crystal_input']['rocking_fwhm']       = 90.30e-6
    config['crystal_input']['pixel_size']         = 0.0001
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
    config['crystal_input']['radius']             = 2.000 # 2.400
    
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
    config['graphite_input']['reflectivity']      = 1.0
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
    
    config['graphite_input']['width']             = 0.200
    config['graphite_input']['height']            = 0.200
    
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
    
    Chords 0 and 1 use Xe44+, while chords 2 and 3 use Xe55+
    config['scenario_input']['graphite_corners'][chord number, sight number, corner number, 3D coordinates]
    config['scenario_input']['crystal_corners'][chord number, corner number, 3D coordinates]
    config['scenario_input']['detector_centers'][chord number, 3D coordinates]
    """
    config['scenario_input']['graphite_corners'] = np.empty([4,3,4,3], dtype = np.float64)
    config['scenario_input']['crystal_corners']  = np.empty([4,4,3], dtype = np.float64)
    config['scenario_input']['detector_centers'] = np.empty([4,3], dtype = np.float64)
    config['scenario_input']['chord']            = 2
    config['scenario_input']['sight']            = 0
    
    config['scenario_input']['graphite_corners'][0,0,0,:] = np.array([048.5, 9324.9, -619.5])
    config['scenario_input']['graphite_corners'][0,0,1,:] = np.array([018.7, 9312.7, -619.4])
    config['scenario_input']['graphite_corners'][0,0,2,:] = np.array([017.4, 9400.8, -660.3])
    config['scenario_input']['graphite_corners'][0,0,3,:] = np.array([047.4, 9413.0, -660.5])
    config['scenario_input']['graphite_corners'][0,1,0,:] = np.array([094.6, 9224.8, -618.5])
    config['scenario_input']['graphite_corners'][0,1,1,:] = np.array([065.1, 9205.1, -618.2])
    config['scenario_input']['graphite_corners'][0,1,2,:] = np.array([063.3, 9290.9, -659.2])
    config['scenario_input']['graphite_corners'][0,1,3,:] = np.array([093.1, 9310.6, -659.4])
    config['scenario_input']['graphite_corners'][0,2,0,:] = np.array([140.6, 9166.1, -617.9])
    config['scenario_input']['graphite_corners'][0,2,1,:] = np.array([111.4, 9137.9, -617.5])
    config['scenario_input']['graphite_corners'][0,2,2,:] = np.array([109.3, 9219.6, -658.4])
    config['scenario_input']['graphite_corners'][0,2,3,:] = np.array([138.7, 9247.8, -658.7])
    config['scenario_input']['graphite_corners'][1,0,0,:] = np.array([057.2, 9787.3,  588.4])
    config['scenario_input']['graphite_corners'][1,0,1,:] = np.array([027.3, 9776.8,  590.2])
    config['scenario_input']['graphite_corners'][1,0,2,:] = np.array([028.9, 9682.3,  566.3])
    config['scenario_input']['graphite_corners'][1,0,3,:] = np.array([058.6, 9692.8,  564.5])
    config['scenario_input']['graphite_corners'][1,1,0,:] = np.array([102.9, 9575.8,  625.7])
    config['scenario_input']['graphite_corners'][1,1,1,:] = np.array([073.2, 9558.5,  628.7])
    config['scenario_input']['graphite_corners'][1,1,2,:] = np.array([075.2, 9465.8,  604.4])
    config['scenario_input']['graphite_corners'][1,1,3,:] = np.array([104.8, 9483.1,  601.5])
    config['scenario_input']['graphite_corners'][1,2,0,:] = np.array([150.6, 9415.4,  654.1])
    config['scenario_input']['graphite_corners'][1,2,1,:] = np.array([121.1, 9388.9,  658.6])
    config['scenario_input']['graphite_corners'][1,2,2,:] = np.array([123.6, 9300.3,  633.6])
    config['scenario_input']['graphite_corners'][1,2,3,:] = np.array([152.8, 9326.8,  629.1])
    config['scenario_input']['graphite_corners'][2,0,0,:] = np.array([053.1, 9252.0, -669.9])
    config['scenario_input']['graphite_corners'][2,0,1,:] = np.array([018.6, 9222.5, -669.9])
    config['scenario_input']['graphite_corners'][2,0,2,:] = np.array([016.8, 9333.6, -709.9])
    config['scenario_input']['graphite_corners'][2,0,3,:] = np.array([051.5, 9362.9, -709.9])
    config['scenario_input']['graphite_corners'][2,1,0,:] = np.array([100.0, 9151.7, -670.0])
    config['scenario_input']['graphite_corners'][2,1,1,:] = np.array([066.1, 9102.3, -669.9])
    config['scenario_input']['graphite_corners'][2,1,2,:] = np.array([063.9, 9204.1, -710.0])
    config['scenario_input']['graphite_corners'][2,1,3,:] = np.array([098.0, 9253.2, -709.9])
    config['scenario_input']['graphite_corners'][2,2,0,:] = np.array([145.8, 9141.4, -670.0])
    config['scenario_input']['graphite_corners'][2,2,1,:] = np.array([112.5, 9075.5, -669.9])
    config['scenario_input']['graphite_corners'][2,2,2,:] = np.array([110.2, 9164.5, -710.0])
    config['scenario_input']['graphite_corners'][2,2,3,:] = np.array([143.6, 9230.2, -709.9])
    config['scenario_input']['graphite_corners'][3,0,0,:] = np.array([069.2, 9228.1,  633.8])
    config['scenario_input']['graphite_corners'][3,0,1,:] = np.array([034.9, 9181.4,  641.5])
    config['scenario_input']['graphite_corners'][3,0,2,:] = np.array([036.9, 9073.4,  618.6])
    config['scenario_input']['graphite_corners'][3,0,3,:] = np.array([071.0, 9120.3,  611.1])
    config['scenario_input']['graphite_corners'][3,1,0,:] = np.array([116.8, 9152.1,  646.4])
    config['scenario_input']['graphite_corners'][3,1,1,:] = np.array([083.4, 9079.2,  658.4])
    config['scenario_input']['graphite_corners'][3,1,2,:] = np.array([085.4, 8993.6,  631.8])
    config['scenario_input']['graphite_corners'][3,1,3,:] = np.array([118.6, 9066.8,  620.0])
    config['scenario_input']['graphite_corners'][3,2,0,:] = np.array([163.5, 9122.5,  651.2])
    config['scenario_input']['graphite_corners'][3,2,1,:] = np.array([130.8, 9034.8,  665.6])
    config['scenario_input']['graphite_corners'][3,2,2,:] = np.array([132.7, 8971.5,  635.4])
    config['scenario_input']['graphite_corners'][3,2,3,:] = np.array([165.2, 9059.3,  621.2])
    
    config['scenario_input']['crystal_corners'][0,0,:]    = np.array([-055.0, 17704.9, -714.9])
    config['scenario_input']['crystal_corners'][0,1,:]    = np.array([-105.0, 17704.0, -714.9])
    config['scenario_input']['crystal_corners'][0,2,:]    = np.array([-105.0, 17680.6, -745.1])
    config['scenario_input']['crystal_corners'][0,3,:]    = np.array([-055.0, 17681.5, -745.1])
    config['scenario_input']['crystal_corners'][1,0,:]    = np.array([-055.0, 17585.6, -793.5])
    config['scenario_input']['crystal_corners'][1,1,:]    = np.array([-105.0, 17584.6, -793.5])
    config['scenario_input']['crystal_corners'][1,2,:]    = np.array([-105.0, 17559.8, -816.5])
    config['scenario_input']['crystal_corners'][1,3,:]    = np.array([-055.0, 17560.9, -816.5])
    config['scenario_input']['crystal_corners'][2,0,:]    = np.array([-075.0, 18076.2, -674.1])
    config['scenario_input']['crystal_corners'][2,1,:]    = np.array([-125.0, 18075.2, -674.1])
    config['scenario_input']['crystal_corners'][2,2,:]    = np.array([-125.0, 18049.1, -705.9])
    config['scenario_input']['crystal_corners'][2,3,:]    = np.array([-075.0, 18050.1, -705.9])
    config['scenario_input']['crystal_corners'][3,0,:]    = np.array([-075.0, 17997.3, -817.7])
    config['scenario_input']['crystal_corners'][3,1,:]    = np.array([-125.0, 17996.2, -817.7])
    config['scenario_input']['crystal_corners'][3,2,:]    = np.array([-125.0, 17968.0, -842.3])
    config['scenario_input']['crystal_corners'][3,3,:]    = np.array([-075.0, 17969.1, -842.3])
    
    config['scenario_input']['detector_centers'][0,:]     = np.array([-68.8, 17193.2, 1117.6])
    config['scenario_input']['detector_centers'][1,:]     = np.array([-67.4, 17381.1, 1099.3])
    config['scenario_input']['detector_centers'][2,:]     = np.array([-93.2, 17732.9,  977.6])
    config['scenario_input']['detector_centers'][3,:]     = np.array([-94.0, 17928.6,  869.1])
    
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