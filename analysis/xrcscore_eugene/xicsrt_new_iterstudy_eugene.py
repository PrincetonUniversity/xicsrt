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
    
    config['filters']['sightline'] = OrderedDict()
    #config['sources']['focused']   = OrderedDict()
    config['sources']['plasma']    = OrderedDict()
    config['optics']['graphite']   = OrderedDict()
    config['optics']['crystal']    = OrderedDict()
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
    
    config['general']['keep_meta']          = False
    config['general']['keep_images']        = False
    config['general']['keep_history']       = False
    
    config['general']['save_meta']          = True
    config['general']['save_images']        = True
    config['general']['save_history']       = True
    config['general']['save_run_images']    = True
    
    # -------------------------------------------------------------------------
    ## Load filter properties
    """Sightline Settings
    The plasma sightline is a vector that extends from the graphite 
    pre-reflector to the plasma. This improves rendering efficiency since the 
    plasma only needs to render bundles near the sightline.
    The sightline has a thickness (meters)
    """
    config['filters']['sightline']['class_name'] = 'XicsrtBundleFilterSightline'
    config['filters']['sightline']['origin']     = np.array([0.0, 0.0, 0.0])
    config['filters']['sightline']['direction']  = np.array([0.0, 1.0, 0.0])
    config['filters']['sightline']['radius']     = 0.100
    
    # -------------------------------------------------------------------------
    ## Load plasma properties
    """Type and file settings
    setting 'use_profiles' to True causes the code to read the temperature and
    emissivity profiles located at 'temperature_data' and 'emissivity_data' paths
    setting 'use_profiles' to False causes the code to use 'temperature' and
    'emissivity' instead, as flat step-function distributions
    All temperatures are in (eV) and emissivities are in (photons m^-3 s^-1)
    """
    
    config['sources']['plasma']['class_name']       = 'XicsrtPlasmaVmecDatafile'
    config['sources']['plasma']['filter_list']      = ['XicsrtBundleFilterSightline']
    config['sources']['plasma']['use_poisson']      = True
    config['sources']['plasma']['do_monochrome']    = False
    config['sources']['plasma']['temperature_file'] = inpath + 'plasma_temperature.txt'
    config['sources']['plasma']['emissivity_file']  = inpath + 'plasma_emissivity_xe51.txt'
    config['sources']['plasma']['velocity_file']    = inpath + 'plasma_velocity.txt'
    config['sources']['plasma']['wout_file']        = inpath + 'wout_iter.nc'
    config['sources']['plasma']['temperature']      = 1000
    config['sources']['plasma']['emissivity']       = 1e18
    config['sources']['plasma']['velocity']         = np.array([0.0,0.0,0.0])
    
    """Geometry settings
    The plasma is a torus with a 'major_radius' and 'minor_radius' (meters)
    Only a small cubic chunk of the plasma is rendered and emits rays
    This cubic chunk has a 'width', 'height', and 'depth' (meters)
    """
    config['sources']['plasma']['origin']           = np.array([0.0, 6.2, 0.5])
    config['sources']['plasma']['zaxis']            = np.array([0.0, 0.0, 1.0])
    config['sources']['plasma']['xaxis']            = np.array([1.0, 0.0, 0.0])
    config['sources']['plasma']['target']           = np.array([1.0, 0.0, 0.0])
    
    #config['sources']['plasma']['major_radius']     = 6.2
    #config['sources']['plasma']['minor_radius']     = 2.0
    config['sources']['plasma']['width']            = 4.0
    config['sources']['plasma']['height']           = 4.0
    config['sources']['plasma']['depth']            = 7.5
    

    """Bundle Settings
    The plasma works by emitting cubic ray bundles, which have their own settings
    NOTE: plasma volume, bundle volume, bundle count, and bundle_factor are 
    intrinsically linked. Setting 'bundle_type' to 'POINT' will calculate 
    bundle count from plasma volume / bundle volume. Setting 'bundle_type' to
    'VOXEL' will calculate bundle volume from plasma volume / bundle count.
    
    'max_rays' should equal config['general']['number_of_rays']
    'bundle_count' typically should not exceed 1e7 unless running on a cluster
    'bundle_volume'         is the volume of each bundle        (meters^3)
    'time_resolution'       is the emissivity integration time  (sec)
    'spread'                is the angular spread               (degrees)
    'mass_number'           is the impurity mass                (AMU)
    'wavelength'            is the x-ray emission line location (angstroms)
    'linewidth'             is the x-ray natural linewidth      (1/s)
    """
    
    config['sources']['plasma']['max_rays']         = int(1.5e7)
    config['sources']['plasma']['bundle_type']      = 'VOXEL'
    config['sources']['plasma']['bundle_count']     = int(1e5)
    config['sources']['plasma']['bundle_volume']    = 0.01 ** 3
    config['sources']['plasma']['time_resolution']  = 1e-5
    config['sources']['plasma']['spread']           = 1.0
    config['sources']['plasma']['mass_number']      = 131.293
    config['sources']['plasma']['wavelength']       = 2.19 #2.7203 
    config['sources']['plasma']['linewidth']        = 1.129e+14
    
    # -------------------------------------------------------------------------
    ## Load focused extended source properties 
    """
    config['sources']['focused']['class_name']      = 'XicsrtSourceFocused'
    config['sources']['focused']['filter_list']     = ['XicsrtBundleFilterSightline']
    config['sources']['focused']['use_poisson']     = True
    config['sources']['focused']['do_monochrome']   = False
    config['sources']['focused']['temperature']     = 1000
    config['sources']['focused']['intensity']       = 1e7
    config['sources']['focused']['velocity']        = np.array([0.0,0.0,0.0])

    config['sources']['focused']['origin']          = np.array([0.0, 6.2, 0.5])
    config['sources']['focused']['zaxis']           = np.array([0.0, 0.0, 1.0])
    config['sources']['focused']['xaxis']           = np.array([1.0, 0.0, 0.0])
    config['sources']['focused']['target']          = np.array([1.0, 0.0, 0.0])
    config['sources']['focused']['width']           = 4.0
    config['sources']['focused']['height']          = 4.0
    config['sources']['focused']['depth']           = 7.5
    
    config['sources']['focused']['spread']           = 1.0
    config['sources']['focused']['mass_number']      = 131.293
    config['sources']['focused']['wavelength']       = 2.19 #2.7203 
    config['sources']['focused']['linewidth']        = 1.129e+14
    """
    
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
    config['optics']['crystal']['class_name']       = 'XicsrtOpticCrystalSpherical'
    config['optics']['crystal']['do_bragg_check']   = True
    config['optics']['crystal']['do_miss_check']    = True
    config['optics']['crystal']['rocking_type']     = 'FILE'
    config['optics']['crystal']['use_meshgrid']     = False
    config['optics']['crystal']['rocking_mix']      = 1.0
    config['optics']['crystal']['rocking_sigma_file'] = inpath + 'rocking_curve_germanium_sigma.txt'
    config['optics']['crystal']['rocking_pi_file']  = inpath + 'rocking_curve_germanium_pi.txt'
    
    """Crystal settings
    'crystal_spacing'is the inter-atomic spacing (angstrom)
    'reflectivity'   is the maximum reflectivity used for 'STEP' and 'GAUSS'
    'rocking_type'   is the rocking curve FWHM (rad) used for 'STEP' and 'GAUSS'
    'pixel_size'     is the size of the pixels used by the image generator
    'therm_expand'   is the thermal expansion coefficient (1/kelvin)
    """
    config['optics']['crystal']['crystal_spacing']  = 1.42 #1.7059 
    config['optics']['crystal']['reflectivity']     = 1.0
    config['optics']['crystal']['rocking_fwhm']     = 90.30e-6
    config['optics']['crystal']['pixel_size']       = 0.0001
    #config['optics']['crystal']['thermal_expand']   = 5.9e-6
    
    """Geometry Settings
    crystal 'width' and 'height' (meters) only matter when 'use_trimesh' is False
    'radius' is the crystal's radius of curvature (meters)
    """
    config['optics']['crystal']['origin']          = np.array([0.0, 0.0, 0.0])
    config['optics']['crystal']['zaxis']           = np.array([0.0, 0.0, 0.0])
    config['optics']['crystal']['xaxis']           = np.array([0.0, 0.0, 0.0])
    
    config['optics']['crystal']['width']           = 0.040
    config['optics']['crystal']['height']          = 0.050
    config['optics']['crystal']['radius']          = 2.200 #2.400 
    
    """
    Rocking curve FWHM:  90.30 urad
    Darwin Curve, sigma: 48.070 urad
    Darwin Curve, pi:    14.043 urad
    Taken from XoP
    """
    
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
    config['optics']['graphite']['use_meshgrid']      = True
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
    config['optics']['graphite']['reflectivity']      = 1.0
    config['optics']['graphite']['mosaic_spread']     = 0.5
    config['optics']['graphite']['rocking_fwhm']      = 2620e-6
    config['optics']['graphite']['pixel_size']        = 0.001
    #config['optics']['graphite']['therm_expand']      = 20e-6
    
    """Geometry Settings
    graphite 'width' and 'height' (meters) only matter when 'use_meshgrid' is False
    """
    config['optics']['graphite']['origin']            = np.array([1.0, 0.0, 0.0])
    config['optics']['graphite']['zaxis']             = np.array([0.0, 0.0, 0.0])
    config['optics']['graphite']['xaxis']             = np.array([0.0, 0.0, 1.0])
    
    config['optics']['graphite']['width']             = 0.150
    config['optics']['graphite']['height']            = 0.150
    
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
    
    config['optics']['detector']['pixel_size']        = 0.000172
    #config['optics']['detector']['pixel_width']       = 195
    #config['optics']['detector']['pixel_height']      = 1475
    config['optics']['detector']['width']             = (195  * config['optics']['detector']['pixel_size'])
    config['optics']['detector']['height']            = (1475 * config['optics']['detector']['pixel_size'])

    # -------------------------------------------------------------------------
    ## Load scenario
    """
    Convert the numbers given in the XICS presentations into useful information.
    When copying values from the XICS presentations, please place them here.
    
    Chords 0 and 1 use Xe44+, while chords 2 and 3 use Xe55+
    graphite_corners[chord number, sight number, corner number, 3D coordinates]
    crystal_corners[chord number, corner number, 3D coordinates]
    detector_centers[chord number, 3D coordinates]
    """
    graphite_corners = np.empty([4,3,4,3], dtype = np.float64)
    crystal_corners  = np.empty([4,4,3], dtype = np.float64)
    detector_centers = np.empty([4,3], dtype = np.float64)
    chord            = 2
    sight            = 0
    
    graphite_corners[0,0,0,:] = np.array([048.5, 9324.9, -619.5])
    graphite_corners[0,0,1,:] = np.array([018.7, 9312.7, -619.4])
    graphite_corners[0,0,2,:] = np.array([017.4, 9400.8, -660.3])
    graphite_corners[0,0,3,:] = np.array([047.4, 9413.0, -660.5])
    graphite_corners[0,1,0,:] = np.array([094.6, 9224.8, -618.5])
    graphite_corners[0,1,1,:] = np.array([065.1, 9205.1, -618.2])
    graphite_corners[0,1,2,:] = np.array([063.3, 9290.9, -659.2])
    graphite_corners[0,1,3,:] = np.array([093.1, 9310.6, -659.4])
    graphite_corners[0,2,0,:] = np.array([140.6, 9166.1, -617.9])
    graphite_corners[0,2,1,:] = np.array([111.4, 9137.9, -617.5])
    graphite_corners[0,2,2,:] = np.array([109.3, 9219.6, -658.4])
    graphite_corners[0,2,3,:] = np.array([138.7, 9247.8, -658.7])
    graphite_corners[1,0,0,:] = np.array([057.2, 9787.3,  588.4])
    graphite_corners[1,0,1,:] = np.array([027.3, 9776.8,  590.2])
    graphite_corners[1,0,2,:] = np.array([028.9, 9682.3,  566.3])
    graphite_corners[1,0,3,:] = np.array([058.6, 9692.8,  564.5])
    graphite_corners[1,1,0,:] = np.array([102.9, 9575.8,  625.7])
    graphite_corners[1,1,1,:] = np.array([073.2, 9558.5,  628.7])
    graphite_corners[1,1,2,:] = np.array([075.2, 9465.8,  604.4])
    graphite_corners[1,1,3,:] = np.array([104.8, 9483.1,  601.5])
    graphite_corners[1,2,0,:] = np.array([150.6, 9415.4,  654.1])
    graphite_corners[1,2,1,:] = np.array([121.1, 9388.9,  658.6])
    graphite_corners[1,2,2,:] = np.array([123.6, 9300.3,  633.6])
    graphite_corners[1,2,3,:] = np.array([152.8, 9326.8,  629.1])
    graphite_corners[2,0,0,:] = np.array([053.1, 9252.0, -669.9])
    graphite_corners[2,0,1,:] = np.array([018.6, 9222.5, -669.9])
    graphite_corners[2,0,2,:] = np.array([016.8, 9333.6, -709.9])
    graphite_corners[2,0,3,:] = np.array([051.5, 9362.9, -709.9])
    graphite_corners[2,1,0,:] = np.array([100.0, 9151.7, -670.0])
    graphite_corners[2,1,1,:] = np.array([066.1, 9102.3, -669.9])
    graphite_corners[2,1,2,:] = np.array([063.9, 9204.1, -710.0])
    graphite_corners[2,1,3,:] = np.array([098.0, 9253.2, -709.9])
    graphite_corners[2,2,0,:] = np.array([145.8, 9141.4, -670.0])
    graphite_corners[2,2,1,:] = np.array([112.5, 9075.5, -669.9])
    graphite_corners[2,2,2,:] = np.array([110.2, 9164.5, -710.0])
    graphite_corners[2,2,3,:] = np.array([143.6, 9230.2, -709.9])
    graphite_corners[3,0,0,:] = np.array([069.2, 9228.1,  633.8])
    graphite_corners[3,0,1,:] = np.array([034.9, 9181.4,  641.5])
    graphite_corners[3,0,2,:] = np.array([036.9, 9073.4,  618.6])
    graphite_corners[3,0,3,:] = np.array([071.0, 9120.3,  611.1])
    graphite_corners[3,1,0,:] = np.array([116.8, 9152.1,  646.4])
    graphite_corners[3,1,1,:] = np.array([083.4, 9079.2,  658.4])
    graphite_corners[3,1,2,:] = np.array([085.4, 8993.6,  631.8])
    graphite_corners[3,1,3,:] = np.array([118.6, 9066.8,  620.0])
    graphite_corners[3,2,0,:] = np.array([163.5, 9122.5,  651.2])
    graphite_corners[3,2,1,:] = np.array([130.8, 9034.8,  665.6])
    graphite_corners[3,2,2,:] = np.array([132.7, 8971.5,  635.4])
    graphite_corners[3,2,3,:] = np.array([165.2, 9059.3,  621.2])
    
    crystal_corners[0,0,:]    = np.array([-055.0, 17704.9, -714.9])
    crystal_corners[0,1,:]    = np.array([-105.0, 17704.0, -714.9])
    crystal_corners[0,2,:]    = np.array([-105.0, 17680.6, -745.1])
    crystal_corners[0,3,:]    = np.array([-055.0, 17681.5, -745.1])
    crystal_corners[1,0,:]    = np.array([-055.0, 17585.6, -793.5])
    crystal_corners[1,1,:]    = np.array([-105.0, 17584.6, -793.5])
    crystal_corners[1,2,:]    = np.array([-105.0, 17559.8, -816.5])
    crystal_corners[1,3,:]    = np.array([-055.0, 17560.9, -816.5])
    crystal_corners[2,0,:]    = np.array([-075.0, 18076.2, -674.1])
    crystal_corners[2,1,:]    = np.array([-125.0, 18075.2, -674.1])
    crystal_corners[2,2,:]    = np.array([-125.0, 18049.1, -705.9])
    crystal_corners[2,3,:]    = np.array([-075.0, 18050.1, -705.9])
    crystal_corners[3,0,:]    = np.array([-075.0, 17997.3, -817.7])
    crystal_corners[3,1,:]    = np.array([-125.0, 17996.2, -817.7])
    crystal_corners[3,2,:]    = np.array([-125.0, 17968.0, -842.3])
    crystal_corners[3,3,:]    = np.array([-075.0, 17969.1, -842.3])
    
    detector_centers[0,:]     = np.array([-68.8, 17193.2, 1117.6])
    detector_centers[1,:]     = np.array([-67.4, 17381.1, 1099.3])
    detector_centers[2,:]     = np.array([-93.2, 17732.9,  977.6])
    detector_centers[3,:]     = np.array([-94.0, 17928.6,  869.1])
    
    config = setup_real_scenario(config, graphite_corners, crystal_corners,
                                 detector_centers, chord, sight)
    
    return config
    
def get_config_multi(configurations):
    config_multi = dict()
    for ii in range(configurations):
        config = get_config()
        config_multi[str(ii)] = config
        
    return config_multi

def setup_real_scenario(config, g_corners, c_corners, d_centers, chord, sight):
    """
    Rather than generating the entire scenario from scratch, this scenario
    generator takes the information provided by the ITER XICS team and fills in
    all the blanks to produce a complete description of the spectrometer layout
    """
    ## Unpack variables and convert to meters
    g_corners  = g_corners[chord,sight] / 1000
    c_corners  = c_corners[chord]       / 1000
    d_centers  = d_centers[chord]       / 1000

    #calculate geometric properties of all meshes
    g_basis      = np.zeros([3,3], dtype = np.float64)
    c_basis      = np.zeros([3,3], dtype = np.float64)
    d_basis      = np.zeros([3,3], dtype = np.float64)
    
    g_origin     = np.mean(g_corners, axis = 0)
    c_origin     = np.mean(c_corners, axis = 0)
    #d_origin     = np.mean(d_corners, axis = 0)
    d_origin     = d_centers
    
    g_width      = np.linalg.norm(g_corners[0] - g_corners[1])
    c_width      = np.linalg.norm(c_corners[0] - c_corners[1])
    #d_width      = np.linalg.norm(d_corners[0] - d_corners[1])
    
    g_height     = np.linalg.norm(g_corners[0] - g_corners[3])
    c_height     = np.linalg.norm(c_corners[0] - c_corners[3])
    #d_height     = np.linalg.norm(d_corners[0] - d_corners[3])
    
    g_basis[0,:] = g_corners[0] - g_corners[1]
    c_basis[0,:] = c_corners[0] - c_corners[1]
    #d_basis[0,:] = d_corners[0] - d_corners[1]
    
    g_basis[0,:]/= g_width
    c_basis[0,:]/= c_width
    #d_basis[0,:]/= d_width
    
    g_basis[1,:] = g_corners[0] - g_corners[3]
    c_basis[1,:] = c_corners[0] - c_corners[3]
    #d_basis[1,:] = d_corners[0] - d_corners[3]
    
    g_basis[1,:]/= g_height
    c_basis[1,:]/= c_height
    #d_basis[1,:]/= d_height
    
    g_basis[2,:] = np.cross(g_basis[0,:], g_basis[1,:])
    c_basis[2,:] = np.cross(c_basis[0,:], c_basis[1,:])
    #d_basis[2,:] = np.cross(d_basis[0,:], d_basis[1,:])
    d_basis[2,:] = c_origin - d_origin
    d_basis[1,:] = np.array([-1.0, 0.0, 0.0])
    d_basis[0,:] = np.cross(d_basis[2,:], d_basis[1,:])

    
    g_basis[2,:]/= np.linalg.norm(g_basis[2,:])
    c_basis[2,:]/= np.linalg.norm(c_basis[2,:])
    d_basis[2,:]/= np.linalg.norm(d_basis[2,:])
    
    d_basis[0,:]/= np.linalg.norm(d_basis[0,:])
    
    #calculate the graphite pre-reflector's sightline of the plasma
    #start with the crystal-graphite vector, normalize, and reflect it
    sightline    = g_origin - c_origin
    sightline   /= np.linalg.norm(sightline)
    sightline   -= 2 * np.dot(sightline, g_basis[2,:]) * g_basis[2,:]
    
    #triangulate the graphite
    config['optics']['graphite']['mesh_points'] = g_corners
    config['optics']['graphite']['mesh_faces']  = np.array([[2,1,0],[0,3,2]])
    config['optics']['crystal']['mesh_points'] = None
    config['optics']['crystal']['mesh_faces']  = None
    
    ## Repack variables
    #config['sources']['focused']['target']        = g_origin
    config['sources']['plasma']['target']         = g_origin
    config['filters']['sightline']['origin']      = g_origin
    config['filters']['sightline']['direction']   = sightline
    
    config['optics']['graphite']['origin']        = g_origin
    config['optics']['graphite']['zaxis']         = g_basis[2,:]
    config['optics']['graphite']['xaxis']         = g_basis[0,:]
    
    config['optics']['crystal']['origin']         = c_origin
    config['optics']['crystal']['zaxis']          = c_basis[2,:]
    config['optics']['crystal']['xaxis']          = c_basis[0,:]
    config['optics']['crystal']['width']          = c_width
    config['optics']['crystal']['height']         = c_height
        
    config['optics']['detector']['origin']        = d_origin
    config['optics']['detector']['zaxis']         = d_basis[2,:]
    config['optics']['detector']['xaxis']         = d_basis[0,:]
    
    return config

## Run the scripts in order (TEMPORARY - Find a better place to put this code)
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code/analysis/xrcscore_eugene/')

from xicsrt import xicsrt_raytrace, xicsrt_input

visuals = True
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
    xicsrt_visualizer.visualize_layout(config)
    