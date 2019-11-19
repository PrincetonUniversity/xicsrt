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
    config['graphite_input']  = OrderedDict()
    config['crystal_input']   = OrderedDict()
    config['detector_input']  = OrderedDict()

    # -------------------------------------------------------------------------
    # General raytracer properties

    # set ideal_geometry to False to enable offsets, tilts, and thermal expand
    # set backwards_raytrace to True to swap the detector and source
    # set do_visualizations to toggle the visualizations on or off
    # set do_savefiles to toggle whether the program saves .tif files
    # set do_image_analysis to toggle whether the visualizer performs .tif analysis
    # set do_bragg_checks to False to make the optics into perfect X-Ray mirrors
    # set do_miss_checks to False to prevent optics from masking rays that miss
    # change the random seed to alter the random numbers generated
    # possible scenarios include 'MODEL', 'PLASMA', 'BEAM', 'CRYSTAL', 'GRAPHITE', 'SOURCE'
    # possible rocking curve types include 'STEP', 'GAUSS', and 'FILE'

    config['general_input']['number_of_rays']     = int(1e7)
    config['general_input']['number_of_runs']     = 1

    config['general_input']['output_path']        = '/Users/Eugene/PPPL_python_project1/xics_rt_code/results/'
    config['general_input']['output_suffix']      = '.tif'
    config['general_input']['ideal_geometry']     = True
    config['general_input']['backwards_raytrace'] = False
    config['general_input']['do_visualizations']  = True
    config['general_input']['do_savefiles']       = True
    config['general_input']['do_image_analysis']  = False
    config['general_input']['random_seed']        = 123456
    config['general_input']['scenario']           = 'PLASMA'
    config['general_input']['system']             = 'w7x_ar16'
    config['general_input']['shot']               = 180707017

    # Temperature that the optical elements will be cooled to (kelvin)
    config['general_input']['xics_temp'] = 273.0

    # -------------------------------------------------------------------------
    ## Load plasma properties

    # bundle_count
    #   Increase to improve both plasma modeling and raytracing accuracy.
    # bundle_volume
    #   Increase to improve raytracing accuracy
    #   Decrease to improve plasma modeling
    #
    #   This currently only has an effect of bundle_type is 'voxel'.

    config['plasma_input']['max_rays']            = config['general_input']['number_of_rays']
    config['plasma_input']['position']            = np.array([0.0, 0.0, 0.0])
    config['plasma_input']['normal']              = np.array([1.0, 0.0, 0.0])
    config['plasma_input']['orientation']         = np.array([0.0, 0.0, 1.0])
    config['plasma_input']['target']              = np.array([1.0, 0.0, 0.0])
    config['plasma_input']['width']               = 9
    config['plasma_input']['height']              = 0.1
    config['plasma_input']['depth']               = 0.1
    
    config['plasma_input']['major_radius']        = 6.2
    config['plasma_input']['minor_radius']        = 2.0
    config['plasma_input']['temperature_data']    = '../xicsrt/plasma_temperature.txt'
    config['plasma_input']['emissivity_data']     = '../xicsrt/plasma_emissivity_xe44.txt'
    
    config['plasma_input']['space_resolution']    = 0.01
    config['plasma_input']['time_resolution']     = 0.01
    config['plasma_input']['bundle_count']        = 1000000
    config['plasma_input']['bundle_type']         = 'point'
    config['plasma_input']['profile_type']        = 'data'

    config['plasma_input']['spread']              = 2.0       # Angular spread (degrees)
    config['plasma_input']['temperature']         = 1000      # Ion temperature (eV)
    config['plasma_input']['emissivity']          = 1e16      # Emissivity (photons m^-3 s^-1)
    config['plasma_input']['mass']                = 131.293   # Xenon mass (AMU)
    config['plasma_input']['wavelength']          = 2.7203    # Line location (angstroms)
    config['plasma_input']['linewidth']           = 1.129e+14 # Natural linewith (1/s)

    # -------------------------------------------------------------------------
    ## Load source properties
    # Xe44+ w line
    # Additional information on Xenon spectral lines can be found on nist.gov
    config['source_input']['intensity']           = config['general_input']['number_of_rays']
    config['source_input']['position']            = np.array([0.0, 0.0, 0.0])
    config['source_input']['normal']              = np.array([1.0, 0.0, 0.0])
    config['source_input']['orientation']         = np.array([0.0, 0.0, 1.0])
    config['source_input']['target']              = np.array([1.0, 0.0, 0.0])

    config['source_input']['spread']              = 1.0       #Angular spread (degrees)
    config['source_input']['temp']                = 1000      #Ion temperature (eV)
    config['source_input']['mass']                = 131.293   # Xenon mass (AMU)
    config['source_input']['wavelength']          = 2.7203    # Line location (angstroms)
    config['source_input']['linewidth']           = 1.129e+14 # Natural linewith (1/s)

    #These values are arbitrary for now. Set to 0.0 for point source
    config['source_input']['width']               = 0.050
    config['source_input']['height']              = 0.050
    config['source_input']['depth']               = 0.050

    # -------------------------------------------------------------------------
    ## Load spherical crystal properties

    # Rocking curve FWHM:  90.30 urad
    # Darwin Curve, sigma: 48.070 urad
    # Darwin Curve, pi:    14.043 urad
    # Taken from XoP

    config['crystal_input']['position']           = [0.0, 0.0, 0.0]
    config['crystal_input']['normal']             = [0.0, 0.0, 0.0]
    config['crystal_input']['orientation']        = [0.0, 0.0, 0.0]

    config['crystal_input']['width']              = 0.040
    config['crystal_input']['height']             = 0.050
    config['crystal_input']['curvature']          = 1.200

    config['crystal_input']['spacing']            = 1.7059
    config['crystal_input']['reflectivity']       = 1
    config['crystal_input']['rocking_curve']      = 90.30e-6
    config['crystal_input']['pixel_scaling']      = int(200)

    config['crystal_input']['therm_expand']       = 5.9e-6
    config['crystal_input']['sigma_data']         = '../xicsrt/rocking_curve_germanium_sigma.txt'
    config['crystal_input']['pi_data']            = '../xicsrt/rocking_curve_germanium_pi.txt'
    config['crystal_input']['mix_factor']         = 1.0

    config['crystal_input']['do_bragg_checks']    = True
    config['crystal_input']['do_miss_checks']     = True
    config['crystal_input']['rocking_curve_type'] = "FILE"

    # --------------------------------------------------------------------------
    ## Load mosaic graphite properties
    
    # HOPG Crystallite Rocking Curve FWHM: 2620 urad (0.15 degrees)
    # Taken from Ohler et al. “X-ray topographic determination of the granular 
    # structure in a graphite mosaic crystal: a three-dimensional reconstruction”

    config['graphite_input']['position']          = [0.0, 0.0, 0.0]
    config['graphite_input']['normal']            = [0.0, 0.0, 0.0]
    config['graphite_input']['orientation']       = [0.0, 0.0, 0.0]

    config['graphite_input']['width']             = 0.030
    config['graphite_input']['height']            = 0.040

    config['graphite_input']['reflectivity']      = 1
    config['graphite_input']['mosaic_spread']     = 0.5
    config['graphite_input']['spacing']           = 3.35
    config['graphite_input']['rocking_curve']     = 2620e-6
    config['graphite_input']['pixel_scaling']     = int(200)

    config['graphite_input']['therm_expand']      = 20e-6
    config['graphite_input']['sigma_data']        = '../xicsrt/rocking_curve_graphite_sigma.txt'
    config['graphite_input']['pi_data']           = '../xicsrt/rocking_curve_graphite_pi.txt'
    config['graphite_input']['mix_factor']        = 1.0

    config['graphite_input']['do_bragg_checks']   = True
    config['graphite_input']['do_miss_checks']    = True
    config['graphite_input']['rocking_curve_type']= "gaussian"

    # -------------------------------------------------------------------------
    ## Load detector properties

    config['detector_input']['position']          = [0.0, 0.0, 0.0]
    config['detector_input']['normal']            = [0.0, 0.0, 0.0]
    config['detector_input']['orientation']       = [0.0, 0.0, 0.0]

    config['detector_input']['pixel_size']        = 0.000172
    config['detector_input']['horizontal_pixels'] = 195
    config['detector_input']['vertical_pixels']   = 1475
    config['detector_input']['width']             = (config['detector_input']['horizontal_pixels']
                                                    * config['detector_input']['pixel_size'])
    config['detector_input']['height']            = (config['detector_input']['vertical_pixels']
                                                    * config['detector_input']['pixel_size'])

    config['detector_input']['do_miss_checks']    = True

    # -------------------------------------------------------------------------
    ## Load scenario properties
    
    config['scenario_input']['source_graphite_dist']  = 10
    config['scenario_input']['graphite_crystal_dist'] = 8.5
    config['scenario_input']['crystal_detector_dist'] = None

    return config

def get_config_multi(configurations):
    config_multi = list()
    for i in range(configurations):
        config = get_config()
        config_multi.append(config)
        
    return config_multi

## Run the scripts in order (TEMPORARY - Find a better place to put this code)
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code') 

import logging
import json

from xicsrt.xics_rt_initialize import initialize, initialize_multi
from xicsrt.xics_rt_run import run, run_multi

runtype = 'single'
logging.info('Starting Ray-Trace Runs...')

if runtype == 'single':
    config = get_config()
    config = initialize(config)
    output, meta = run(config)
        
if runtype == 'multi':
    config_multi = get_config_multi(10)
    config_multi = initialize_multi(config_multi)
    output, meta = run_multi(config_multi)
        
if runtype == 'save':
    config_multi = get_config_multi(10)
    config_multi = initialize_multi(config_multi)
    # Convert all numpy arrays into json-recognizable lists
    for configuration in range(len(config_multi)):
        for element in config_multi[configuration]:
            for key in config_multi[configuration][element]:
                if type(config_multi[configuration][element][key]) is np.ndarray:
                    config_multi[configuration][element][key] = (
                            config_multi[configuration][element][key].tolist())
                
    with open('xicsrt_input.json', 'w') as input_file:
        json.dump(config_multi ,input_file, indent = 1, sort_keys = True)
        print('xicsrt_input.json saved!')
        
if runtype == 'load':
    try:
        with open('xicsrt_input.json', 'r') as input_file:
            config_multi = json.load(input_file)

    except FileNotFoundError:
        print('xicsrt_input.json not found!')
        
    # Convert all lists back into numpy arrays
    for configuration in range(len(config_multi)):
        for element in config_multi[configuration]:
            for key in config_multi[configuration][element]:
                if type(config_multi[configuration][element][key]) is list:
                    config_multi[configuration][element][key] = np.array(
                            config_multi[configuration][element][key])    
    output, meta = run_multi(config_multi)
