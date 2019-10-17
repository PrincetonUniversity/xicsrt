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
This script manages the xicsrt_input.txt input file. It comes with a default 
set of variables and direct access to the scenario generator xics_rt_scenarios.
While it can't directly run ray-traces, it is nonetheless a vital part of the 
ray-tracing pipeline.
"""
#%% IMPORTS
print('Importing Packages...')

## This is only needed since I have not actually installed xicsrt
#YVY: Changed filepath
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code')             

## Start Logging
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

## Start Profiling
from xicsrt.util import profiler
profiler.startProfiler()

profiler.start('Total Time')
profiler.start('Import Time')

## Begin normal imports
import numpy as np
from collections import OrderedDict

import json

from xicsrt.xics_rt_scenarios import bragg_angle, source_location_bragg
from xicsrt.xics_rt_scenarios import setup_beam_scenario, setup_crystal_test
from xicsrt.xics_rt_scenarios import setup_graphite_test, setup_source_test

## MirPyIDL imports
import mirutil.hdf5

profiler.stop('Import Time')
#%% OPEN
"""
Check to see if xicsrt_input.json exists. If no, create it. If yes, read it.
"""
try:
    with open('xicsrt_input.json', 'r') as input_file:
        xicsrt_input = json.load(input_file)
except FileNotFoundError:
    xicsrt_input = list()
    
#%% INPUTS
"""
Input Section
Contains information about detectors, crystals, sources, etc.

Crystal and detector parameters are taken from the W7-X Op1.2 calibration
This can be found in the file w7x_ar16.cerdata for shot 180707017
"""
print('Initializing Variables...')
profiler.start('Input Setup Time')

general_input   = OrderedDict()
source_input    = OrderedDict()
crystal_input   = OrderedDict()
graphite_input  = OrderedDict()
detector_input  = OrderedDict()

## Set up general properties about the raytracer, including random seed
"""
set ideal_geometry to False to enable offsets, tilts, and thermal expand
set backwards_raytrace to True to swap the detector and source
set do_visualizations to toggle the visualizations on or off
set do_savefiles to toggle whether the program saves .tif files
set do_image_analysis to toggle whether the visualizer performs .tif analysis
set do_bragg_checks to False to make the optics into perfect X-Ray mirrors
set do_miss_checks to False to prevent optics from masking rays that miss
change the random seed to alter the random numbers generated
possible scenarios include 'MODEL', 'BEAM', 'CRYSTAL', 'GRAPHITE', 'SOURCE'
possible rocking curve types include 'STEP', 'GAUSS', and 'FILE'
"""
general_input['ideal_geometry']     = True
general_input['backwards_raytrace'] = False
general_input['do_visualizations']  = True
general_input['do_savefiles']       = True
general_input['do_image_analysis']  = True
general_input['do_bragg_checks']    = True
general_input['do_miss_checks']     = True
general_input['random_seed']        = 1234567
general_input['scenario']           = 'MODEL'
general_input['rocking_curve_type'] = 'FILE'
general_input['system']             = 'w7x_ar16'
general_input['shot']               = 180707017

# Temperature that the optical elements will be cooled to (kelvin)
general_input['xics_temp']          = 273.0

# Number of rays to launch
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
source_input['intensity']           = int(1e7)
general_input['number_of_runs']     = 1

# Xenon mass in AMU
source_input['mass']                = 131.293

# Line location (angstroms) and natural linewith (1/s)
# Xe44+ w line
# Additional information on Xenon spectral lines can be found on nist.gov
source_input['wavelength']          = 2.7203
source_input['linewidth']           = 1.129e+14

"""
Rocking curve FWHM in radians
This is taken from x0h for quartz 1,1,-2,0
Darwin Curve, sigma: 48.070 urad
Darwin Curve, pi:    14.043 urad
Graphite Rocking Curve FWHM in radians
Taken from XOP: 8765 urad
"""

## Load spherical crystal properties from hdf5 file
config_dict = mirutil.hdf5.hdf5ToDict(
        '/Users/Eugene/PPPL_python_project1/w7x_ar16_180707017_geometry.hdf5')

crystal_input['position']           = config_dict['CRYSTAL_LOCATION']
crystal_input['normal']             = config_dict['CRYSTAL_NORMAL']
crystal_input['orientation']        = config_dict['CRYSTAL_ORIENTATION']

crystal_input['width']              = 0.040
crystal_input['height']             = 0.010
crystal_input['curvature']          = 1.200

crystal_input['spacing']            = 1.7059
crystal_input['reflectivity']       = 1
crystal_input['rocking_curve']      = 90.30e-6
crystal_input['pixel_scaling']      = int(200)

crystal_input['therm_expand']       = 5.9e-6
crystal_input['sigma_data']         = '../xicsrt/rocking_curve_germanium_sigma.txt'
crystal_input['pi_data']            = '../xicsrt/rocking_curve_germanium_pi.txt'
crystal_input['mix_factor']         = 1.0

crystal_input['do_bragg_checks']    = True
crystal_input['do_miss_checks']     = True

## Load mosaic graphite properties
graphite_input['position']          = config_dict['CRYSTAL_LOCATION']
graphite_input['normal']            = config_dict['CRYSTAL_NORMAL']
graphite_input['orientation']       = config_dict['CRYSTAL_ORIENTATION']

graphite_input['width']             = 0.030
graphite_input['height']            = 0.040

graphite_input['reflectivity']      = 1
graphite_input['mosaic_spread']     = 0.5
graphite_input['spacing']           = 3.35
graphite_input['rocking_curve']     = 8765e-6
graphite_input['pixel_scaling']     = int(200)

graphite_input['therm_expand']      = 20e-6
graphite_input['sigma_data']        = '../xicsrt/rocking_curve_graphite_sigma.txt'
graphite_input['pi_data']           = '../xicsrt/rocking_curve_graphite_pi.txt'
graphite_input['mix_factor']        = 1.0

graphite_input['do_bragg_checks']   = True
graphite_input['do_miss_checks']    = True

## Load detector properties
detector_input['position']          = config_dict['DETECTOR_LOCATION']
detector_input['normal']            = config_dict['DETECTOR_NORMAL']
detector_input['orientation']       = config_dict['DETECTOR_ORIENTATION']

detector_input['pixel_size']        = 0.000172
detector_input['horizontal_pixels'] = int(config_dict['X_SIZE'])
detector_input['vertical_pixels']   = int(config_dict['Y_SIZE'])
detector_input['width']             = (detector_input['horizontal_pixels'] 
                                    * detector_input['pixel_size'])
detector_input['height']            = (detector_input['vertical_pixels'] 
                                    * detector_input['pixel_size'])

detector_input['do_miss_checks']    = True

## Load source properties
source_input['position']            = np.array([0, 0, 0])
source_input['normal']              = np.array([0, 1, 0])
source_input['orientation']         = np.array([0, 0, 1])
source_input['target']              = crystal_input['position']

#Angular spread of source in degrees
#This needs to be matched to the source distance and crystal size
source_input['spread']              = 1.0
#Ion temperature in eV
source_input['temp']                = 1000

#These values are arbitrary for now. Set to 0.0 for point source
source_input['width']               = 0.1
source_input['height']              = 0.1
source_input['depth']               = 0.1

profiler.stop('Input Setup Time')

#%% SCENARIO
"""
Each of these scenarios corresponds to a script located in xics_rt_tools.py 
which assembles the optical elements into a specific configuration based on
input parameters
"""
print('Arranging Scenarios...')
profiler.start('Scenario Setup Time')

# Analytical solutions for spectrometer geometry involving crystal focus
crystal_input['bragg'] = bragg_angle(source_input['wavelength'], crystal_input['spacing'])
crystal_input['meridi_focus']  = crystal_input['curvature'] * np.sin(crystal_input['bragg'])
crystal_input['sagitt_focus']  = - crystal_input['meridi_focus'] / np.cos(2 * crystal_input['bragg'])
graphite_input['bragg'] = bragg_angle(source_input['wavelength'], graphite_input['spacing'])
crystal_input['effective_width'] = (2 * crystal_input['meridi_focus'] *
        np.sin(graphite_input['rocking_curve'] / 2) * (
        1 / np.sin(crystal_input['bragg'] + graphite_input['rocking_curve'] / 2) +
        1 / np.sin(crystal_input['bragg'] - graphite_input['rocking_curve'] / 2)))

## Set up a legacy beamline scenario
if general_input['scenario'] == 'LEGACY':
    source_input, crystal_input, detector_input, = source_location_bragg(
    source_input, crystal_input, detector_input,
    3.5,                                    # Distance from Crystal
    0,                                      # Meridional Offset
    0,                                      # Saggital Offset
    )
    
    source_input['target'] = crystal_input['position']
    source_input['normal'] = (crystal_input['position'] - source_input['position'])
    source_input['normal']/=  np.linalg.norm(source_input['normal'])

    #This direction is rather abitrary and is not (in general)
    #in the meridional direction.
    source_input['orientation'] = np.cross(np.array([0, 0, 1]), source_input['normal'])
    source_input['orientation'] /= np.linalg.norm(source_input['orientation'])

## Set up a beamline test scenario
elif general_input['scenario'] == 'BEAM' or general_input['scenario'] == 'MODEL':
    [general_input, source_input, graphite_input, crystal_input, detector_input] = setup_beam_scenario(
     general_input, source_input, graphite_input, crystal_input, detector_input,
     1.000,                                 #source-graphite distance
     8.500,                                 #graphite-crystal distance
     crystal_input['meridi_focus']        , #crystal-detector distance
     np.array([0,0,0], dtype = np.float64), #graphite offset (meters)
     np.array([0,0,0], dtype = np.float64), #graphite tilt (radians)
     np.array([0,0,0], dtype = np.float64), #crystal offset (meters)
     np.array([0,0,0], dtype = np.float64), #crystal tilt (radians)
     np.array([0,0,0], dtype = np.float64), #detector offset (meters)
     np.array([0,0,0], dtype = np.float64), #detector tilt (radians)
     )

## Set up a crystal test scenario
elif general_input['scenario'] == 'CRYSTAL':
    [source_input, crystal_input, detector_input] = setup_crystal_test(
     source_input, crystal_input, detector_input, 
     8.500 + 1.000                        , #source-crystal distance
     crystal_input['meridi_focus']        , #crystal-detector distance
     np.array([0,0,0], dtype = np.float64), #crystal offset (meters)
     00 * np.pi / 180                     , #crystal tilt (radians)
     )                      
    
    graphite_input['position']     = crystal_input['position']
    graphite_input['normal']       = crystal_input['normal']
    graphite_input['orientation']  = crystal_input['orientation']
    
## Set up a graphite test scenario
elif general_input['scenario'] == 'GRAPHITE':
    [source_input, graphite_input, detector_input] = setup_graphite_test(
     source_input, graphite_input, detector_input, 
     1,                                     #source-graphite distance
     1,                                     #graphite-detector distance
     )
    
    crystal_input['position']       = graphite_input['position']
    crystal_input['normal']         = graphite_input['normal']
    crystal_input['orientation']    = graphite_input['orientation']
    
## Set up a source test scenario
elif general_input['scenario'] == 'SOURCE':
    [source_input, detector_input] = setup_source_test(
     source_input, detector_input, 1        #source-detector distance
     )
    
## Backwards raytracing involves swapping the source and detector
if general_input['backwards_raytrace']:
    swap_position   = source_input['position']
    swap_orientation= source_input['orientation']
    swap_normal     = source_input['normal']
    
    source_input['position']    = detector_input['position']
    source_input['orientation'] = detector_input['orientation']
    source_input['normal']      = detector_input['normal']
    
    detector_input['position']     = swap_position
    detector_input['orientation']  = swap_orientation
    detector_input['normal']       = swap_normal
    
## Simulate linear thermal expansion
# This is calculated AFTER spectrometer geometry setup to simulate non-ideal conditions
if general_input['ideal_geometry'] is False:
    crystal_input['spacing']  *= 1 + crystal_input['therm_expand']  * (general_input['xics_temp'] - 273)
    graphite_input['spacing'] *= 1 + graphite_input['therm_expand'] * (general_input['xics_temp'] - 273)
    
profiler.stop('Scenario Setup Time')

#%% SAVE
# Convert all numpy arrays into json-recognizable lists
print('Saving Dictionaries...')
profiler.start('Dictionary Save Time')
source_input['position'] = source_input['position'].tolist()
source_input['normal'] = source_input['normal'].tolist()
source_input['orientation'] = source_input['orientation'].tolist()
source_input['target'] = source_input['target'].tolist()

crystal_input['position'] = crystal_input['position'].tolist()
crystal_input['normal'] = crystal_input['normal'].tolist()
crystal_input['orientation'] = crystal_input['orientation'].tolist()

graphite_input['position'] = graphite_input['position'].tolist()
graphite_input['normal'] = graphite_input['normal'].tolist()
graphite_input['orientation'] = graphite_input['orientation'].tolist()

detector_input['position'] = detector_input['position'].tolist()
detector_input['normal'] = detector_input['normal'].tolist()
detector_input['orientation'] = detector_input['orientation'].tolist()

# Compact all inputs into a single input dictionary named save_input
# save_input represents one configuration
save_input = dict()
save_input['general_input'] = general_input
save_input['source_input'] = source_input
save_input['crystal_input'] = crystal_input
save_input['graphite_input'] = graphite_input
save_input['detector_input'] = detector_input

profiler.stop('Dictionary Save Time')
# Ask the user what they want to do with the input file
asking_for_input = True
while asking_for_input is True:
    print('What would you like to do with the input file?')
    config_input = input('SAVE | APPEND | COPY | CLEAR | QUIT: ')

    if config_input == 'SAVE':
        xicsrt_input = list()
        xicsrt_input.append(save_input)
        with open('xicsrt_input.json', 'w') as input_file:
                json.dump(xicsrt_input ,input_file, indent = 1, sort_keys = True)
        asking_for_input = False
    
    if config_input == 'APPEND':
        xicsrt_input.append(save_input)
        with open('xicsrt_input.json', 'w') as input_file:
                json.dump(xicsrt_input ,input_file, indent = 1, sort_keys = True)
        asking_for_input = False        
                
    if config_input == 'COPY':
        with open('xicsrt_input.json', 'w') as input_file:
                json.dump(xicsrt_input ,input_file, indent = 1, sort_keys = True)
        asking_for_input = False        
                
    if config_input == 'CLEAR':
        xicsrt_input = list()
        with open('xicsrt_input.json', 'w') as input_file:
                json.dump(xicsrt_input ,input_file, indent = 1, sort_keys = True)
        asking_for_input = False

    if config_input == 'QUIT':
        asking_for_input = False    
    
    else:
        print('Invalid reply, please try again.')
            

print('Done!')
profiler.stop('Total Time')
profiler.report()
