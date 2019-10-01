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
This script loads up all of the initial parameters and object classes needed to
execute the xicsrt raytracing pipeline
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

import os
import argparse
import time
import csv

from xicsrt.xics_rt_visualizer import visualize_layout, visualize_vectors
from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_detectors import Detector
from xicsrt.xics_rt_optics import SphericalCrystal, MosaicGraphite
from xicsrt.xics_rt_raytrace import raytrace
from xicsrt.xics_rt_model import analytical_model
from xicsrt.xics_rt_tools import bragg_angle
from xicsrt.xics_rt_tools import source_location_bragg, setup_beam_scenario
from xicsrt.xics_rt_tools import setup_crystal_test, setup_graphite_test

## MirPyIDL imports
import mirutil.hdf5

profiler.stop('Import Time')
time.sleep(0.1)
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
config_input    = OrderedDict()

## Set up general properties about the configuration file
# Possible modes include 'APPEND', 'CLEAR', 'RUN'
config_input['mode']        = 'APPEND'
config_input['file_name']   = 'config_test.csv'

## Set up general properties about the raytracer, including random seed
# Possible scenarios include 'MODEL', 'BEAM', 'CRYSTAL', GRAPHITE'
general_input['ideal_geometry']     = True
general_input['backwards_raytrace'] = True
general_input['do_visualizations']  = True
general_input['random_seed']        = 123456
general_input['scenario']           = 'BEAM'
general_input['system']             = 'w7x_ar16'
general_input['shot']               = 180707017

# Temperature that the optical elements will be cooled to (kelvin)
general_input['xics_temp'] = 273.0

# Number of rays to launch
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
source_input['source_intensity']    = int(1e6)
general_input['number_of_runs']     = 1

# Xenon mass in AMU
source_input['source_mass']     = 131.293

# Line location (angstroms) and natural linewith (1/s)
# Xe44+ w line
# Additional information on Xenon spectral lines can be found on nist.gov
source_input['source_wavelength']   = 2.7203
source_input['source_linewidth']    = 1.129e+14

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

crystal_input['crystal_position']       = config_dict['CRYSTAL_LOCATION']
crystal_input['crystal_normal']         = config_dict['CRYSTAL_NORMAL']
crystal_input['crystal_orientation']    = config_dict['CRYSTAL_ORIENTATION']

crystal_input['crystal_curvature']      = 1.200
crystal_input['crystal_spacing']        = 1.70578
crystal_input['crystal_width']          = 0.600
crystal_input['crystal_height']         = 0.080

crystal_input['crystal_reflectivity']   = 1
crystal_input['crystal_rocking_curve']  = 90.30e-6
crystal_input['crystal_pixel_scaling']  = int(200)

crystal_input['crystal_therm_expand']   = 5.9e-6

## Load mosaic graphite properties
graphite_input['graphite_position']         = config_dict['CRYSTAL_LOCATION']
graphite_input['graphite_normal']           = config_dict['CRYSTAL_NORMAL']
graphite_input['graphite_orientation']      = config_dict['CRYSTAL_ORIENTATION']

graphite_input['graphite_width']            = 0.060
graphite_input['graphite_height']           = 0.250
graphite_input['graphite_reflectivity']     = 1
graphite_input['graphite_mosaic_spread']    = 0.5
graphite_input['graphite_spacing']          = 3.35
graphite_input['graphite_rocking_curve']    = 8765e-6
graphite_input['graphite_pixel_scaling']    = int(200)

graphite_input['graphite_therm_expand']     = 20e-6

## Load detector properties
detector_input['detector_position']     = config_dict['DETECTOR_LOCATION']
detector_input['detector_normal']       = config_dict['DETECTOR_NORMAL']
detector_input['detector_orientation']  = config_dict['DETECTOR_ORIENTATION']


detector_input['pixel_size']            = 0.000172
detector_input['horizontal_pixels']     = int(config_dict['X_SIZE'])
detector_input['vertical_pixels']       = int(config_dict['Y_SIZE'])

## Load source properties
source_input['source_position']     = np.array([0, 0, 0])
source_input['source_normal']       = np.array([0, 1, 0])
source_input['source_orientation']  = np.array([0, 0, 1])
source_input['source_target']       = crystal_input['crystal_position']

#Angular spread of source in degrees
#This needs to be matched to the source distance and crystal size
source_input['source_spread']   = 1.0
#Ion temperature in eV
source_input['source_temp']     = 1000

#These values are arbitrary for now. Set to 0.0 for point source
source_input['source_width']   = 0.0
source_input['source_height']  = 0.0
source_input['source_depth']   = 0.0

profiler.stop('Input Setup Time')
time.sleep(0.1)
#%% SCENARIO
"""
Each of these scenarios corresponds to a script located in xics_rt_tools.py 
which assembles the optical elements into a specific configuration based on
input parameters
"""
print('Arranging Scenarios...')
profiler.start('Scenario Setup Time')

# Analytical solutions for spectrometer geometry involving crystal focus
crystal_input['crystal_bragg'] = bragg_angle(source_input['source_wavelength'], crystal_input['crystal_spacing'])
crystal_input['meridi_focus']  = crystal_input['crystal_curvature'] * np.sin(crystal_input['crystal_bragg'])
crystal_input['sagitt_focus']  = - crystal_input['meridi_focus'] / np.cos(2 * crystal_input['crystal_bragg'])

## Set up a legacy beamline scenario
if general_input['scenario'] == 'LEGACY':
    source_input['source_position'] = source_location_bragg(
        # Distance from Crystal
        3.5,
        # Offset in meridional direction (typically vertical).
        0,
        # Offset in sagital direction (typically horizontal).
        0,
        graphite_input['graphite_position'],
        graphite_input['graphite_normal'], 
        0, 
        graphite_input['graphite_spacing'],
        detector_input['detector_position'],
        source_input['source_wavelength'])
    
    source_input['source_target'] = crystal_input['crystal_position']
    source_input['source_normal'] = (crystal_input['crystal_position'] - source_input['source_position'])
    source_input['source_normal']/=  np.linalg.norm(source_input['source_normal'])

    #This direction is rather abitrary and is not (in general)
    #in the meridional direction.
    source_input['source_orientation'] = np.cross(np.array([0, 0, 1]), source_input['source_normal'])
    source_input['source_orientation'] /= np.linalg.norm(source_input['source_orientation'])

## Set up a beamline test scenario
elif general_input['scenario'] == 'BEAM' or general_input['scenario'] == 'MODEL':
    [source_input['source_position']        ,
     source_input['source_normal']          , 
     source_input['source_orientation']     ,
     graphite_input['graphite_position']    ,
     graphite_input['graphite_normal']      ,
     graphite_input['graphite_orientation'] ,
     crystal_input['crystal_position']      ,
     crystal_input['crystal_normal']        ,
     crystal_input['crystal_orientation']   ,
     detector_input['detector_position']    ,
     detector_input['detector_normal']      ,
     detector_input['detector_orientation'] , 
     source_input['source_target']] = setup_beam_scenario(
     crystal_input['crystal_spacing'], 
     graphite_input['graphite_spacing'],
     1,                                     #source-graphite distance
     crystal_input['sagitt_focus'],         #graphite-crystal distance
     crystal_input['meridi_focus'],         #crystal-detector distance
     source_input['source_wavelength'],
     general_input['backwards_raytrace'],
     np.array([0,0,0], dtype = np.float64), #graphite offset (meters)
     np.array([0,0,0], dtype = np.float64), #graphite tilt (radians)
     np.array([0,0,0], dtype = np.float64), #crystal offset (meters)
     np.array([0,0,0], dtype = np.float64), #crystal tilt (radians)
     np.array([0,0,0], dtype = np.float64), #detector offset (meters)
     np.array([0,0,0], dtype = np.float64), #detector tilt (radians)
     )

## Set up a crystal test scenario
elif general_input['scenario'] == 'CRYSTAL':
    [source_input['source_position']        ,
     source_input['source_normal']          ,
     source_input['source_orientation']     , 
     crystal_input['crystal_position']      ,
     crystal_input['crystal_normal']        ,
     crystal_input['crystal_orientation']   ,
     detector_input['detector_position']    ,
     detector_input['detector_normal']      ,
     detector_input['detector_orientation'] , 
     source_input['source_target']] = setup_crystal_test(
            crystal_input['crystal_spacing'], 
            5, 1, source_input['source_wavelength'])
    
    graphite_input['graphite_position']     = crystal_input['crystal_position']
    graphite_input['graphite_normal']       = crystal_input['crystal_normal']
    graphite_input['graphite_orientation']  = crystal_input['crystal_orientation']
    
## Set up a graphite test scenario
elif general_input['scenario'] == 'GRAPHITE':
    [source_input['source_position']        ,
     source_input['source_normal']          ,
     source_input['source_orientation']     , 
     graphite_input['graphite_position']    ,
     graphite_input['graphite_normal']      ,
     graphite_input['graphite_orientation'] ,
     detector_input['detector_position']    ,
     detector_input['detector_normal']      ,
     detector_input['detector_orientation'] , 
     source_input['source_target']] = setup_graphite_test(
            graphite_input['graphite_spacing'], 
            5, 1, source_input['source_wavelength'])
    
    crystal_input['crystal_position']       = graphite_input['graphite_position']
    crystal_input['crystal_normal']         = graphite_input['graphite_normal']
    crystal_input['crystal_orientation']    = graphite_input['graphite_orientation']
    
## Backwards raytracing involves swapping the source and detector
if general_input['backwards_raytrace']:
    swap_position   = source_input['source_position']
    swap_orientation= source_input['source_orientation']
    swap_normal     = source_input['source_normal']
    
    source_input['source_position']    = detector_input['detector_position']
    source_input['source_orientation'] = detector_input['detector_orientation']
    source_input['source_normal']      = detector_input['detector_normal']
    
    detector_input['detector_position']     = swap_position
    detector_input['detector_orientation']  = swap_orientation
    detector_input['detector_normal']       = swap_normal
    
## Simulate linear thermal expansion
# This is calculated AFTER spectrometer geometry setup to simulate non-ideal conditions
if general_input['ideal_geometry'] is False:
    crystal_input['crystal_spacing']   *= 1 + crystal_input['crystal_therm_expand']   * (general_input['xics_temp'] - 273)
    graphite_input['graphite_spacing'] *= 1 + graphite_input['graphite_therm_expand'] * (general_input['xics_temp'] - 273)
    
profiler.stop('Scenario Setup Time')
time.sleep(0.1)
#%% CONFIGURATION
# Perform changes to the configuration file if requested
""" EXPERIMENTAL
if config_input['mode'] == 'APPEND':
    configuration = [source_input, graphite_input, crystal_input, detector_input]
    csv_columns = ['KEY','VALUE']
    with open(config_input['file_name'], "w") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames = csv_columns)
        csvwriter.writeheader()
    for data in configuration:
        csvwriter.writerow(data)
"""

#%% SETUP
## Pipe all of the configuration settings into their respective objects
print('Setting Up Optics...')
profiler.start('Class Setup Time')

pilatus     = Detector(detector_input, general_input)
crystal     = SphericalCrystal(crystal_input, general_input)
graphite    = MosaicGraphite(graphite_input, general_input)
source      = FocusedExtendedSource(source_input, general_input)

profiler.stop('Class Setup Time')
time.sleep(0.1)
#%% VISUALIZATION
## Use MatPlotLib Plot3D to visualize the setup
profiler.start('Initial Visual Time')

if general_input['do_visualizations'] is True:
    print('Plotting Visualization...')
    plt1, ax1 = visualize_layout(general_input, source_input, graphite_input, 
                                 crystal_input, detector_input)
    plt1.show()

profiler.stop('Initial Visual Time')
time.sleep(0.1)
#%% RAYTRACE
## Begin Raytracing
print('Beginning Raytracing...')
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suffix'
        ,help='A suffix to add to the end of the image name.'
        ,type=str)
    parser.add_argument(
        '--path'
        ,default=''
        ,help='The path to store the results.'
        ,type=str)    
    args = parser.parse_args()
    
    if general_input['scenario'] == 'BEAM':
        if general_input['backwards_raytrace'] is False:
            output = raytrace(
                source
                ,pilatus
                ,graphite        
                ,crystal
                ,number_of_runs = general_input['number_of_runs']
                ,collect_optics = True)
            
        if general_input['backwards_raytrace'] is True:
            output = raytrace(
                source
                ,pilatus
                ,crystal        
                ,graphite
                ,number_of_runs = general_input['number_of_runs']
                ,collect_optics = True)
            
    if general_input['scenario'] == 'CRYSTAL':
        output = raytrace(
                source
                ,pilatus
                ,crystal
                ,number_of_runs = general_input['number_of_runs']
                ,collect_optics = True)
        
    if general_input['scenario'] == 'GRAPHITE':
        output = raytrace(
                source
                ,pilatus
                ,graphite
                ,number_of_runs = general_input['number_of_runs']
                ,collect_optics = True)
        
    if general_input['scenario'] == 'MODEL':
        output = analytical_model(source, crystal, graphite, pilatus, 
                                  source_input, graphite_input, crystal_input,
                                  detector_input, general_input)
    
    time.sleep(0.5)
## Create the output path if needed
    if args.path:
        if not os.path.exists(args.path):
           os.mkdir(args.path)
                    
    #create detector image file
    filename = 'xicsrt_detector'
    if args.suffix:
        filename += '_'+args.suffix
    filename += '.tif'
    filepath = os.path.join(args.path, filename)
    print('Exporting detector image: {}'.format(filepath))
    pilatus.output_image(filepath, rotate=False)
    
    #create graphite image file
    filename = 'xicsrt_graphite'
    if args.suffix:
        filename += '_'+args.suffix
    filename += '.tif'
    filepath = os.path.join(args.path, filename)
    print('Exporting graphite image: {}'.format(filepath))
    graphite.output_image(filepath, rotate=False)
    
    #create crystal image file
    filename = 'xicsrt_crystal'
    if args.suffix:
        filename += '_'+args.suffix
    filename += '.tif'
    filepath = os.path.join(args.path, filename)
    print('Exporting crystal image:  {}'.format(filepath))
    crystal.output_image(filepath, rotate=False)
time.sleep(0.1)
#%% OUTPUT
## Add the rays to the previous Axes3D plot

profiler.start('Final Visual Time')

if general_input['do_visualizations'] is True:
    print("Plotting Rays...")
    for ii in range(len(output)):
        plt2, ax2 = visualize_vectors(output[ii], general_input, source_input, 
                                      graphite_input, crystal_input, 
                                      detector_input)
        plt2.show()

profiler.stop('Final Visual Time')

time.sleep(0.1)
#%% REPORT
profiler.stop('Total Time')
profiler.stopProfiler()
print('')
time.sleep(0.5)
profiler.report()