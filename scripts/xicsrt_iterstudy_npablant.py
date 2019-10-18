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
#sys.path.append('/Users/Eugene/PPPL_python_project1')
#sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code')
sys.path.append('/u/npablant/code/mirproject/xicsrt')

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

## Set up general properties about the raytracer, including random seed
# Possible scenarios include 'MODEL', 'BEAM', 'CRYSTAL', GRAPHITE'
general_input['ideal_geometry']     = True
general_input['backwards_raytrace'] = False
general_input['do_bragg_checks']    = True
general_input['do_simple_bragg']    = False
general_input['do_visualizations']  = False
general_input['random_seed']        = 123456
general_input['scenario']           = 'BEAM'
general_input['system']             = 'w7x_ar16'
general_input['shot']               = 180707017

# Temperature that the optical elements will be cooled to (kelvin)
general_input['xics_temp'] = 273.0

# Number of rays to launch
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
source_input['intensity']    = int(1e6)
general_input['number_of_runs']     = 1

# Xenon mass in AMU
source_input['mass']     = 131.293

# Line location (angstroms) and natural linewith (1/s)
# Xe44+ w line
# Additional information on Xenon spectral lines can be found on nist.gov
source_input['wavelength']   = 2.7203
#source_input['linewidth']    = 1.129e+14
source_input['linewidth']    = 0


"""
Rocking curve FWHM in radians
This is taken from x0h for quartz 1,1,-2,0
Darwin Curve, sigma: 48.070 urad
Darwin Curve, pi:    14.043 urad
Graphite Rocking Curve FWHM in radians
Taken from XOP: 8765 urad
"""

## Load spherical crystal properties from hdf5 file
#config_dict = mirutil.hdf5.hdf5ToDict(
#        '/Users/Eugene/PPPL_python_project1/w7x_ar16_180707017_geometry.hdf5')
config_dict = mirutil.hdf5.hdf5ToDict(
        '/u/npablant/temp/w7x_ar16_180707017_geometry.hdf5')

crystal_input['position']       = config_dict['CRYSTAL_LOCATION']
crystal_input['normal']         = config_dict['CRYSTAL_NORMAL']
crystal_input['orientation']    = config_dict['CRYSTAL_ORIENTATION']

crystal_input['curvature']      = 2.400

crystal_input['spacing']        = 1.70578
crystal_input['width']          = 0.0500
crystal_input['height']         = 0.0400

crystal_input['reflectivity']   = 1
crystal_input['rocking_curve']  = 90.30e-6
crystal_input['pixel_scaling']  = int(200)

crystal_input['therm_expand']   = 5.9e-6

## Load mosaic graphite properties
graphite_input['position']         = config_dict['CRYSTAL_LOCATION']
graphite_input['normal']           = config_dict['CRYSTAL_NORMAL']
graphite_input['orientation']      = config_dict['CRYSTAL_ORIENTATION']

graphite_input['width']            = 0.030
graphite_input['height']           = 0.040
graphite_input['reflectivity']     = 1
graphite_input['mosaic_spread']    = 0.5
graphite_input['spacing']          = 3.35
graphite_input['rocking_curve']    = 8765e-6
#graphite_input['_rocking_curve']    = 1e-1
graphite_input['pixel_scaling']    = int(200)

graphite_input['therm_expand']     = 20e-6

## Load detector properties
detector_input['position']     = config_dict['DETECTOR_LOCATION']
detector_input['normal']       = config_dict['DETECTOR_NORMAL']
detector_input['orientation']  = config_dict['DETECTOR_ORIENTATION']


detector_input['pixel_size']            = 0.000172
detector_input['horizontal_pixels']     = int(config_dict['X_SIZE'])
detector_input['vertical_pixels']       = int(config_dict['Y_SIZE'])

#detector_input['pixel_size']            = 0.001
#detector_input['horizontal_pixels']     = int(500)
#detector_input['vertical_pixels']       = int(500)

detector_input['width'] = detector_input['horizontal_pixels']*detector_input['pixel_size']
detector_input['height'] = detector_input['vertical_pixels']*detector_input['pixel_size']

## Load source properties
source_input['position']     = np.array([0, 0, 0])
source_input['normal']       = np.array([0, 1, 0])
source_input['orientation']  = np.array([0, 0, 1])
source_input['target']       = crystal_input['position']

#Ion temperature in eV
source_input['temp']     = 0.0

#These values are arbitrary for now. Set to 0.0 for point source
#source_input['width']   = 0.0
#source_input['height']  = 0.0
#source_input['depth']   = 0.0

#Angular spread of source in degrees
#This needs to be matched to the source distance and crystal size
#source_input['spread']   = 2.0
#source_input['width']   = 0.1
#source_input['height']  = 0.1
#source_input['depth']   = 4.0

source_input['spread']   = 0.3
source_input['width']   = 0.2
source_input['height']  = 0.2
source_input['depth']   = 0.0


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
crystal_input['sagitt_focus']  = -1 * crystal_input['meridi_focus'] / np.cos(2 * crystal_input['bragg'])

## Set up a legacy beamline scenario
if general_input['scenario'] == 'LEGACY':
    source_input['position'] = source_location_bragg(
        # Distance from Crystal
        3.5,
        # Offset in meridional direction (typically vertical).
        0,
        # Offset in sagital direction (typically horizontal).
        0,
        graphite_input['position'],
        graphite_input['normal'],
        0, 
        graphite_input['spacing'],
        detector_input['position'],
        source_input['wavelength'])
    
    source_input['target'] = crystal_input['position']
    source_input['normal'] = (crystal_input['position'] - source_input['position'])
    source_input['normal']/=  np.linalg.norm(source_input['normal'])

    #This direction is rather abitrary and is not (in general)
    #in the meridional direction.
    source_input['orientation'] = np.cross(np.array([0, 0, 1]), source_input['normal'])
    source_input['orientation'] /= np.linalg.norm(source_input['orientation'])


## Set up a beamline test scenario
elif general_input['scenario'] == 'BEAM' or general_input['scenario'] == 'MODEL':
    total_distance = 9.0+4.0
    dist_crystal_detector = crystal_input['meridi_focus']
    dist_graphite_crystal = 8.5
    dist_source_graphite = total_distance - dist_graphite_crystal

    [source_input['position']        ,
     source_input['normal']          ,
     source_input['orientation']     ,
     graphite_input['position']    ,
     graphite_input['normal']      ,
     graphite_input['orientation'] ,
     crystal_input['position']      ,
     crystal_input['normal']        ,
     crystal_input['orientation']   ,
     detector_input['position']    ,
     detector_input['normal']      ,
     detector_input['orientation'] ,
     source_input['target']] = setup_beam_scenario(
     crystal_input['spacing'],
     graphite_input['spacing'],
     dist_source_graphite,                  # source-graphite distance
     dist_graphite_crystal,                 # graphite-crystal distance
     dist_crystal_detector,                 # crystal-detector distance
     source_input['wavelength'],
     general_input['backwards_raytrace'],
     np.array([0,0,0], dtype = np.float64), # graphite offset (meters)
     np.array([0,0,0], dtype = np.float64), # graphite tilt (radians)
     np.array([0,0,0], dtype = np.float64), # crystal offset (meters)
     np.array([0,0,0], dtype = np.float64), # crystal tilt (radians)
     np.array([0,0,0], dtype = np.float64), # detector offset (meters)
     np.array([0,0,0], dtype = np.float64), # detector tilt (radians)
     )

## Set up a crystal test scenario
elif general_input['scenario'] == 'CRYSTAL':
    [source_input['position']        ,
     source_input['normal']          ,
     source_input['orientation']     ,
     crystal_input['position']      ,
     crystal_input['normal']        ,
     crystal_input['orientation']   ,
     detector_input['position']    ,
     detector_input['normal']      ,
     detector_input['orientation'] ,
     source_input['target']] = setup_crystal_test(
            crystal_input['spacing'],
            5, 1, source_input['wavelength'])
    
    graphite_input['position']     = crystal_input['position']
    graphite_input['normal']       = crystal_input['normal']
    graphite_input['orientation']  = crystal_input['orientation']
    
## Set up a graphite test scenario
elif general_input['scenario'] == 'GRAPHITE':
    [source_input['position']        ,
     source_input['normal']          ,
     source_input['orientation']     ,
     graphite_input['position']    ,
     graphite_input['normal']      ,
     graphite_input['orientation'] ,
     detector_input['position']    ,
     detector_input['normal']      ,
     detector_input['orientation'] ,
     source_input['target']] = setup_graphite_test(
            graphite_input['spacing'],
            5, 1, source_input['wavelength'])
    
    crystal_input['position']       = graphite_input['position']
    crystal_input['normal']         = graphite_input['normal']
    crystal_input['orientation']    = graphite_input['orientation']
    
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
    crystal_input['spacing']   *= 1 + crystal_input['therm_expand']   * (general_input['xics_temp'] - 273)
    graphite_input['spacing'] *= 1 + graphite_input['therm_expand'] * (general_input['xics_temp'] - 273)
    
profiler.stop('Scenario Setup Time')



#%% SETUP
## Pipe all of the configuration settings into their respective objects
print('Setting Up Optics...')
profiler.start('Class Setup Time')

pilatus     = Detector(detector_input, general_input)
crystal     = SphericalCrystal(crystal_input, general_input)
graphite    = MosaicGraphite(graphite_input, general_input)
source      = FocusedExtendedSource(source_input, general_input)

profiler.stop('Class Setup Time')

#%% VISUALIZATION
## Use MatPlotLib Plot3D to visualize the setup
profiler.start('Initial Visual Time')

if general_input['do_visualizations'] is True:
    print('Plotting Visualization...')
    plt1, ax1 = visualize_layout(general_input, source_input, graphite_input, 
                                 crystal_input, detector_input)
    plt1.show()

profiler.stop('Initial Visual Time')

profiler.stop('Total Time')

#%% RAYTRACE
## Begin Raytracing
if __name__ == '__main__':
    print('Beginning Raytracing...')
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


#%% REPORT
profiler.stop('Total Time')
profiler.stopProfiler()
print('')
time.sleep(0.5)
profiler.report()