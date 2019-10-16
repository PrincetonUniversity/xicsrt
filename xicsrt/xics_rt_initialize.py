# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:02:59 2019

Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
Initializes and runs the ray-tracer. Also performs pre- and post-raytracing 
visualizations, as well as the profiler report.
"""
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code')

from xicsrt.util import profiler
profiler.startProfiler()

profiler.start('Total Time')
profiler.start('Import Time')

import argparse
import json
import os
import numpy as np
from collections import OrderedDict

from xicsrt.xics_rt_visualizer import visualize_layout, visualize_vectors
from xicsrt.xics_rt_visualizer import visualize_model, visualize_images
from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_detectors import Detector
from xicsrt.xics_rt_optics import SphericalCrystal, MosaicGraphite
from xicsrt.xics_rt_raytrace import raytrace
from xicsrt.xics_rt_model import analytical_model
profiler.stop('Import Time')

#%% LOAD
# Set up OrederedDicts
profiler.start('Load Time')
try:
    with open('../scripts/xicsrt_input.json', 'r') as input_file:
        xicsrt_input = json.load(input_file)
except FileNotFoundError:
    print('Input file xicsrt_input.json does not exist')
    print('Run xicsrt_w7x_ar16_eugene.py to generate input file')
    sys.exit()
"""
general_input   = xicsrt_input['general_input']
source_input    = xicsrt_input['source_input']
crystal_input   = xicsrt_input['crystal_input']
graphite_input  = xicsrt_input['graphite_input']
detector_input  = xicsrt_input['detector_input']
"""
# Deconvert all lists back into numpy arrays
# I swear all this nesting is worth it
for configuration in range(len(xicsrt_input)):
    for element in xicsrt_input[configuration]:
        for key in xicsrt_input[configuration][element]:
            if type(xicsrt_input[configuration][element][key]) is list:
                xicsrt_input[configuration][element][key] = np.array(
                        xicsrt_input[configuration][element][key])

profiler.stop('Load Time')

#%% VISUALIZATION
## Use MatPlotLib Plot3D to visualize the setup
"""
profiler.start('Initial Visual Time')

if general_input['do_visualizations'] is True:
    print('Plotting Visualization...')
    plt1, ax1 = visualize_layout(general_input, source_input, graphite_input, 
                                 crystal_input, detector_input)
    plt1.show()

profiler.stop('Initial Visual Time')
"""
#%% RAYTRACE
## Begin Raytracing
print('Beginning Raytracing...')
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
    
    # create the rays_total dictionary to count the total number of rays
    rays_total = dict()
    rays_total['total_generated']  = 0
    rays_total['total_graphite']   = 0
    rays_total['total_crystal']    = 0
    rays_total['total_detector']   = 0
    
    # loop through each configuration in the configuration input file
    for jj in range(len(xicsrt_input)):
        # input the dictionaries from xicsrt_input
        general_input   = xicsrt_input[jj]['general_input']
        source_input    = xicsrt_input[jj]['source_input']
        crystal_input   = xicsrt_input[jj]['crystal_input']
        graphite_input  = xicsrt_input[jj]['graphite_input']
        detector_input  = xicsrt_input[jj]['detector_input']
        
        # pipe all of the configuration settings into their respective objects
        print('')
        print('Setting Up Optics for Configuration: {} of {}...'.format(
                jj + 1, len(xicsrt_input)))
        
        profiler.start('Class Setup Time')

        pilatus     = Detector(detector_input, general_input)
        crystal     = SphericalCrystal(crystal_input, general_input)
        graphite    = MosaicGraphite(graphite_input, general_input)
        source      = FocusedExtendedSource(source_input, general_input)

        profiler.stop('Class Setup Time')

        #raytrace runs
        if general_input['scenario'] == 'BEAM':
            if general_input['backwards_raytrace'] is False:
                output, rays_count = raytrace(source, pilatus, graphite, crystal
                    ,number_of_runs = general_input['number_of_runs']
                    ,collect_optics = True)
                
            if general_input['backwards_raytrace'] is True:
                output, rays_count = raytrace(source, pilatus, crystal, graphite
                    ,number_of_runs = general_input['number_of_runs']
                    ,collect_optics = True)
                
        if general_input['scenario'] == 'CRYSTAL':
            output, rays_count = raytrace(source, pilatus, crystal
                    ,number_of_runs = general_input['number_of_runs']
                    ,collect_optics = True)
            
        if general_input['scenario'] == 'GRAPHITE':
            output, rays_count = raytrace(source, pilatus, graphite
                    ,number_of_runs = general_input['number_of_runs']
                    ,collect_optics = True)
            
        if general_input['scenario'] == 'SOURCE':
            output, rays_count = raytrace(source, pilatus
                    ,number_of_runs = general_input['number_of_runs']
                    ,collect_optics = True)
            
        if general_input['scenario'] == 'MODEL':
            output, metadata = analytical_model(source, crystal, graphite, pilatus, 
                                                source_input, graphite_input, 
                                                crystal_input, detector_input, 
                                                general_input)
            
        for key in rays_total:
            rays_total[key] += rays_count[key]
            
    # after all raytrace runs for all configurations, report the ray totals
    print('')
    print('Total Rays Generated: {:6.4e}'.format(rays_total['total_generated']))
    print('Total Rays on HOPG:   {:6.4e}'.format(rays_total['total_graphite']))
    print('Total Rays on Crystal:{:6.4e}'.format(rays_total['total_crystal']))
    print('Total Rays Detected:  {:6.4e}'.format(rays_total['total_detector']))
    print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
        rays_total['total_detector'] / rays_total['total_generated'],
        np.sqrt(rays_total['total_detector']) / rays_total['total_generated'],
        rays_total['total_detector'] / rays_total['total_generated'] * 100))
    print('')
    
## Create the output path if needed
    if general_input['do_savefiles'] is True:
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
"""
profiler.start('Final Visual Time')

if general_input['do_visualizations'] is True:
    print("Plotting Results...")
    
    if general_input['scenario'] == 'MODEL':
        fig2, ax2 = visualize_model(output, metadata, general_input, source_input, 
                                    graphite_input, crystal_input, detector_input)
    else:
        for ii in range(len(output)):
            fig2, ax2 = visualize_vectors(output[ii], general_input, source_input, 
                                          graphite_input, crystal_input, detector_input)
    fig2.show()
    
if general_input['do_image_analysis'] is True:
    fig3, ax3 = visualize_images()
    fig3.show()
    input('Press [Enter] to close the image analysis window...')
    
profiler.stop('Final Visual Time')
"""
#%% REPORT
profiler.stop('Total Time')
profiler.stopProfiler()
print('')

if general_input['scenario'] == 'MODEL':    
    # models also come with metadata, print that out
    print('')
    print('Analytical Model Results')
    
    if general_input['backwards_raytrace'] is True:
        print('Header  |  Crystal Bragg | Crystal Dist | Graphite Bragg | Graphite Dist')
        print('Setup   |  {:6.6} deg   | {:6.6} m   | {:6.6} deg    | {:6.6} m'.format(
            crystal_input['bragg'] * 180 / np.pi,
            np.linalg.norm(crystal_input['position'] - source_input['position']),
            graphite_input['bragg'] * 180 / np.pi,
            np.linalg.norm(graphite_input['position'] - crystal_input['position']),
            np.linalg.norm(detector_input['position'] - graphite_input['position'])))
        
        for jj in range(len(metadata[0]['distance'])):
            if   metadata[0]['distance'][jj] == 10.0 or metadata[0]['distance'][jj] == 0.0:
                print('Ray {}   |  [MISSED]      | [MISSED]     | [MISSED]       | [MISSED]'.format(jj))
                
            elif metadata[1]['distance'][jj] == 10.0 or metadata[1]['distance'][jj] == 0.0:
                print('Ray {}   |  {:6.6} deg   | {:6.6} m   | [MISSED]       | [MISSED]'.format(
                    jj,
                    float(metadata[0]['incident_angle'][jj]) * 180 / np.pi,
                    float(metadata[0]['distance'][jj])))  
                
            else:
                print('Ray {}   |  {:6.6} deg   | {:6.6} m   | {:6.6} deg    | {:6.6} m'.format(
                    jj,
                    float(metadata[0]['incident_angle'][jj]) * 180 / np.pi,
                    float(metadata[0]['distance'][jj]),
                    float(metadata[1]['incident_angle'][jj]) * 180 / np.pi,
                    float(metadata[1]['distance'][jj])))  

profiler.report()
