# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
Initializes and runs the ray-tracer for a given set of configurations.
"""

from xicsrt.util import profiler

profiler.start('Import Time')

import logging
import os
import numpy as np

from xicsrt.xics_rt_sources    import FocusedExtendedSource
from xicsrt.xics_rt_plasmas    import RealPlasma, ToroidalPlasma
from xicsrt.xics_rt_detectors  import Detector
from xicsrt.xics_rt_optics     import SphericalCrystal, MosaicGraphite
from xicsrt.xics_rt_raytrace   import raytrace
from xicsrt.xics_rt_model      import analytical_model
from xicsrt.xics_rt_visualizer import visualize_layout, visualize_model
from xicsrt.xics_rt_visualizer import visualize_vectors, visualize_bundles

profiler.stop('Import Time')

def run(config, config_number = None):
    ## Initial Setup

    # Initialize the random seed
    np.random.seed(config['general_input']['random_seed'])
    
    # Setup Classes
    profiler.start('Class Setup Time')
    pilatus  = Detector(                config['detector_input'])
    crystal  = SphericalCrystal(        config['crystal_input'])
    graphite = MosaicGraphite(          config['graphite_input'])
    source   = FocusedExtendedSource(   config['source_input'])
    plasma   = RealPlasma(              config['plasma_input'])
    
    profiler.stop('Class Setup Time')

    scenario = str.upper(config['general_input']['scenario'])
    
    ## Initial Visualization
    if config['general_input']['do_visualizations'] is True:
        fig1, ax1 = visualize_layout(config)
        fig1.show()

    ## Raytrace Runs
    if scenario == 'REAL' or scenario == 'PLASMA':
        output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
            plasma, pilatus, graphite, crystal)

    elif scenario == 'THROUGHPUT':
        output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
            plasma, pilatus, crystal)

    elif scenario == 'BEAM':
        if config['general_input']['backwards_raytrace'] is False:
            output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
                source, pilatus, graphite, crystal)

        if config['general_input']['backwards_raytrace'] is True:
            output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
                source, pilatus, crystal, graphite)

    elif scenario == 'MANFRED':
        output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
            source, pilatus, crystal)

    elif scenario == 'CRYSTAL':
        output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
            source, pilatus, crystal)

    elif scenario == 'GRAPHITE':
        output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
            source, pilatus, graphite)

    elif scenario == 'SOURCE':
        output_hits, output, rays_count = raytrace(config['general_input']['number_of_runs'],
            source, pilatus)

    else:
        raise Exception('Scenario unknown: {}'.format(scenario))

    #if scenario == 'MODEL':
    #    output, metadata = analytical_model(
    #        source, crystal, graphite, pilatus
    #        , source_input, graphite_input
    #        , crystal_input, detector_input
    #        , general_input)
    
    ## Final Visualization
    if config['general_input']['do_visualizations'] is True:
        fig2, ax2 = visualize_bundles(config, output)
        fig2.show()
        for ii in range(len(output)):
            fig3, ax3 = visualize_vectors(config, output, ii)
            fig3.show()
    
    ## Save Outputs
    if config['general_input']['do_savefiles'] is True:
        ## Create the output path if needed
        if not os.path.exists(config['general_input']['output_path']):
            os.mkdir(config['general_input']['output_path'])

        # create detector image file
        filename = 'xicsrt_detector'
        if config_number is not None:
            filename+= '_'
            filename+= str(int(config_number))
        filename+= config['general_input']['output_suffix']
        filepath = os.path.join(config['general_input']['output_path'], filename)
        print('Exporting detector image: {}'.format(filepath))
        pilatus.output_image(filepath, rotate=False)

        # create graphite image file
        filename = 'xicsrt_graphite'
        if config_number is not None:
            filename+= '_'
            filename+= str(int(config_number))
        filename+= config['general_input']['output_suffix']
        filepath = os.path.join(config['general_input']['output_path'], filename)
        print('Exporting graphite image: {}'.format(filepath))
        graphite.output_image(filepath, rotate=False)

        # create crystal image file
        filename = 'xicsrt_crystal'
        if config_number is not None:
            filename+= '_'
            filename+= str(int(config_number))
        filename+= config['general_input']['output_suffix']
        filepath = os.path.join(config['general_input']['output_path'], filename)
        print('Exporting crystal image:  {}'.format(filepath))
        crystal.output_image(filepath, rotate=False)

    return output_hits, output, rays_count


def run_multi(config_multi):

    profiler.start('XICSRT Run')

    # create the rays_total dictionary to count the total number of rays
    rays_total = dict()
    rays_total['total_generated'] = 0
    rays_total['total_graphite']  = 0
    rays_total['total_crystal']   = 0
    rays_total['total_detector']  = 0
    
    hits_final   = []
    output_final = []

    # loop through each configuration in the configuration input file
    for jj in range(len(config_multi)):

        ## Object Setup
        print('')
        print('Setting Up Optics for Configuration: {} of {}'.format(
            jj + 1, len(config_multi)))

        output_hits, output, rays_count = run(config_multi[jj], jj)
        
        hits_final.append(output_hits)
        output_final.append(output)
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

    profiler.stop('XICSRT Run')

    return hits_final, output_final, rays_total