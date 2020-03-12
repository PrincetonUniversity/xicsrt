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

import logging
import os
import numpy as np

from xicsrt.sources._XicsrtSourceFocused        import XicsrtSourceFocused
from xicsrt.sources._XicsrtPlasmaVmecDatafile   import XicsrtPlasmaVmecDatafile
from xicsrt.optics._XicsrtOpticDetector         import XicsrtOpticDetector
from xicsrt.optics._XicsrtOpticCrystalSpherical import XicsrtOpticCrystalSpherical
from xicsrt.optics._XicsrtOpticMosaicGraphite   import XicsrtOpticMosaicGraphite
from xicsrt.visual.xicsrt_visualizer            import visualize_layout
from xicsrt.visual.xicsrt_visualizer            import visualize_vectors, visualize_bundles
from xicsrt.filters._XicsrtBundleFilterSightline import XicsrtBundleFilterSightline

from xicsrt.xicsrt_raytrace   import raytrace


def run(config, config_number = None):
    ## Initial Setup

    # Initialize the random seed
    np.random.seed(config['general_input']['random_seed'])
    
    # Setup Classes
    pilatus  = XicsrtOpticDetector(        config['detector_input'])
    crystal  = XicsrtOpticCrystalSpherical(config['crystal_input'])
    graphite = XicsrtOpticMosaicGraphite(  config['graphite_input'])
    source   = XicsrtSourceFocused(        config['source_input'])
    plasma   = XicsrtPlasmaVmecDatafile(   config['plasma_input'])
    filter   = XicsrtBundleFilterSightline(config['filter_input'])
    
    runs     = config['general_input']['number_of_runs']
    save     = config['general_input']['do_savefiles']
    view     = config['general_input']['do_visualizations']
    scenario = str.upper(config['general_input']['scenario'])
    
    ## Initial Visualization
    if view is True:
        fig1, ax1 = visualize_layout(config)
        fig1.show()

    ## Raytrace Runs
    if scenario == 'REAL' or scenario == 'PLASMA':
        plasma.filter_list.append(filter)
        output, rays_count = raytrace(plasma, pilatus, graphite, crystal,
                                      number_of_runs = runs, collect_optics = save)

    elif scenario == 'THROUGHPUT':
        output, rays_count = raytrace(plasma, pilatus, crystal,
                                      number_of_runs = runs, collect_optics = save)

    elif scenario == 'BEAM':
        if config['general_input']['backwards_raytrace'] is False:
            output, rays_count = raytrace(source, pilatus, graphite, crystal,
                                          number_of_runs = runs, collect_optics = save)

        if config['general_input']['backwards_raytrace'] is True:
            output, rays_count = raytrace(source, pilatus, crystal, graphite,
                                          number_of_runs = runs, collect_optics = save)

    elif scenario == 'MANFRED':
        output, rays_count = raytrace(source, pilatus, crystal,
                                      number_of_runs = runs, collect_optics = save)

    elif scenario == 'CRYSTAL':
        output, rays_count = raytrace(source, pilatus, crystal,
                                      number_of_runs = runs, collect_optics = save)

    elif scenario == 'GRAPHITE':
        output, rays_count = raytrace(source, pilatus, graphite,
                                      number_of_runs = runs, collect_optics = save)

    elif scenario == 'SOURCE':
        output, rays_count = raytrace(source, pilatus,
                                      number_of_runs = runs, collect_optics = save)

    else:
        raise Exception('Scenario unknown: {}'.format(scenario))

    #if scenario == 'MODEL':
    #    output, metadata = analytical_model(
    #        source, crystal, graphite, pilatus
    #        , source_input, graphite_input
    #        , crystal_input, detector_input
    #        , general_input)
    
    ## Final Visualization
    if view is True:
        fig2, ax2 = visualize_bundles(config, output)
        fig2.show()
        for ii in range(len(output)):
            fig3, ax3 = visualize_vectors(config, output, ii)
            fig3.show()
    
    ## Save Outputs
    if save is True:
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

    return output, rays_count


def run_multi(config_multi):

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

        output, rays_count = run(config_multi[jj], jj)
        
        hits_final.append(output)
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

    return hits_final, output_final, rays_total