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
from xicsrt.xics_rt_detectors  import Detector
from xicsrt.xics_rt_optics     import SphericalCrystal, MosaicGraphite
from xicsrt.xics_rt_raytrace   import raytrace
from xicsrt.xics_rt_model      import analytical_model

from xicsrt.xics_rt_plasmas            import ToroidalPlasma, CubicPlasma
from xicsrt.plasma.xics_rt_plasma_vmec import FluxSurfacePlasma

profiler.stop('Import Time')

def run(config, name=None):
    ## Initial Setup

    # Initialize the random seed
    np.random.seed(config['general_input']['random_seed'])
    
    # Setup Classes
    profiler.start('Class Setup Time')

    detector  = Detector(config['detector_input'])
    crystal  = SphericalCrystal(config['crystal_input'])
    graphite = MosaicGraphite(config['graphite_input'])

    # This needs to be generalized somehow.
    if config['general_input']['scenario'].lower() == 'plasma':
        if config['source_input']['plasma_type'].lower() == 'vmec':
            source = FluxSurfacePlasma(config['source_input'])
        if config['source_input']['plasma_type'].lower() == 'toroidal':
            source = ToroidalPlasma(config['source_input'])
        if config['source_input']['plasma_type'].lower() == 'cubic':
            source = CubicPlasma(config['source_input'])
        else:
            raise Exception('Plasma type unknown: {}'.format(config['source_input']['plasma_type']))
    else:
        source = FocusedExtendedSource(config['source_input'])

    profiler.stop('Class Setup Time')

    scenario = str.upper(config['general_input']['scenario'])

    ## Raytrace Runs
    if scenario == 'PLASMA':
        output, rays_count = raytrace(
            source, detector, graphite, crystal
            ,number_of_runs=config['general_input']['number_of_runs']
            ,collect_optics=True)

    elif scenario == 'BEAM':
        if config['general_input']['backwards_raytrace'] is False:
            output, rays_count = raytrace(
                source, detector, graphite, crystal
                ,number_of_runs = config['general_input']['number_of_runs']
                ,collect_optics = True)

        if config['general_input']['backwards_raytrace'] is True:
            output, rays_count = raytrace(
                source, detector, crystal, graphite
                ,number_of_runs = config['general_input']['number_of_runs']
                ,collect_optics = True)

    elif scenario == 'CRYSTAL':
        output, rays_count = raytrace(
            source, detector, crystal
            ,number_of_runs = config['general_input']['number_of_runs']
            ,collect_optics = True)

    elif scenario == 'GRAPHITE':
        output, rays_count = raytrace(
            source, detector, graphite
            ,number_of_runs = config['general_input']['number_of_runs']
            ,collect_optics = True)

    elif scenario == 'SOURCE':
        output, rays_count = raytrace(
            source, detector
            ,number_of_runs = config['general_input']['number_of_runs']
            ,collect_optics = True)

    elif scenario == 'THROUGHPUT':
         output, rays_count = raytrace(
             source, detector, crystal
             ,number_of_runs = config['general_input']['number_of_runs']
             ,collect_optics = True)

    elif scenario == 'MANFRED':
        output, rays_count = raytrace(
             source, detector, crystal
             ,number_of_runs = config['general_input']['number_of_runs']
             ,collect_optics = True)  

    else:
        raise Exception('Scenario unknown: {}'.format(scenario))

    #if scenario == 'MODEL':
    #    output, metadata = analytical_model(
    #        source, crystal, graphite, detector
    #        , source_input, graphite_input
    #        , crystal_input, detector_input
    #        , general_input)
    
    ## Save Outputs
    if config['general_input']['do_savefiles'] is True:
        ## Create the output path if needed
        if not os.path.exists(config['general_input']['output_path']):
            os.mkdir(config['general_input']['output_path'])

        # create detector image file
        filename = 'xicsrt_detector'
        if name is not None:
            filename += '_'+str(name)
        filename += config['general_input']['output_suffix']
        filepath = os.path.join(config['general_input']['output_path'], filename)
        print('Exporting detector image: {}'.format(filepath))
        detector.output_image(filepath, rotate=False)

        # create graphite image file
        filename = 'xicsrt_graphite'
        if name is not None:
            filename += '_'+str(name)
        filename+= config['general_input']['output_suffix']
        filepath = os.path.join(config['general_input']['output_path'], filename)
        print('Exporting graphite image: {}'.format(filepath))
        graphite.output_image(filepath, rotate=False)

        # create crystal image file
        filename = 'xicsrt_crystal'
        if name is not None:
            filename += '_'+str(name)
        filename+= config['general_input']['output_suffix']
        filepath = os.path.join(config['general_input']['output_path'], filename)
        print('Exporting crystal image:  {}'.format(filepath))
        crystal.output_image(filepath, rotate=False)

    return output, rays_count


def run_multi(config_multi):

    profiler.start('XICSRT Run')

    # create the rays_total dictionary to count the total number of rays
    rays_total = dict()
    rays_total['total_generated'] = 0
    rays_total['total_graphite']  = 0
    rays_total['total_crystal']   = 0
    rays_total['total_detector']  = 0

    output_final = []

    # loop through each configuration in the configuration input file
    for jj, key in enumerate(config_multi):

        ## Object Setup
        logging.info('')
        logging.info('Setting Up Optics for Configuration: {} of {}'.format(
            jj + 1, len(config_multi)))

        output, rays_count = run(config_multi[key], key)

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

    return output_final, rays_total
