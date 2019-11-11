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
import argparse
import json
import os
import numpy as np

from xicsrt.xics_rt_sources    import FocusedExtendedSource
from xicsrt.xics_rt_plasmas    import CubicPlasma, CylindricalPlasma, ToroidalPlasma
from xicsrt.xics_rt_detectors  import Detector
from xicsrt.xics_rt_optics     import SphericalCrystal, MosaicGraphite
from xicsrt.xics_rt_raytrace   import raytrace
from xicsrt.xics_rt_model      import analytical_model
from xicsrt.xics_rt_visualizer import visualize_layout, visualize_model, visualize_vectors

from xicsrt.plasma.xics_rt_vmec import FluxSurfacePlasma

profiler.stop('Import Time')

def run(config):

    ## Input Dictionaries
    general_input = config['general_input']
    plasma_input = config['plasma_input']
    source_input = config['source_input']
    crystal_input = config['crystal_input']
    graphite_input = config['graphite_input']
    detector_input = config['detector_input']

    # Initialize the random seed.
    np.random.seed(general_input['random_seed'])

    profiler.start('Class Setup Time')
    pilatus = Detector(detector_input)
    crystal = SphericalCrystal(crystal_input)
    graphite = MosaicGraphite(graphite_input)
    source = FocusedExtendedSource(source_input)
    # This needs to be generalized somehow.
    if config['general_input']['scenario'].lower() == 'plasma':
        if config['plasma_input']['plasma_type'].lower() == 'vmec':
            plasma = FluxSurfacePlasma(plasma_input)
        if config['plasma_input']['plasma_type'].lower() == 'cubic':
            plasma = CubicPlasma(plasma_input)
    profiler.stop('Class Setup Time')

    scenario = str.lower(config['general_input']['scenario'])

    ## Raytrace Runs
    if scenario == 'plasma':
        output, rays_count = raytrace(
            plasma, pilatus, graphite, crystal
            , number_of_runs=general_input['number_of_runs']
            , collect_optics=True)

    elif scenario == 'beam':
        if general_input['backwards_raytrace'] is False:
            output, rays_count = raytrace(
                source, pilatus, graphite, crystal
                , number_of_runs=general_input['number_of_runs']
                , collect_optics=True)

        if general_input['backwards_raytrace'] is True:
            output, rays_count = raytrace(
                source, pilatus, crystal, graphite
                , number_of_runs=general_input['number_of_runs']
                , collect_optics=True)

    elif scenario == 'crystal':
        output, rays_count = raytrace(
            source, pilatus, crystal
            , number_of_runs=general_input['number_of_runs']
            , collect_optics=True)

    elif scenario == 'graphite':
        output, rays_count = raytrace(
            source, pilatus, graphite
            , number_of_runs=general_input['number_of_runs']
            , collect_optics=True)

    elif scenario == 'source':
        output, rays_count = raytrace(
            source, pilatus
            , number_of_runs=general_input['number_of_runs']
            , collect_optics=True)

    else:
        raise Exception('Scenario unknown: {}'.format(scenario))

    #if scenario == 'MODEL':
    #    output, metadata = analytical_model(
    #        source, crystal, graphite, pilatus
    #        , source_input, graphite_input
    #        , crystal_input, detector_input
    #        , general_input)

    if general_input['do_savefiles'] is True:
        ## Create the output path if needed
        if not os.path.exists(general_input['output_path']):
            os.mkdir(general_input['output_path'])

        # create detector image file
        filename = 'xicsrt_detector'
        if general_input['output_suffix']:
            filename += '_' + general_input['output_suffix']
        filename += '.tif'
        filepath = os.path.join(general_input['output_path'], filename)
        print('Exporting detector image: {}'.format(filepath))
        pilatus.output_image(filepath, rotate=False)

        # create graphite image file
        filename = 'xicsrt_graphite'
        if general_input['output_suffix']:
            filename += '_' + general_input['output_suffix']
        filename += '.tif'
        filepath = os.path.join(general_input['output_path'], filename)
        print('Exporting graphite image: {}'.format(filepath))
        graphite.output_image(filepath, rotate=False)

        # create crystal image file
        filename = 'xicsrt_crystal'
        if general_input['output_suffix']:
            filename += '_' + general_input['output_suffix']
        filename += '.tif'
        filepath = os.path.join(general_input['output_path'], filename)
        print('Exporting crystal image:  {}'.format(filepath))
        crystal.output_image(filepath, rotate=False)

    return output, rays_count


def run_multi(config_multi):

    profiler.start('XICSRT Run')

    # create the rays_total dictionary to count the total number of rays
    rays_total = dict()
    rays_total['total_generated'] = 0
    rays_total['total_graphite'] = 0
    rays_total['total_crystal'] = 0
    rays_total['total_detector'] = 0

    output_final = []

    # loop through each configuration in the configuration input file
    for jj, key in enumerate(config_multi):

        ## Object Setup
        logging.info('')
        logging.info('Setting Up Optics for Configuration: {} of {}'.format(
            jj + 1, len(config_multi)))

        output, rays_count = run(config_multi[key])

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