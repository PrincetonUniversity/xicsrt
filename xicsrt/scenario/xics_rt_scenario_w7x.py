# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir pablant <npablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------

"""

from xicsrt.util import profiler

profiler.start('Import Time')
import numpy as np
import logging
import os

from collections import OrderedDict

from xicsrt.xics_rt_raytrace   import raytrace
from xicsrt import xics_rt_input

from xicsrt.xics_rt_detectors  import Detector
from xicsrt.xics_rt_optics     import SphericalCrystal
from xicsrt.plasma.xics_rt_plasma_w7x_simple import W7xSimplePlasma

profiler.stop('Import Time')

def initialize(config):

    # Setup our plasma box to be radial.
    config['source_input']['normal'] = config['source_input']['position']
    config['source_input']['normal'] /= np.linalg.norm(config['source_input']['normal'])

    config['source_input']['orientation'] = np.cross(config['source_input']['normal'], np.array([0,0,1]))
    config['source_input']['orientation'] /= np.linalg.norm(config['source_input']['orientation'])

    config['source_input']['target'] = config['crystal_input']['position']

    xics_rt_input.config_to_numpy(config)
    return config


def run(config, name=None, do_random_seed=True):

    if do_random_seed:
        # Initialize the random seed.
        np.random.seed(config['general_input']['random_seed'])

    profiler.start('Class Setup Time')

    detector = Detector(config['detector_input'])
    crystal = SphericalCrystal(config['crystal_input'])
    source = W7xSimplePlasma(config['source_input'])

    profiler.stop('Class Setup Time')

    scenario = str.lower(config['general_input']['scenario'])

    ## Raytrace Runs

    output, meta = raytrace(
        source
        ,detector
        ,crystal
        ,number_of_runs=config['general_input']['number_of_runs']
        ,collect_optics=True)

    ## Save Outputs
    if config['general_input']['do_savefiles'] is True:
        ## Create the output path if needed
        if not os.path.exists(config['general_input']['output_path']):
            os.mkdir(config['general_input']['output_path'])

        # create source image file
        if False:
            filename = 'xicsrt_source'
            if name is not None:
                filename += '_'+str(name)
            if config['general_input']['output_suffix'] is not None:
                filename += config['general_input']['output_suffix']
            filepath = os.path.join(config['general_input']['output_path'], filename)
            print('Exporting source image:  {}'.format(filepath))
            source.output_image(filepath, rotate=False)

        # create crystal image file
        if True:
            filename = 'xicsrt_crystal'
            if name is not None:
                filename += '_'+str(name)
            if config['general_input']['output_suffix'] is not None:
                filename += config['general_input']['output_suffix']
            filepath = os.path.join(config['general_input']['output_path'], filename)
            print('Exporting crystal image:  {}'.format(filepath))
            crystal.output_image(filepath, rotate=False)

        # create detector image file
        if True:
            filename = 'xicsrt_detector'
            if name is not None:
                filename += '_'+str(name)
            if config['general_input']['output_suffix'] is not None:
                filename += config['general_input']['output_suffix']
            filepath = os.path.join(config['general_input']['output_path'], filename)
            print('Exporting detector image: {}'.format(filepath))
            detector.output_image(filepath, rotate=False)

    return output, meta


def run_multi(config_multi):

    profiler.start('XICSRT Run')

    # create the rays_total dictionary to count the total number of rays
    rays_total = OrderedDict()
    rays_total['total_generated'] = 0
    rays_total['total_crystal'] = 0
    rays_total['total_detector'] = 0

    output_final = []
    rays_final = []

    # loop through each configuration in the configuration input file
    for jj, key in enumerate(config_multi):

        ## Object Setup
        print('', flush=True)
        logging.info('')
        logging.info('Doing raytrace for Configuration: {} of {}'.format(
            jj + 1, len(config_multi)))

        if jj == 0:
            do_random_seed = True
        else:
            do_random_seed = False
        output, rays_count = run(config_multi[key], key, do_random_seed=do_random_seed)

        output_final.append(output)
        rays_final.append(rays_count)
        for key in rays_total:
            rays_total[key] += rays_count[key]

    # after all raytrace runs for all configurations, report the ray totals
    print('')
    print('Multi Config Rays Generated: {:6.4e}'.format(rays_total['total_generated']))
    print('Multi Config Rays on Crystal:{:6.4e}'.format(rays_total['total_crystal']))
    print('Multi Config Rays Detected:  {:6.4e}'.format(rays_total['total_detector']))
    print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
        rays_total['total_detector'] / rays_total['total_generated'],
        np.sqrt(rays_total['total_detector']) / rays_total['total_generated'],
        rays_total['total_detector'] / rays_total['total_generated'] * 100))
    print('')

    profiler.stop('XICSRT Run')

    return output_final, rays_final