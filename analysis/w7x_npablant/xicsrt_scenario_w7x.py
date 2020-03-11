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

from multiprocessing import Pool

from xicsrt.xicsrt_raytrace   import raytrace
from xicsrt import xicsrt_input

from xicsrt.optics._XicsrtOpticDetector import XicsrtOpticDetector
from xicsrt.optics._XicsrtOpticCrystalSpherical import XicsrtOpticCrystalSpherical
from xicsrt.sources._XicsrtPlasmaW7xSimple import XicsrtPlasmaW7xSimple

profiler.stop('Import Time')

def initialize(config):

    # Setup our plasma box to be radial.
    config['source_input']['zaxis'] = config['source_input']['origin']
    config['source_input']['zaxis'] /= np.linalg.norm(config['source_input']['zaxis'])

    config['source_input']['xaxis'] = np.cross(config['source_input']['zaxis'], np.array([0,0,1]))
    config['source_input']['xaxis'] /= np.linalg.norm(config['source_input']['xaxis'])

    config['source_input']['target'] = config['crystal_input']['origin']

    xicsrt_input.config_to_numpy(config)
    return config

def run(config, name=None, do_random_seed=True):

    if do_random_seed:
        # Initialize the random seed.
        logging.info('Seeding np.random with {}'.format(config['general_input']['random_seed']))
        np.random.seed(config['general_input']['random_seed'])

    profiler.start('Class Setup Time')

    detector = XicsrtOpticDetector(config['detector_input'], strict=False)
    crystal = XicsrtOpticCrystalSpherical(config['crystal_input'], strict=False)
    source = XicsrtPlasmaW7xSimple(config['source_input'], strict=False)

    objects = {'source':source, 'crystal':crystal, 'detector':detector}
    
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

    return output, meta, objects


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

        output, rays_count, objects = run(config_multi[key], key, do_random_seed=do_random_seed)

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


def run_multiprocessing(config_multi):

    profiler.start('XICSRT Run')

    # create the rays_total dictionary to count the total number of rays
    rays_total = OrderedDict()
    rays_total['total_generated'] = 0
    rays_total['total_crystal'] = 0
    rays_total['total_detector'] = 0

    output_final = []
    rays_final = []
    objects_final = []

    result_list = []
    with Pool() as pool:
        # loop through each configuration in the configuration input file
        # and add a new run into the pool.
        for ii, name in enumerate(config_multi):
            logging.info('Launching raytrace for Configuration: {}'.format(name))

            # Make sure each run uses a unique random seed.
            if config_multi[name]['general_input']['random_seed'] is not None:
                config_multi[name]['general_input']['random_seed'] += ii
            arg = (config_multi[name], name)
            result = pool.apply_async(run, arg)
            result_list.append(result)
        pool.close()
        pool.join()

    # Gather all the results together.
    for result in result_list:
        output, rays_count, objects = result.get()

        output_final.append(output)
        rays_final.append(rays_count)
        objects_final.append(objects)
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

    return output_final, rays_final, objects_final
