# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    James Kring <jdk0026@tigermail.auburn.edu>

Entry point to XICSRT.
Contains the main functions that are called to perform raytracing.
"""

import numpy as np
from PIL import Image
import logging
import os

from copy import deepcopy
from collections import OrderedDict

from xicsrt.util import profiler

from xicsrt import xicsrt_config
from xicsrt import xicsrt_input
from xicsrt.objects._Dispatcher import Dispatcher
from xicsrt.objects._RayArray import RayArray


def raytrace(config):
    """
    Perform a series of ray tracing runs.

    Each run will rebuild all objects, reset the random seed and then
    perform the requested number of iterations.

    If the option 'save_images' is set, then images will be saved
    at the completion of each run. The saving of these run images
    is one reason to use this routine rather than just increasing
    the number of iterations: periodic outputs during long computations.

    Also see :func:`~xicsrt.xicsrt_multiprocessing.raytrace` for a
    multiprocessing version of this routine.
    """
    profiler.start('raytrace_multi')
    
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    check_config(config)

    # Make local copies of some options.
    num_runs = config['general']['number_of_runs']
    random_seed = config['general']['random_seed']
    
    output_list = []
    
    for ii in range(num_runs):
        logging.info('Starting run: {} of {}'.format(ii + 1, num_runs))
        config_run = deepcopy(config)
        config_run['general']['output_run_suffix'] = '{:04d}'.format(ii)

        # Make sure each run uses a unique random seed.
        if random_seed is not None:
            random_seed += ii
        config_run['general']['random_seed'] = random_seed
        
        iteration = raytrace_single(config_run, internal=True)
        output_list.append(iteration)
        
    output = combine_raytrace(output_list)

    # Reset the configuration options that were unique to the individual runs.
    output['config']['general']['output_run_suffix'] = config['general']['output_run_suffix']
    output['config']['general']['random_seed'] = config['general']['output_run_suffix']

    if config['general']['save_images']:
        save_images(output)
    if config['general']['print_results']:
        print_raytrace(output)
    if config['general']['save_config']:
        xicsrt_input.save_config(output['config'])
        
    profiler.stop('raytrace_multi')
    return output

def raytrace_single(config, internal=False):
    """
    Perform a single raytrace run consisting of multiple iterations.

    If history is enabled, sort the rays into those that are detected and
    those that are lost (found and lost). The found ray history will be
    returned in full. The lost ray history will be truncated to allow
    analysis of lost ray pattern while still limiting memory usage.
    """
    profiler.start('raytrace')

    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    config = xicsrt_config.config_to_numpy(config)
    check_config(config)

    logging.info('Seeding np.random with {}'.format(config['general']['random_seed']))
    np.random.seed(config['general']['random_seed'])

    num_iter = config['general']['number_of_iter']
    max_lost_iter = int(config['general']['history_max_lost']/num_iter)

    # Setup the dispatchers.
    if 'filters' in config:
        filters = Dispatcher(config, 'filters')
        filters.instantiate()
        filters.setup()
        filters.initialize()
        config['filters'] = filters.get_config()
    else:
        filters = None

    sources = Dispatcher(config, 'sources')
    sources.instantiate()
    sources.apply_filters(filters)
    sources.setup()
    sources.initialize()
    config['sources'] = sources.get_config()

    optics = Dispatcher(config, 'optics')
    optics.instantiate()
    optics.setup()
    optics.initialize()
    config['optics'] = optics.get_config()

    # Do the actual raytracing
    output_list = []
    for ii in range(num_iter):
        logging.info('Starting iteration: {} of {}'.format(ii + 1, num_iter))

        single = _raytrace_iter(config, sources, optics)
        sorted = _sort_raytrace(single, max_lost=max_lost_iter)
        output_list.append(sorted)

    output = combine_raytrace(output_list)

    if internal is False:
        if config['general']['print_results']:
            print_raytrace(output)
        if config['general']['save_config']:
            xicsrt_input.save_config(output['config'])

    if config['general']['save_images']:
        save_images(output)

    profiler.stop('raytrace')
    # profiler.report()
    return output

def _raytrace_iter(config, sources, optics):
    """ 
    Perform a single iteration of raytracing with the given sources and optics.
    The returned rays are unsorted.
    """
    profiler.start('raytrace_single')

    # Copy some config entries to local variables.
    # This is only to make tho code below more readable.
    keep_meta    = config['general']['keep_meta']
    keep_images  = config['general']['keep_images']
    keep_history = config['general']['keep_history']

    rays = sources.generate_rays(history=keep_history)
    rays = optics.trace(rays, history=keep_history, images=keep_images)

    # Combine sources and optics outputs.
    meta    = OrderedDict()
    image   = OrderedDict()
    history = OrderedDict()
    
    #if keep_meta:
    for key in sources.meta:
        meta[key] = sources.meta[key]
    for key in optics.meta:
        meta[key] = optics.meta[key]
    
    if keep_images:
        for key in sources.image:
            image[key] = sources.image[key]
        for key in optics.image:
            image[key] = optics.image[key]    
    
    if keep_history:     
        for key in sources.history:
            history[key] = sources.history[key]
        for key in optics.history:
            history[key] = optics.history[key]

    output = OrderedDict()
    output['config'] = config
    output['meta'] = meta
    output['image'] = image
    output['history'] = history
    
    profiler.stop('raytrace_single')
    return output


def _sort_raytrace(input, max_lost=None):
    """
    Sort the rays into 'lost' and 'found' rays, then truncate
    the number of lost rays.
    """
    if max_lost is None:
        max_lost = 1000

    profiler.start('_sort_raytrace')

    output = OrderedDict()
    output['config'] = input['config']
    output['total'] = OrderedDict()
    output['total']['meta'] = OrderedDict()
    output['total']['image'] = OrderedDict()
    output['found'] = OrderedDict()
    output['found']['meta'] = OrderedDict()
    output['found']['history'] = OrderedDict()
    output['lost'] = OrderedDict()
    output['lost']['meta'] = OrderedDict()
    output['lost']['history'] = OrderedDict()

    output['total']['meta'] = input['meta']
    output['total']['image'] = input['image']

    if len(input['history']) > 0:
        key_opt_list = list(input['history'].keys())
        key_opt_last = key_opt_list[-1]

        w_found = np.flatnonzero(input['history'][key_opt_last]['mask'])
        w_lost = np.flatnonzero(np.invert(input['history'][key_opt_last]['mask']))

        # Save only a portion of the lost rays so that our lost history does
        # not become too large.
        max_lost = min(max_lost, len(w_lost))
        index_lost = np.arange(len(w_lost))
        np.random.shuffle(index_lost)
        w_lost = w_lost[index_lost[:max_lost]]

        for key_opt in key_opt_list:
            output['found']['history'][key_opt] = OrderedDict()
            output['lost']['history'][key_opt] = OrderedDict()

            for key_ray in input['history'][key_opt]:
                output['found']['history'][key_opt][key_ray] = input['history'][key_opt][key_ray][w_found]
                output['lost']['history'][key_opt][key_ray] = input['history'][key_opt][key_ray][w_lost]

    profiler.stop('_sort_raytrace')

    return output

def combine_raytrace(input_list):
    """
    Produce a combined output from a list of raytrace outputs.
    """
    profiler.start('combine_raytrace')

    output = OrderedDict()
    output['config'] = input_list[0]['config']
    output['total'] = OrderedDict()
    output['total']['meta'] = OrderedDict()
    output['total']['image'] = OrderedDict()
    output['found'] = OrderedDict()
    output['found']['meta'] = OrderedDict()
    output['found']['history'] = OrderedDict()
    output['lost'] = OrderedDict()
    output['lost']['meta'] = OrderedDict()
    output['lost']['history'] = OrderedDict()

    num_iter = len(input_list)
    key_opt_list = list(input_list[0]['total']['meta'].keys())
    key_opt_last = key_opt_list[-1]

    # Combine the meta data.
    for key_opt in key_opt_list:
        output['total']['meta'][key_opt] = OrderedDict()
        key_meta_list = list(input_list[0]['total']['meta'][key_opt].keys())
        for key_meta in key_meta_list:
            output['total']['meta'][key_opt][key_meta] = 0
            for ii_run in range(num_iter):
                output['total']['meta'][key_opt][key_meta] += input_list[ii_run]['total']['meta'][key_opt][key_meta]

    # Combine the images.
    for key_opt in key_opt_list:
        if key_opt in input_list[0]['total']['image']:
            output['total']['image'][key_opt] = np.zeros(input_list[0]['total']['image'][key_opt].shape)
            for ii_run in range(num_iter):
                output['total']['image'][key_opt] += input_list[ii_run]['total']['image'][key_opt]

    # Combine all the histories.
    if len(input_list[ii_run]['found']['history']) > 0:
        final_num_found = 0
        final_num_lost = 0
        for ii_run in range(num_iter):
            final_num_found += len(input_list[ii_run]['found']['history'][key_opt_last]['mask'])
            final_num_lost += len(input_list[ii_run]['lost']['history'][key_opt_last]['mask'])

        rays_found_temp = RayArray()
        rays_found_temp.zeros(final_num_found)

        rays_lost_temp = RayArray()
        rays_lost_temp.zeros(final_num_lost)

        for key_opt in key_opt_list:
            output['found']['history'][key_opt] = rays_found_temp.copy()
            output['lost']['history'][key_opt] = rays_lost_temp.copy()

        index_found = 0
        index_lost = 0
        for ii_run in range(num_iter):
            num_found = len(input_list[ii_run]['found']['history'][key_opt_last]['mask'])
            num_lost = len(input_list[ii_run]['lost']['history'][key_opt_last]['mask'])

            for key_opt in key_opt_list:
                for key_ray in output['found']['history'][key_opt]:
                    output['found']['history'][key_opt][key_ray][index_found:index_found + num_found] = (
                        input_list[ii_run]['found']['history'][key_opt][key_ray][:])
                    output['lost']['history'][key_opt][key_ray][index_lost:index_lost + num_lost] = (
                        input_list[ii_run]['lost']['history'][key_opt][key_ray][:])

            index_found += num_found
            index_lost += num_lost

    profiler.stop('combine_raytrace')
    return output

def check_config(config):
    """
    Check the general section of the configuration dictionary.
    """

    # Check if anything needs to be saved.
    do_save = False
    for key in config['general']:
        if 'save' in key:
            if config['general'][key]:
                do_save = True

    if do_save:
        if not os.path.exists(config['general']['output_path']):
            if not config['general']['make_directories']:
                raise Exception('Output directory does not exist. Create directory or set make_directories to True.')

def print_raytrace(results):
    """
    Print out some information and statistics from the raytracing results.
    """

    key_opt_list = list(results['total']['meta'].keys())
    num_source = results['total']['meta'][key_opt_list[0]]['num_out']
    num_detector = results['total']['meta'][key_opt_list[-1]]['num_out']
    
    print('')
    print('Rays Generated: {:6.3e}'.format(num_source))
    print('Rays Detected:  {:6.3e}'.format(num_detector))
    print('Efficiency:     {:6.3e} ± {:3.1e} ({:7.5f}%)'.format(
        num_detector / num_source,
        np.sqrt(num_detector) / num_source,
        num_detector / num_source * 100))
    print('')

def save_images(results):
    """
    Save images from the raytracing results.
    """

    rotate = False

    prefix = results['config']['general']['output_prefix']
    suffix = results['config']['general']['output_suffix']
    run_suffix = results['config']['general']['output_run_suffix']
    ext = results['config']['general']['image_extension']

    if results['config']['general']['make_directories']:
        os.makedirs(results['config']['general']['output_path'], exist_ok=True)

    for key_opt in results['config']['optics']:
        if key_opt in results['total']['image']:
            filename = '_'.join(filter(None, (prefix, key_opt, suffix, run_suffix)))+ext
            filepath = os.path.join(results['config']['general']['output_path'], filename)
            
            image_temp = results['total']['image'][key_opt]
            if rotate:
                image_temp = np.rot90(image_temp)
            
            generated_image = Image.fromarray(image_temp)
            generated_image.save(filepath)
            
            logging.info('Saved image: {}'.format(filepath))
        
