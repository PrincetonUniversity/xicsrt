# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir pablant <npablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  - James Kring <jdk0026@tigermail.auburn.edu>
"""

import numpy as np
from PIL import Image
import logging
import os

from copy import deepcopy
from collections import OrderedDict

from xicsrt.util import profiler

from xicsrt import xicsrt_config
from xicsrt.xicsrt_dispatch import XicsrtDispatcher
from xicsrt.xicsrt_objects import RayArray

def raytrace(config, internal=False):
    """
    Perform a series of ray tracing iterations.

    If history is enabled, sort the rays into those that are detected and 
    those that are lost (found and lost). The found ray history will be 
    returned in full. The lost ray history will be truncated to allow 
    visualizaiton while still limited memory usage.
    """
    profiler.start('raytrace')

    logging.info('Seeding np.random with {}'.format(config['general']['random_seed']))
    np.random.seed(config['general']['random_seed'])

    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    
    num_iter = config['general']['number_of_iter']
    
    output_list = []
    for ii in range(num_iter):
        logging.info('Starting iteration: {} of {}'.format(ii + 1, num_iter))
        
        single = raytrace_single(config)
        sorted = sort_raytrace(single)
        output_list.append(sorted)
        
    output = combine_raytrace(output_list)

    if internal is False:
        if config['general']['save_images']:
            save_images(output)
        if config['general']['print_results']:
            print_raytrace(output)

    if config['general']['save_run_images']:
        save_images(output)

    profiler.stop('raytrace')
    #profiler.report()
    return output

def raytrace_multi(config):
    """
    Perform a series of ray tracing runs.

    Each raytracing run will perform the requested number of iterations.
    Each run will produce a single output image.
    """
    profiler.start('raytrace_multi')
    
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)

    
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
        
        iteration = raytrace(config_run, internal=True)
        output_list.append(iteration)
        
    output = combine_raytrace(output_list)
    output['config'] = config

    if config['general']['save_images']:
        save_images(output)
    if config['general']['print_results']:
        print_raytrace(output)
        
    profiler.stop('raytrace_multi')
    return output
    
def raytrace_single(config):
    """ 
    Rays are generated from sources and then passed through 
    the optics in the order listed in the configuration file.

    Rays consists of origin, direction, wavelength, and weight.
    """
    profiler.start('raytrace_single')
    
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    config = xicsrt_config.config_to_numpy(config)

    # Combine the user and default object pathlists.
    pathlist = []
    pathlist.extend(config['general']['pathlist_objects'])
    pathlist.extend(config['general']['pathlist_default'])

    # Setup the dispatchers.
    sources = XicsrtDispatcher(config['sources'], pathlist)
    optics  = XicsrtDispatcher(config['optics'],  pathlist)

    sources.instantiate_objects()
    sources.initialize()
    rays = sources.generate_rays(history=False)
    
    optics.instantiate_objects()
    optics.initialize()
    rays = optics.raytrace(rays, history=False, images=True)

    # Combine sources and optics.
    meta = OrderedDict()
    image = OrderedDict()
    history = OrderedDict()
    
    for key in sources.meta:
        meta[key] = sources.meta[key]
    for key in sources.image:
        image[key] = sources.image[key]
    for key in sources.history:
        history[key] = sources.history[key]

    for key in optics.meta:
        meta[key] = optics.meta[key]
    for key in optics.image:
        image[key] = optics.image[key]
    for key in optics.history:
        history[key] = optics.history[key]   
    
    output = OrderedDict()
    output['config'] = config
    output['meta'] = meta
    output['image'] = image
    output['history'] = history

    profiler.stop('raytrace_single')
    return output

def sort_raytrace(input, max_lost=None):
    if max_lost is None:
        max_lost = 1000
    
    profiler.start('sort_raytrace')
    
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
        w_lost  = np.flatnonzero(np.invert(input['history'][key_opt_last]['mask']))

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

    profiler.stop('sort_raytrace')
        
    return output
    
def combine_raytrace(input_list):
    """
    Produce a combined output from a list of raytrace_single outputs into a combined 
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

    # Combine all the histories
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
                    output['found']['history'][key_opt][key_ray][index_found:index_found+num_found] = (
                        input_list[ii_run]['found']['history'][key_opt][key_ray][:])
                    output['lost']['history'][key_opt][key_ray][index_lost:index_lost+num_lost] = (
                        input_list[ii_run]['lost']['history'][key_opt][key_ray][:])
 
    profiler.stop('combine_raytrace')
    return output
       
def print_raytrace(results):

    key_opt_list = list(results['total']['meta'].keys())
    num_source = results['total']['meta'][key_opt_list[0]]['num_out']
    num_detector = results['total']['meta'][key_opt_list[-1]]['num_out']
    
    print('')
    print('Rays Generated: {:6.3e}'.format(num_source))
    print('Rays Detected:  {:6.3e}'.format(num_detector))
    print('Efficiency:     {:6.3e} ± {:3.1e} ({:7.5f}%)'.format(
        num_detector / num_source,
        np.sqrt(num_detector / num_source),
        num_detector / num_source * 100))
    print('')

def save_images(results):

    rotate = False

    prefix = results['config']['general']['output_prefix']
    suffix = results['config']['general']['output_suffix']
    run_suffix = results['config']['general']['output_run_suffix']
    ext = results['config']['general']['image_extension']
    
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
        