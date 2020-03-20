# -*- coding: utf-8 -*-
"""
@author: James
@editor: Eugene
"""

import numpy as np
import logging

from copy import deepcopy
from collections import OrderedDict

from xicsrt.util import profiler

from xicsrt.xicsrt_dispatch import XicsrtDispatcher
from xicsrt.xicsrt_objects import RayArray

def raytrace_single(config):
    """ 
    Rays are generated from sources and then passed through 
    the optics in the order listed in the configuration file.

    Rays consists of origin, direction, wavelength, and weight.
    """
    profiler.start('raytrace_single')
    
    sources = XicsrtDispatcher(config['sources'], config['general']['class_pathlist'])
    optics  = XicsrtDispatcher(config['optics'],  config['general']['class_pathlist'])

    sources.instantiate_objects()
    sources.initialize()
    rays = sources.generate_rays(history=True)
    
    optics.instantiate_objects()
    optics.initialize()
    rays = optics.raytrace(rays, history=True)

    profiler.stop('raytrace_single')

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

    output = {}
    output['meta'] = meta
    output['image'] = image
    output['history'] = history
    
    #meta = {}
    #meta['total_rays'] = np.len(rays['mask'])
    
    #print('')
    #print('Rays Generated: {:6.4e}'.format(meta['total_rays'])
    #print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
    #    rays_count['total_detector'] / rays_count['total_generated'],
    #    np.sqrt(rays_count['total_detector']) / rays_count['total_generated'],
    #    rays_count['total_detector'] / rays_count['total_generated'] * 100))
    #print('')

    return output

def raytrace(config):
    """
    Run a series of ray tracing runs.  Save all rays that make it to the detector
    and a subset of rays that are lost.
    """

    # This needs to be move to a default config method.
    if config['general']['number_of_runs'] is None:
        config['general']['number_of_runs'] = number_of_runs = 1
    num_runs = config['general']['number_of_runs']
        
    meta = {}
    
    history = OrderedDict()
    history['found'] = []
    history['lost'] = []

    for ii in range(num_runs):
        logging.info('Starting iteration: {} of {}'.format(ii + 1, num_runs))

        output = raytrace_single(config)

        key_list = list(output.history.keys())
        key_last = key_list[-1]

        w_found = np.flatnonzero(output.history[key_last]['mask'])
        w_lost  = np.flatnonzero(np.invert(output.history[key_last]['mask']))

        # Save only a portion of the lost rays so that our lost history does
        # not become too large.
        max_lost = int(10000/number_of_runs)
        max_lost = min(max_lost, len(w_lost))
        index_lost = np.arange(len(w_lost))
        np.random.shuffle(index_lost)
        w_lost = w_lost[index_lost[:max_lost]]

        found = OrderedDict()
        lost  = OrderedDict()
        for key_opt in key_list:
            found[key_opt] = {}
            lost[key_opt] = {}

            for key_ray in output.history[key_opt]:
                found[key_opt][key_ray] = output.history[key_opt][key_ray][w_found]
                lost[key_opt][key_ray] = output.history[key_opt][key_ray][w_found]

        history['found'].append(found)
        history['lost'].append(lost)


    # Calculate total rays in truncated history.
    final_num_found = 0
    final_num_lost = 0
    for ii_run in range(num_runs):
        final_num_found += len(history['found'][ii_run][key_last]['mask'])
        final_num_lost += len(history['lost'][ii_run][key_last]['mask'])
    
    # Now combine all the histories.
    rays_found_temp = RayArray()
    rays_found_temp.zeros(final_num_found)
    
    rays_lost_temp = RayArray()
    rays_lost_temp.zeros(final_num_lost)

    final_history = OrderedDict()
    final_history['found'] = OrderedDict()
    final_history['lost'] = OrderedDict()

    for key in key_list:
        final_history['found'][key] = rays_found_temp.copy()
        final_history['lost'][key] = rays_lost_temp.copy()

    index_found = 0
    index_lost = 0
    for ii_run in range(num_runs):
        num_found = history['found'][ii_run][key_last]['mask']
        num_lost = history['lost'][ii_run][key_last]['mask']

        for key_opt in key_list:
            for key_ray in final_history['found'][key_opt]:
                final_history['found'][key_opt][key_ray][index_found:index_found+num_found] = (
                    history['found'][ii_run][key_opt][key_ray][:])
                final_history['lost'][key_opt][key_ray][index_lost:index_lost+num_lost] = (
                    history['lost'][ii_run][key_opt][key_ray][:])

    #if count['total_generated'] == 0:
    #    raise ValueError('No rays generated. Check inputs.')
    
    #print('')
    #print('Final Rays Generated: {:6.4e}'.format(count['total_generated']))
    #print('Final Rays on HOPG:   {:6.4e}'.format(count['total_graphite']))
    #print('Final Rays on Crystal:{:6.4e}'.format(count['total_crystal']))
    #print('Final Rays Detected:  {:6.4e}'.format(count['total_detector']))
    #print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
    #    count['total_detector'] / count['total_generated'],
    #    np.sqrt(count['total_detector']) / count['total_generated'],
    #    count['total_detector'] / count['total_generated'] * 100))
    #print('')

    return final_history


