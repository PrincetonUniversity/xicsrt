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
    
    output = OrderedDict()
    output['meta'] = meta
    output['image'] = image
    output['history'] = history

    return output

def sort_raytrace(input, max_lost=None):
    if max_lost is None:
        max_lost = 1000
        
    output = OrderedDict()
    output['total'] = OrderedDict()
    output['total']['meta'] = OrderedDict()
    output['found'] = OrderedDict()
    output['found']['meta'] = OrderedDict()
    output['found']['history'] = OrderedDict()
    output['lost'] = OrderedDict()
    output['lost']['meta'] = OrderedDict()
    output['lost']['history'] = OrderedDict()
    
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

    output['total']['meta'] = input['meta']

    return output
    
def combine_raytrace(input_list):
    """
    Produce a combined output from a list of raytrace_single outputs into a combined 
    """
    output = OrderedDict()
    output['total'] = OrderedDict()
    output['total']['meta'] = OrderedDict()
    output['found'] = OrderedDict()
    output['found']['meta'] = OrderedDict()
    output['found']['history'] = OrderedDict()
    output['lost'] = OrderedDict()
    output['lost']['meta'] = OrderedDict()
    output['lost']['history'] = OrderedDict()

    num_runs = len(input_list)
    key_opt_list = list(input_list[0]['found']['history'].keys())
    key_opt_last = key_opt_list[-1]
        
    # Combine the meta data
    for key_opt in key_opt_list:
        output['total']['meta'][key_opt] = OrderedDict()
        key_meta_list = list(input_list[0]['total']['meta'][key_opt].keys())
        for key_meta in key_meta_list:
            output['total']['meta'][key_opt][key_meta] = 0
            for ii_run in range(num_runs):
                output['total']['meta'][key_opt][key_meta] += input_list[ii_run]['total']['meta'][key_opt][key_meta]
        
    # Combine all the histories
    final_num_found = 0
    final_num_lost = 0
    for ii_run in range(num_runs):
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
    for ii_run in range(num_runs):
        num_found = len(input_list[ii_run]['found']['history'][key_opt_last]['mask'])
        num_lost = len(input_list[ii_run]['lost']['history'][key_opt_last]['mask'])

        for key_opt in key_opt_list:
            for key_ray in output['found']['history'][key_opt]:
                output['found']['history'][key_opt][key_ray][index_found:index_found+num_found] = (
                    input_list[ii_run]['found']['history'][key_opt][key_ray][:])
                output['lost']['history'][key_opt][key_ray][index_lost:index_lost+num_lost] = (
                    input_list[ii_run]['lost']['history'][key_opt][key_ray][:])
    
    return output
       
def raytrace(config):
    """
    Run a series of ray tracing runs.  Save all rays that make it to the detector
    and a subset of rays that are lost.
    """

    # This needs to be moved to a default config method.
    if config['general']['number_of_runs'] is None:
        config['general']['number_of_runs'] = number_of_runs = 1
    num_runs = config['general']['number_of_runs']
    
    output_list = []
    for ii in range(num_runs):
        logging.info('Starting iteration: {} of {}'.format(ii + 1, num_runs))
        
        single = raytrace_single(config)
        sorted = sort_raytrace(single)
        output_list.append(sorted)
        
    output = combine_raytrace(output_list)

    return output

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

    
    #output['sources'] = sources
    #output['optics'] = optics
    
    #meta = {}
    #meta['total_rays'] = np.len(rays['mask'])
    
    #print('')
    #print('Rays Generated: {:6.4e}'.format(meta['total_rays'])
    #print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
    #    rays_count['total_detector'] / rays_count['total_generated'],
    #    np.sqrt(rays_count['total_detector']) / rays_count['total_generated'],
    #    rays_count['total_detector'] / rays_count['total_generated'] * 100))
    #print('')
