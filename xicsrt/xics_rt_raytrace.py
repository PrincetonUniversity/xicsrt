# -*- coding: utf-8 -*-
"""
@author: James
@editor: Eugene
"""

import numpy as np
from copy import deepcopy

from xicsrt.util import profiler

from xicsrt.xics_rt_objects import RayArray

def raytrace_single(source, detector, *optics):
    """
    Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin, direction, wavelength, and weight.
    """
    
    single_count = {}
    single_count['total_generated']  = 0
    single_count['total_graphite']   = 0
    single_count['total_crystal']    = 0
    single_count['total_detector']   = 0

    profiler.start('Raytrace Run')

    # Rays history resets after every run and only returns on the last one
    single_history = []

    profiler.start('Ray Generation')
    rays = source.generate_rays()
    profiler.stop('Ray Generation')
    single_history.append(deepcopy(rays))

    print(' Rays Generated:    {:6.4e}'.format(rays['direction'].shape[0]))
    single_count['total_generated'] += rays['direction'].shape[0]

    for optic in optics:
        profiler.start('Ray Tracing')
        rays = optic.light(rays)
        profiler.stop('Ray Tracing')
        single_history.append(deepcopy(rays))

        profiler.start('Collection: Optics')
        optic.collect_rays(rays)
        profiler.stop('Collection: Optics')

        if optic.__name__ == 'SphericalCrystal':
            single_count['total_crystal'] += optic.photon_count
        elif optic.__name__ == 'MosaicGraphite':
            single_count['total_graphite'] += optic.photon_count

    profiler.start('Ray Tracing')
    rays = detector.light(rays)
    profiler.stop('Ray Tracing')
    single_history.append(deepcopy(rays))

    profiler.start('Collection: Detector')
    detector.collect_rays(rays)
    profiler.stop('Collection: Detector')

    single_count['total_detector'] += detector.photon_count
    profiler.stop('Raytrace Run')

    return single_history, single_count


def raytrace(number_of_runs, source, detector, *optics):
    """
    Run a series of ray tracing runs and add up all the results in the end.
    Then, return two raytrace outputs: 'rays_history' contains a truncated copy
    of all rays in the raytrace run, while 'hits_history' contains the history
    of only the rays that hit the detector.
    """

    if number_of_runs is None: number_of_runs = 1
    
    hits_history = []
    
    rays_history = []

    rays_count   = {}
    rays_count['total_generated']  = 0
    rays_count['total_graphite']   = 0
    rays_count['total_crystal']    = 0
    rays_count['total_detector']   = 0

    for ii in range(number_of_runs):
        print('')
        print('Starting iteration: {} of {}'.format(ii + 1, number_of_runs))

        single_history, single_count = raytrace_single(source, detector, *optics)

        for key in single_count:
            rays_count[key] += single_count[key]
            
        #find which rays hit the detector, these are the hits
        #create two large masks to keep track
        hits = np.flatnonzero(single_history[-1]['mask'])
        miss = np.flatnonzero(np.invert(single_history[-1]['mask']))
        
        #to avoid saving too many miss, randomly cull rays until there are 5000
        #hits will not be culled
        max_len = int(5000 / number_of_runs)
        if len(miss) >= max_len:
            cutter = np.random.randint(0, len(miss), len(miss))
            miss = miss[cutter <= max_len]
            
        #use the hits/miss masks to cut single_history into hit and missed rays
        single_hits_history = []
        single_miss_history = []
        
        for optic in range(len(single_history)):
            single_hits_history.append({})
            single_miss_history.append({})
            
            for key in single_history[optic]:
                single_hits_history[optic][key] = single_history[optic][key][hits]
                single_miss_history[optic][key] = single_history[optic][key][miss]
        
        #append the single histories to the global histories
        if hits_history == []:
            hits_history = single_hits_history
        else:
            for optic in range(len(hits_history)):
                for key in hits_history[optic]:
                    hits_history[optic][key] = np.append(
                        hits_history[optic][key], single_hits_history[optic][key], axis = 0)
                
        if rays_history == []:
            rays_history = single_miss_history
        else:
            for optic in range(len(rays_history)):
                for key in rays_history[optic]:
                    rays_history[optic][key] = np.append(
                        rays_history[optic][key], single_miss_history[optic][key], axis = 0)
        
    print('')
    print('Final Rays Generated: {:6.4e}'.format(rays_count['total_generated']))
    print('Final Rays on HOPG:   {:6.4e}'.format(rays_count['total_graphite']))
    print('Final Rays on Crystal:{:6.4e}'.format(rays_count['total_crystal']))
    print('Final Rays Detected:  {:6.4e}'.format(rays_count['total_detector']))
    print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
        rays_count['total_detector'] / rays_count['total_generated'],
        np.sqrt(rays_count['total_detector']) / rays_count['total_generated'],
        rays_count['total_detector'] / rays_count['total_generated'] * 100))
    print('')

    return hits_history, rays_history, rays_count
