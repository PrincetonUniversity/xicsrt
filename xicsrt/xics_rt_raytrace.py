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
    
    rays_count = dict()
    rays_count['total_generated']  = 0
    rays_count['total_graphite']   = 0
    rays_count['total_crystal']    = 0
    rays_count['total_detector']   = 0

    profiler.start('Raytrace Run')

    # Rays history resets after every run and only returns on the last one
    rays_history = []

    profiler.start('Ray Generation')
    rays = source.generate_rays()
    profiler.stop('Ray Generation')
    rays_history.append(deepcopy(rays))


    print(' Rays Generated:    {:6.4e}'.format(rays['direction'].shape[0]))
    rays_count['total_generated'] += rays['direction'].shape[0]

    for optic in optics:
        profiler.start('Ray Tracing')
        rays = optic.light(rays)
        profiler.stop('Ray Tracing')
        rays_history.append(deepcopy(rays))

        profiler.start('Collection: Optics')
        optic.collect_rays(rays)
        profiler.stop('Collection: Optics')

        if optic.__name__ == 'SphericalCrystal':
            rays_count['total_crystal'] += optic.photon_count
        elif optic.__name__ == 'MosaicGraphite':
            rays_count['total_graphite'] += optic.photon_count

    profiler.start('Ray Tracing')
    rays = detector.light(rays)
    profiler.stop('Ray Tracing')
    rays_history.append(deepcopy(rays))

    profiler.start('Collection: Detector')
    detector.collect_rays(rays)
    profiler.stop('Collection: Detector')

    rays_count['total_detector'] += detector.photon_count
    profiler.stop('Raytrace Run')

    """
    print('')
    print('Total Rays Generated: {:6.4e}'.format(rays_count['total_generated']))
    print('Total Rays on HOPG:   {:6.4e}'.format(rays_count['total_graphite']))
    print('Total Rays on Crystal:{:6.4e}'.format(rays_count['total_crystal']))
    print('Total Rays Detected:  {:6.4e}'.format(rays_count['total_detector']))
    print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
        rays_count['total_detector'] / rays_count['total_generated'],
        np.sqrt(rays_count['total_detector']) / rays_count['total_generated'],
        rays_count['total_detector'] / rays_count['total_generated'] * 100))
    print('')
    """

    return rays_history, rays_count


def raytrace(number_of_runs, source, detector, *optics):
    """
    Run a series of ray tracing runs.  Save all rays that make it to the detector
    and a subset of rays that are lost.
    """

    if number_of_runs is None: number_of_runs = 1

    rays_count = dict()
    rays_count['total_generated']  = 0
    rays_count['total_graphite']   = 0
    rays_count['total_crystal']    = 0
    rays_count['total_detector']   = 0

    history = {}
    history['found'] = []
    history['lost'] = []
    
    rays_history = {}
    rays_history['found'] = []
    rays_history['lost'] = []

    for ii in range(number_of_runs):
        print('')
        print('Starting iteration: {} of {}'.format(ii + 1, number_of_runs))

        single_history, single_count = raytrace_single(
                source, detector, *optics)

        for key in single_count:
            rays_count[key] += single_count[key]

        w_found = np.flatnonzero(single_history[-1]['mask'])
        w_lost  = np.flatnonzero(np.invert(single_history[-1]['mask']))
        
        #to avoid saving too many rays, randomly cull rays until there are 5000
        if len(w_lost) >= 5000:
            cutter = np.random.randint(0, len(w_lost), len(w_lost))
            w_lost = w_lost[cutter <= 5000]

        found = []
        lost  = []
        for optic in range(len(single_history)):
            found.append({})
            lost.append({})

            for key in single_history[optic]:
                found[optic][key] = single_history[optic][key][w_found]
                lost[optic][key]  = single_history[optic][key][w_lost]

        history['found'].append(found)
        history['lost'].append(lost)

    # Now combine all the histories.
    ray_temp = RayArray()
    ray_temp.zeros(0)


    for optic in range(len(history['found'][0])):
        rays_history['found'].append(ray_temp.copy())
        rays_history['lost'].append(ray_temp.copy())

    for run in range(len(history['found'])):
        for optic in range(len(history['found'][0])):
            rays_history['found'][optic].extend(history['found'][run][optic])
            rays_history['lost'][optic].extend(history['lost'][run][optic])

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

    return rays_history, rays_count


