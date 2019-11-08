# -*- coding: utf-8 -*-
"""
@author: James
@editor: Eugene
"""

import numpy as np
from copy import deepcopy

from xicsrt.util import profiler

from xicsrt.xics_rt_objects import RayArray

def raytrace_single(source, detector, *optics,  number_of_runs=None, collect_optics=None):
    """ 
    Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin, direction, wavelength, and weight.
    """

    if collect_optics is None: collect_optics = False
    
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
        if collect_optics:
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

    return rays_history, rays_count


def raytrace(source, detector, *optics, number_of_runs=None, collect_optics=None):
    """
    Run a series of ray tracing runs.  Save all rays that make it to the detector
    and a subset of rays that are lost.
    """

    if number_of_runs is None: number_of_runs = 1

    count = dict()
    count['total_generated']  = 0
    count['total_graphite']   = 0
    count['total_crystal']    = 0
    count['total_detector']   = 0

    history = {}
    history['found'] = []
    history['lost'] = []

    for ii in range(number_of_runs):
        print('')
        print('Starting iteration: {} of {}'.format(ii + 1, number_of_runs))

        history_temp, count_temp = raytrace_single(source, detector, *optics, collect_optics=collect_optics)

        for key in count_temp:
            count[key] += count_temp[key]

        w_found = np.flatnonzero(history_temp[-1]['mask'])
        w_lost = np.flatnonzero(np.invert(history_temp[-1]['mask']))

        # Save only a portion of the lost rays so that our lost history does
        # not become too large.
        max_lost = 5000
        lost_max = min(max_lost, len(w_lost))
        index_lost = np.arange(len(w_lost))
        np.random.shuffle(index_lost)
        w_lost = w_lost[index_lost[:lost_max]]

        found = []
        lost = []
        for ii_opt in range(len(history_temp)):
            found.append({})
            lost.append({})

            for key in history_temp[ii_opt]:
                found[ii_opt][key] = history_temp[ii_opt][key][w_found]
                lost[ii_opt][key] = history_temp[ii_opt][key][w_lost]

        history['found'].append(found)
        history['lost'].append(lost)

    # Now combine all the histories.
    ray_temp = RayArray()
    ray_temp.zeros(0)

    history_final = {}
    history_final['found'] = []
    history_final['lost'] = []
    for ii_opt in range(len(history['found'][0])):
        history_final['found'].append(ray_temp.copy())
        history_final['lost'].append(ray_temp.copy())

    for ii_run in range(len(history['found'])):
        for ii_opt in range(len(history['found'][0])):
            history_final['found'][ii_opt].extend(history['found'][ii_run][ii_opt])
            history_final['lost'][ii_opt].extend(history['lost'][ii_run][ii_opt])

    print('')
    print('Final Rays Generated: {:6.4e}'.format(count['total_generated']))
    print('Final Rays on HOPG:   {:6.4e}'.format(count['total_graphite']))
    print('Final Rays on Crystal:{:6.4e}'.format(count['total_crystal']))
    print('Final Rays Detected:  {:6.4e}'.format(count['total_detector']))
    print('Efficiency: {:6.2e} ± {:3.1e} ({:7.5f}%)'.format(
        count['total_detector'] / count['total_generated'],
        np.sqrt(count['total_detector']) / count['total_generated'],
        count['total_detector'] / count['total_generated'] * 100))
    print('')

    return history_final, count


