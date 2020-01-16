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
    
    single_count = dict()
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
    The lest raytrace run also returns a rays history which describes everything
    about the run, making it easier to visualize.
    """

    if number_of_runs is None: number_of_runs = 1

    rays_count = dict()
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
            
    rays_history = single_history

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
