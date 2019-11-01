# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:13:39 2017
Edited on Fri Sep 06 11:37:11 2019

@author: James
@editor: Eugene
"""

import numpy as np
from copy import deepcopy

from xicsrt.util import profiler

def raytrace(source, detector, *optics,  number_of_runs=None, collect_optics=None):
    """ 
    Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin, direction, wavelength, and weight.
    """
    
    if number_of_runs is None: number_of_runs = 1
    if collect_optics is None: collect_optics = False
    
    rays_count = dict()
    rays_count['total_generated']  = 0
    rays_count['total_graphite']   = 0
    rays_count['total_crystal']    = 0
    rays_count['total_detector']   = 0
    
    for ii in range(number_of_runs):
        profiler.start('Raytrace Run')
        print('')
        print('Starting iteration: {} of {}'.format(ii + 1, number_of_runs))
        
        # Rays history resets after every run and only returns on the last one
        rays_history = []
            
        profiler.start('Ray Generation')
        rays = source.generate_rays()
        rays_history.append(deepcopy(rays))
        profiler.stop('Ray Generation')
                
        print(' Rays Generated:    {:6.4e}'.format(rays['direction'].shape[0]))
        rays_count['total_generated'] += rays['direction'].shape[0]

        for optic in optics:
            profiler.start('Ray Tracing')
            rays = optic.light(rays)
            rays_history.append(deepcopy(rays))
            profiler.stop('Ray Tracing')

            profiler.start('Collection: Optics')
            if collect_optics:
                optic.collect_rays(rays)
            profiler.stop('Collection: Optics')
            
            if optic.__name__ == 'SphericalCrystal':
                rays_count['total_crystal'] += optic.photon_count
            elif optic.__name__ == 'MosaicGraphite':
                rays_count['total_graphite'] += optic.photon_count

        profiler.start('Collection: Detector')
        rays = detector.light(rays)
        rays_history.append(deepcopy(rays))
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
