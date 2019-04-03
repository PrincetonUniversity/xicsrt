# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:13:39 2017

@author: James
"""
import logging
from xicsrt.util import profiler

def raytrace(source, detector, *optics, number_of_runs=None, collect_optics=None):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), wavelength (W), and weight (w).
    """
    
    if number_of_runs is None: number_of_runs = 1
    if collect_optics is None: collect_optics = False
    
    total_generated = 0
    total_crystal = 0
    total_detector = 0
    
    for ii in range(number_of_runs):
        print('')
        print('Starting iteration: {} of {}'.format(ii, number_of_runs))
        profiler.start('Ray Generation')
        O, D, W, w  = source.generate_rays()
        profiler.stop('Ray Generation')
        start_number = len(D)
        print(' Rays Generated:    {:6.4e}'.format(D.shape[0]))
        total_generated += D.shape[0]

        for optic in optics:
            profiler.start('Ray Tracing')
            O, D, W, w = optic.light(O, D, W, w)
            profiler.stop('Ray Tracing')

            profiler.start('Collection: Optics')
            if collect_optics:
                optic.collect_rays(O, D, W, w)
            profiler.stop('Collection: Optics')

        total_crystal += D.shape[0]
        
        profiler.start('Collection: Detector')
        detector.collect_rays(O, D, W, w)
        profiler.stop('Collection: Detector')
        
        total_detector += detector.photon_count
        print(' Rays on Detector:  {:6.4e}'.format(detector.photon_count))    

    print('')
    print('Total Rays Generated: {:6.4e}'.format(total_generated))
    print('Total Rays Reflected: {:6.4e}'.format(total_crystal))
    print('Total Rays Detected:  {:6.4e}'.format(total_detector))
    print('Efficiency: {:6.4f}%'.format(total_detector/total_generated * 100))
    
    return O, D, W, w
        
        
def raytrace_special(source, detector, crystal, number_of_runs=None):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), wavelength (W), and weight (w).
    
    This function also lets the crystal collect the rays to allow for the
    determination of which rays satisfy the bragg condition.
    """

    if number_of_runs is None: number_of_runs = 1
    
    total_generated = 0
    total_crystal = 0
    total_detector = 0
    
    for ii in range(number_of_runs):
        profiler.start('Ray Generation')
        O, D, W, w = source.generate_rays()
        profiler.stop('Ray Generation')
        print('Rays Generated: ' + str(len(D)))
        total_generated += D.shape[0]
        
        profiler.start('Ray Tracing')
        O, D, W, w  = crystal.light(O, D, W, w)
        profiler.stop('Ray Tracing')
        print('Rays from Crystal: ' + str(len(D)))
        total_crystal += D.shape[0]
        
        profiler.start('Collection: Detector')
        detector.collect_rays(O, D, W, w)
        profiler.stop('Collection: Detector')

        # Collect all the rays that are reflected from the crystal.
        profiler.start('Collection: Crystal')
        crystal.collect_rays(O, D, W, w)
        profiler.stop('Collection: Crystal')
        
        # Collect only the rays that actually make it to the detector.
        #clause = detector.clause
        #O1, D1, W1, w1 = O[clause], D[clause], W[clause], w[clause]
        #crystal.collect_rays(O1, D1, W1, w)

    print('')
    print('Total Rays Generated: {:6.4e}'.format(total_generated))
    print('Total Rays Reflected: {:6.4e}'.format(total_crystal))
        
    return O, D, W, w
