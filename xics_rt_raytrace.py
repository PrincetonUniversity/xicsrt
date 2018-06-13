# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:13:39 2017

@author: James
"""
import time

def raytrace(source, detector, *optics):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), wavelength (W), and weight (w).
    """
    t2 = time.time()
    O, D, W, w  = source.generate_rays()
    print("Took " + str(round(time.time() - t2, 4)) + ' sec: Ray Generation TIme')
    start_number = len(D)
    print('Rays Generated: ' + str(len(D)))

    for optic in optics:
        t2 = time.time()
        O, D, W, w = optic.light(O, D, W, w)
        
        print("Took " + str(round(time.time() - t2, 4)) + ' sec: Optic Time')
        #optic.collect_rays(O, D, W)
        
        print('Rays from Crystal: ' + str(len(D)))
    t2 = time.time()
    detector.collect_rays(O, D, W, w)
    print("Took " + str(round(time.time() - t2, 4)) + ' sec: Collection Time')
    print('Efficiency: ' + str(round((detector.photon_count/start_number) * 100,5)) + '%')
    return      
        
        
def raytrace_special( source, detector, crystal):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), wavelength (W), and weight (w).
    
    This function also lets the crystal collect the rays to allow for the
    determination of which rays satisfy the bragg condition.
    """
    t2 = time.time()
    O, D, W, w = source.generate_rays()
    print("Took " + str(round(time.time() - t2, 4)) + ' sec: Ray Generation TIme')
    
    print('Rays Generated: ' + str(len(D)))
    
    O, D, W, w  = crystal.light(O, D, W, w)
    print('Rays from Crystal: ' + str(len(D)))
    
    detector.collect_rays(O, D, W, w)
    clause = detector.clause
    
    O1, D1, W1, w1 = O[clause], D[clause], W[clause], w[clause]

    crystal.collect_rays(O1, D1, W1, w)
    
    return 