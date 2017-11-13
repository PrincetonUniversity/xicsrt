# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:13:39 2017

@author: James
"""
import time

def raytrace(source, detector, *optics):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), and wavelength (W).
    """
    t2 = time.time()
    O, D, W = source.generate_rays()
    print("Took " + str(round(time.time() - t2, 4)) + ' sec: Ray Generation TIme')
    print('Rays Generated: ' + str(len(D)))
        
    for optic in optics:
        t2 = time.time()
        O, D, W = optic.light(O, D, W)
        print("Took " + str(round(time.time() - t2, 4)) + ' sec: Optic Time')
        #optic.collect_rays(O, D, W)
        print('Rays from Crystal: ' + str(len(D)))
    t2 = time.time()
    detector.collect_rays(O, D, W)
    print("Took " + str(round(time.time() - t2, 4)) + ' sec: Collection Time')
    return      
        
        
def raytrace_special(duration, source, detector, crystal):
    """ Rays are generated from source and then passed through the optics in
    the order listed. Finally, they are collected by the detector. 
    Rays consists of origin (O), direction (D), and wavelength (W).
    
    This function also lets the crystal collect the rays to allow for the
    determination of which rays satisfy the bragg condition.
    """
    
    O, D, W = source.generate_rays(duration)
    print('Rays Generated: ' + str(len(D)))
    
    O, D, W = crystal.light(O, D, W)
    print('Rays from Crystal: ' + str(len(D)))
    
    detector.collect_rays(O, D, W)
    clause = detector.clause
    
    O1, D1, W1 = O[clause], D[clause], W[clause]
    #O1, D1, W1 = O, D, W
    crystal.collect_rays(O1, D1, W1)
    
    return 