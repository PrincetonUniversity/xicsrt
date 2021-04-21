#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:22:23 2021

@author: nathanbartlett
"""


import numpy as np

def build_aperture(X_local, m, aperture_info = None):
    
    m_initial = m.copy()
    for ii in range(len(aperture_info)):
        m_test = m_initial.copy()
        
        if aperture_info[ii]['logic'] == 'and':
        
            m &= aperture_selection(X_local, m_test, aperture_info,ii)
            
        elif aperture_info[ii]['logic'] == 'or':
            
            m |= aperture_selection(X_local, m_test, aperture_info,ii)
            
        elif aperture_info[ii]['logic'] == 'not':
            
            m &= ~(aperture_selection(X_local, m_test, aperture_info,ii))
        
        else:
            raise Exception('Aperture shape is not known.')
            
    m &= m_initial
            
    return m
       
def aperture_selection(X_local, m, aperture_info = None, aperture_number = 0):
    aperture_shape = aperture_info[aperture_number]['shape']
    if aperture_shape is None: aperture_shape = 'none'
    aperture_shape = aperture_shape.lower()
    if aperture_shape == 'none':
        func = no_aperture
    elif aperture_shape == 'circle':
        func = circle_aperture
    elif aperture_shape == 'square':
        func = square_aperture
    elif aperture_shape == 'elipse':
        func = elipse_aperture
    else:
        raise Exception(f'Aperture shape: "{aperture_shape}" is not known.')

    return func(X_local, m, aperture_info, aperture_number)
    

def no_aperture(X_local, m, aperture_info, aperture_number = 0):
    
    return m

def circle_aperture(X_local, m, aperture_info,aperture_number = 0):
    # circle aperture.
    relative_origin_x = aperture_info[aperture_number]['origin'][0]
    relative_origin_y = aperture_info[aperture_number]['origin'][1]
    aperture_size = aperture_info[aperture_number]['size'][0]
    m[m] &= ((X_local[m,0] -relative_origin_x)**2 + (X_local[m,1] + relative_origin_y)**2  < aperture_size**2)
    
    return m

def square_aperture(X_local, m, aperture_info,aperture_number = 0):
    # rectangular aperture
    aperture_size_x = aperture_info[aperture_number]['size'][0]
    aperture_size_y = aperture_info[aperture_number]['size'][1]
    relative_origin_x = aperture_info[aperture_number]['origin'][0]
    relative_origin_y = aperture_info[aperture_number]['origin'][1]
    m[m] &= (np.abs((X_local[m,0] - relative_origin_x)) < aperture_size_x / 2)
    m[m] &= (np.abs((X_local[m,1] - relative_origin_y)) < aperture_size_y / 2)
    
    return m

def elipse_aperture(X_local, m, aperture_info,aperture_number = 0):
    # rectangular aperture
    aperture_size_x = aperture_info[aperture_number]['size'][0]
    aperture_size_y = aperture_info[aperture_number]['size'][1]
    relative_origin_x = aperture_info[aperture_number]['origin'][0]
    relative_origin_y = aperture_info[aperture_number]['origin'][1]
    m[m] &= (((X_local[m,0] - relative_origin_x)/aperture_size_x)**2 + ((X_local[m,1] - relative_origin_y)/aperture_size_y)**2< 1)
    
    return m

























