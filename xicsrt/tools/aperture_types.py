#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:22:23 2021

@author: nathanbartlett
"""


import numpy as np

def build_aperture(X_local, m, aperture_info = None, aperture_logic = 'physical'):
    
    m = aperture_logic_selection(X_local, m, aperture_info, aperture_logic)
    
    return m

def aperture_logic_selection(X_local, m, aperture_info = None, aperture_logic = 'physical'):
    
    if aperture_logic == 'physical':
        m = aperture_logic_physical(X_local, m, aperture_info, aperture_number = 0)
    elif aperture_logic == 'and':
        m = aperture_logic_and(X_local, m, aperture_info, aperture_number = 0)
    elif aperture_logic == 'not':
        m = aperture_logic_not(X_local, m, aperture_info, aperture_number = 0)
        
    return m
        
def aperture_logic_physical(X_local, m, aperture_info = None, aperture_number = 0):
    
    m_initial = m.copy()
    for ii in range(len(aperture_info)):
        m_test = m_initial.copy()
            
        if ii == 0:
        
            m &= aperture_selection(X_local, m_test, aperture_info,ii)
            
        else:
            
            m |= aperture_selection(X_local, m_test, aperture_info,ii)
    
    return m

def aperture_logic_and(X_local, m, aperture_info = None, aperture_number = 0):
    
    m_initial = m.copy()
    for ii in range(len(aperture_info)):
        m_test = m_initial.copy()
        m &= aperture_selection(X_local, m_test, aperture_info,ii)
            
    return m

def aperture_logic_not(X_local, m, aperture_info = None, aperture_number = 0):
    m_initial = m.copy()
    m = aperture_logic_physical(X_local, m, aperture_info, aperture_number = 0)
    m[m_initial] = (~ m[m_initial])
            
    return m
       
def aperture_selection(X_local, m, aperture_info = None, aperture_number = 0):
    aperture_type = aperture_info[aperture_number]['type']
    if aperture_type is None: aperture_type = 'none'
    aperture_type = aperture_type.lower()
    if aperture_type == 'none':
        func = no_aperture
    elif aperture_type == 'circle':
        func = circle_aperture
    elif aperture_type == 'square':
        func = square_aperture
    else:
        raise Exception(f'Aperture type: "{aperture_type}" is not known.')

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


