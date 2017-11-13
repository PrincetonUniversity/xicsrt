# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

@author: James
"""

import numpy as np
import time

t1 = time.time() 
from xics_rt_sources import UniformAnalyticSource, DirectedSource, PointSource
from xics_rt_detectors import Detector
from xics_rt_optics import SphericalCrystal
from xics_rt_raytrace import raytrace, raytrace_special
from xics_rt_tools import source_location_bragg


print("Took " + str(round(time.time() - t1, 4)) + ' sec: Import Time' )
input = {}
input['wavelength']          = 3.95 # in angstroms    
input['crystal_location']    = np.array([-8.61314000, 3.28703000,  0.08493510])
input['crystal_normal']      = np.array([0.54276416, -0.83674273,  0.07258566])
input['crystal_orientation'] = np.array([-0.83598556,-0.54654120, -0.04920219])
input['crystal_curvature']   = 1.45040 
input['crystal_width']       = .040
input['crystal_height']      = .100    
input['crystal_spacing']     = 2.45676 #in angstroms        
input['crystal_center']      = (input['crystal_location'] 
                                + (input['crystal_curvature'] 
                                   * input['crystal_normal']))
input['rocking_curve']       = .000068
input['reflectivity']        = 1
input['pixel_scaling']       = int(1)

input['detector_location']   = np.array([-8.67295866, 2.12754909,  0.11460174])
input['detector_normal']     = np.array([0.06377482,  0.99491214, -0.07799110])
input['detector_orientation']= np.array([-0.99468769,0.05704335, -0.08568812])

input['pixel_size']          = .000172

input['x_size']              = 195
input['y_size']              = 1475

print("Took " + str(round(time.time() - t1, 4)) + ' sec: Input Time' )






pilatus = Detector(input['detector_location'], input['detector_normal'], 
                   input['detector_orientation'],
                   input['x_size'], input['y_size'], input['pixel_size'])

crystal = SphericalCrystal(input['crystal_location'], input['crystal_normal'], 
                           input['crystal_orientation'],
                           input['crystal_curvature'], 
                           input['crystal_spacing'], input['rocking_curve'], 
                           input['reflectivity'], 
                           input['crystal_width'], input['crystal_height'],
                           input['pixel_scaling'])


"""
Given the detector and crystal locations, find a source position that satisfies
the Bragg condition.
Source output is 5 million rays. 10 degree spread, 3.95 angstrom wavelength,
temperature is 1100 eV, mass number is 112 (Cadmium).

Source is UniformAnalyticSource which requires a crystal argument.
DirectedSource and PointSource do not require a crystal.
"""


    
input['source_position'] = source_location_bragg(0.09, 0, 0.000006,
                                                 input['crystal_location'],
                                                 input['crystal_normal'], 
                                                 input['crystal_curvature'], 
                                                 input['crystal_spacing'],
                                                 input['detector_location'], 
                                                 input['wavelength'])

input['source_direction']= ((input['crystal_location'] - input['source_position'])/
                    np.linalg.norm((input['crystal_location'] - input['source_position']) ))

input['source_spread']   = 20  #degrees
input['source_intensity']= 1000
input['source_temp']     = 1100  # in eV
input['source_mass']     = 112 # in atomic units (Cadmium = 112)



source = DirectedSource(input['source_position'], 
                               input['source_direction'], 
                               input['source_spread'],
                               input['source_intensity'], 
                               input['wavelength'], 
                               input['source_temp'],
                               input['source_mass']) 

print("Took " + str(round(time.time() - t1, 4)) + ' sec: Class Setup Time' )

"""
Start the raytracing code. 
Output detector image to 'test.tif'
"""


raytrace(source, pilatus, crystal)

pilatus.output_image('plus_offset1.tif')
  

print("Took " + str(round(time.time() - t1, 4)) + ' sec: Total Time' )
