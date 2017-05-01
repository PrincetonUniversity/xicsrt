# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

@author: James
"""

import numpy as np
import time
from xics_rt_classes_v1e1 import Detector, raytrace, SphericalCrystal
from xics_rt_classes_v1e1 import DirectedSource, raytrace_special
from xics_rt_tools import source_location, source_location_bragg

input = {}
    
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


input['detector_location']   = np.array([-8.67295866, 2.12754909,  0.11460174])
input['detector_normal']     = np.array([0.06377482,  0.99491214, -0.07799110])
input['detector_orientation'] = np.array([-0.99468769,0.05704335, -0.08568812])

input['pixel_size']          = .000172

input['x_size']              = 195
input['y_size']              = 1475


t1 = time.time()



pilatus = Detector(input['detector_location'], input['detector_normal'], 
                   input['detector_orientation'],
                   input['x_size'], input['y_size'], input['pixel_size'])

crystal = SphericalCrystal(input['crystal_location'], input['crystal_normal'], 
                           input['crystal_orientation'],
                           input['crystal_curvature'], 
                           input['crystal_spacing'], input['rocking_curve'], 
                           input['reflectivity'], 
                           input['crystal_width'], input['crystal_height'])
"""
Given the detector and crystal locations, find a source position that satisfies
the Bragg condition.
Source output is a million rays. 10 degree spread, 3.95 angstrom wavelength,
temperature is 11 eV, mass number is 112 (Cadmium).
"""


input['wavelength']      = 3.95 # in angstroms
input['source_position'] = source_location_bragg(.01, 0, 
                                                 input['crystal_location'],
                                                 input['crystal_normal'], 
                                                 input['crystal_curvature'], 
                                                 input['crystal_spacing'],
                                                 input['detector_location'], 
                                                 input['wavelength'])

input['source_direction']= ((input['crystal_location'] - input['source_position'])/
                    np.linalg.norm((input['crystal_location'] - input['source_position']) ))

input['source_spread']   = 10  #degrees
input['source_intensity']= 1000000
input['source_temp']     = 11  # in eV
input['source_mass']     = 112 # in atomic units (Cadmium = 112)

  
source = DirectedSource(input['source_position'], input['source_direction'], input['source_spread'],
                        input['source_intensity'], input['wavelength'], input['source_temp'],
                        input['source_mass']) 


"""
Start the raytracing code. 
Output detector image to 'new_xics_image.tif'
"""

raytrace(1, source, pilatus, crystal)
pilatus.output_image('ded2.tif')


#raytrace_special(1, source, pilatus, crystal)
#pilatus.output_image('new_detector1.tif')
#crystal.output_image('new_crystal1.tif')



print("Took " + str(round(time.time() - t1, 4)) + ' sec: Total Time' )
