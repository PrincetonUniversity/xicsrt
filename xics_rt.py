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


    
CRYSTAL_LOCATION    = np.array([-8.61314000,     3.28703000,     0.08493510])
CRYSTAL_NORMAL      = np.array([0.54276416,     -0.83674273,     0.07258566])
CRYSTAL_ORIENTATION = np.array([-0.83598556,    -0.54654120,    -0.04920219])
CRYSTAL_CURVATURE   = 1.45040 
CRYSTAL_WIDTH       = .040
CRYSTAL_HEIGHT      = .100    
CRYSTAL_SPACING     = 2.45676 #in angstroms        
CRYSTAL_CENTER      = CRYSTAL_LOCATION + CRYSTAL_CURVATURE * CRYSTAL_NORMAL

DETECTOR_LOCATION   = np.array([-8.67295866,     2.12754909,     0.11460174])
DETECTOR_NORMAL     = np.array([0.06377482,      0.99491214,    -0.07799110])
DETECTOR_ORIENTATION = np.array([-0.99468769,     0.05704335,    -0.08568812])

PIXEL_SIZE          = .000172

X_SIZE              = 195
Y_SIZE              = 1475


t1 = time.time()



pilatus = Detector(DETECTOR_LOCATION, DETECTOR_NORMAL, DETECTOR_ORIENTATION,
                   195, 1475, .000172)

crystal = SphericalCrystal(CRYSTAL_LOCATION, CRYSTAL_NORMAL, 
                           CRYSTAL_ORIENTATION, CRYSTAL_CURVATURE, 
                           CRYSTAL_SPACING, .000068, 1, 
                           CRYSTAL_WIDTH, CRYSTAL_HEIGHT)
"""
Given the detector and crystal locations, find a source position that satisfies
the Bragg condition.
Source output is a million rays. 10 degree spread, 3.95 angstrom wavelength,
temperature is 11 eV, mass number is 112 (Cadmium).
"""

source_position = source_location_bragg(.01, 0, CRYSTAL_LOCATION, CRYSTAL_NORMAL, 
                                  CRYSTAL_CURVATURE, CRYSTAL_SPACING,
                                  DETECTOR_LOCATION, 4)

source_direction = ((CRYSTAL_LOCATION - source_position)/
                    np.linalg.norm((CRYSTAL_LOCATION - source_position) ))
    
source = DirectedSource(source_position, source_direction, 10, 100000, 3.95,
                            11, 112) 


"""
Start the raytracing code. 
Output detector image to 'new_xics_image.tif'
"""

#raytrace(1, source, pilatus, crystal)
#pilatus.output_image('ded2.tif')


raytrace_special(1, source, pilatus, crystal)
pilatus.output_image('new_detector1.tif')
crystal.output_image('new_crystal1.tif')



print("Took " + str(round(time.time() - t1, 4)) + ' sec: Total Time' )
