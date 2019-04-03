# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

@author: James
"""

import numpy as np
import time

t1 = time.time() 
from xics_rt_sources import UniformAnalyticSource, DirectedSource, PointSource, ExtendedSource
from xics_rt_detectors import Detector
from xics_rt_optics import SphericalCrystal
from xics_rt_raytrace import raytrace, raytrace_special
from xics_rt_tools import source_location_bragg

float64 = np.float64
print("Took " + str(round(time.time() - t1, 4)) + ' sec: Import Time' )

"""
Input Section
Contains information about detectors, crystals, sources, etc.
"""

input = {}
input['wavelength']          = float64(3.9495) # in angstroms    
input['crystal_location']    = np.array([float64(-8.61314000), float64(3.28703000),  float64(0.08493510)])
input['crystal_normal']      = np.array([float64(0.54276416), float64(-0.83674273),  float64(0.07258566)])
input['crystal_orientation'] = np.array([float64(-0.83598556),float64(-0.54654120), float64(-0.04920219)])
input['crystal_curvature']   = float64(1.45040) 
input['crystal_width']       = float64(.040)
input['crystal_height']      = float64(.100)    
input['crystal_spacing']     = float64(2.456760000) #in angstroms        
input['crystal_center']      = (input['crystal_location'] 
                                + (input['crystal_curvature'] 
                                   * input['crystal_normal']))
input['rocking_curve']       = float64(.000068) * 2
#input['rocking_curve']       = float64(.000010)
input['reflectivity']        = 1
input['crystal_pixel_scaling']       = int(200)

input['detector_location']   = np.array([float64(-8.67295866), float64(2.12754909), float64(0.11460174)])
input['detector_normal']     = np.array([float64(0.06377482),  float64(0.99491214), float64(-0.07799110)])
input['detector_orientation']= np.array([float64(-0.99468769), float64(0.05704335), float64(-0.08568812)])

input['pixel_size']          = float64(.000172)

input['x_size']              = 195
input['y_size']              = 1475

input['source_position'] = source_location_bragg(float64(.04), 0, 0 ,
                                                 input['crystal_location'],
                                                 input['crystal_normal'], 
                                                 input['crystal_curvature'], 
                                                 input['crystal_spacing'],
                                                 input['detector_location'], 
                                                 input['wavelength'])

input['source_direction']= ((input['crystal_location'] - input['source_position'])/
                    np.linalg.norm((input['crystal_location'] - input['source_position']) ))

input['source_spread']   = 1 #degrees
input['source_intensity']= 100000

input['source_temp']     = 5000# in eV
input['source_mass']     = 39.948 # in atomic units (Cadmium = 112, Argon = 39.948)
#input['natural_linewidth'] = 0.0037 * 2.5
input['natural_linewidth'] = 0.0

input['source_width']   = .1
input['source_height']  = .1
input['source_depth']   = .1 
input['source_orientation'] = np.cross(np.array([0, 0, 1]), input['source_direction'])
input['source_orientation'] = input['source_orientation']/np.linalg.norm(input['source_orientation'])
 


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
                           input['crystal_pixel_scaling'])


"""
Given the detector and crystal locations, find a source position that satisfies
the Bragg condition.
Source output is 5 million rays. 1 degree spread, 3.95 angstrom wavelength,
temperature is 5000 eV, mass number is 112 (Cadmium).

Source is UniformAnalyticSource which requires a crystal argument.
DirectedSource, ExtendedSource, and PointSource do not require a crystal.
"""
    

"""
Examples for all different source types are given. 
"""
source = DirectedSource(input['source_position'], 
                               input['source_direction'], 
                               input['source_spread'],
                               input['source_intensity'], 
                               input['wavelength'], 
                               input['source_temp'],
                               input['source_mass'],
                               input['natural_linewidth'])

"""
source = ExtendedSource(input['source_position'], 
                               input['source_direction'], 
                               input['source_orientation'],
                               input['source_width'],
                               input['source_height'],
                               input['source_depth'],
                               input['source_spread'],
                               input['source_intensity'], 
                               input['wavelength'], 
                               input['source_temp'],
                               input['source_mass'],
                               input['natural_linewidth']) 



sourcep = UniformAnalyticSource(input['source_position'], 
                               input['source_direction'], 
                               input['source_spread'],
                               input['source_intensity'], 
                               input['wavelength'], 
                               input['source_temp'],
                               input['source_mass'],
                               input['natural_linewidth'],
                               crystal) 

source = PointSource(input['source_position'], 
                               input['source_intensity'], 
                               input['wavelength'], 
                               input['source_temp'],
                               input['source_mass'],
                               input['natural_linewidth']) 
"""


print("Took " + str(round(time.time() - t1, 4)) + ' sec: Class Setup Time' )

    
"""
Raytracing can begin.
In principle, more than one optice can be used in sequence.
"""
raytrace(source, pilatus, crystal)
    
"""
Image is output ast test.tif
"""
pilatus.output_image(str('test' + '.tif'))


print("Took " + str(round(time.time() - t1, 4)) + ' sec: Total Time' )


"""
The crystal can also serve a detector for troubleshooting purposes such as 
making sure the source is directed as it should be.
To do so requires raytrace_special. The resolution of the crystal produced 
image depends on the crystal_pixel scaling input.


raytrace_special(source_test, pilatus, crystal)

pilatus.output_image(str('rocking_test'+'.tif'))
crystal.output_image(str('extended_test'+ str(i) + 'crystal.tif'))

"""
