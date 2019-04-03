# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

@author: James
"""

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from xicsrt.util import profiler
profiler.startProfiler()

profiler.start('Total Time')
profiler.start('Import Time')

import numpy as np
import time

t1 = time.time() 
from xicsrt.xics_rt_sources import UniformAnalyticSource, DirectedSource, PointSource, ExtendedSource, FocusedExtendedSource
from xicsrt.xics_rt_detectors import Detector
from xicsrt.xics_rt_optics import SphericalCrystal
from xicsrt.xics_rt_raytrace import raytrace, raytrace_special
from xicsrt.xics_rt_tools import source_location_bragg

float64 = np.float64
profiler.stop('Import Time')


"""
Input Section
Contains information about detectors, crystals, sources, etc.
"""

input = {}
input['wavelength']          = float64(3.9490) # in angstroms    
input['crystal_location']    = np.array([float64(-8.61314000), float64(3.28703000),  float64(0.08493510)])
input['crystal_normal']      = np.array([float64(0.54276416), float64(-0.83674273),  float64(0.07258566)])
input['crystal_orientation'] = np.array([float64(-0.83598556),float64(-0.54654120), float64(-0.04920219)])
input['crystal_curvature']   = float64(1.45040) 
input['crystal_width']       = float64(.2)
input['crystal_height']      = float64(.2)    
input['crystal_spacing']     = float64(2.456760000) #in angstroms        
input['crystal_center']      = (input['crystal_location'] 
                                + (input['crystal_curvature'] 
                                   * input['crystal_normal']))
input['rocking_curve']       = float64(.000068) * 2
input['reflectivity']        = 1
input['crystal_pixel_scaling']       = int(200)

input['detector_location']   = np.array([float64(-8.67295866), float64(2.12754909), float64(0.11460174)])
input['detector_normal']     = np.array([float64(0.06377482),  float64(0.99491214), float64(-0.07799110)])
input['detector_orientation']= np.array([float64(-0.99468769), float64(0.05704335), float64(-0.08568812)])

input['pixel_size']          = float64(.000172)

input['x_size']              = 195
input['y_size']              = 1475

input['source_position'] = source_location_bragg(
    # Distance from Crystal
    float64(3.5),
    # Offset in meridional direction (typically vertical).
    0,
    # Offset in sagital direction (typically horizontal).
    0,
    input['crystal_location'],
    input['crystal_normal'], 
    input['crystal_curvature'], 
    input['crystal_spacing'],
    input['detector_location'], 
    input['wavelength'])

input['source_direction']= ((input['crystal_location'] - input['source_position'])/
                            np.linalg.norm((input['crystal_location'] - input['source_position']) ))

# Angular spread of source in degrees
input['source_spread']   = 0.5
# Number of rays to launch.
input['source_intensity']= int(1e6)

# Ion temperature in eV
input['source_temp']     = 3000

# Argon mass in AMU.
input['source_mass']     = 39.948
input['natural_linewidth'] = 0.0037 * 2.5


input['source_width']   = 0.1
input['source_height']  = 0.5
input['source_depth']   = 0.0 
input['source_orientation'] = np.cross(np.array([0, 0, 1]), input['source_direction'])
input['source_orientation'] = input['source_orientation']/np.linalg.norm(input['source_orientation'])


profiler.start('Class Setup Time')

pilatus = Detector(
    input['detector_location']
    ,input['detector_normal']
    ,input['detector_orientation']
    ,input['x_size']
    ,input['y_size']
    ,input['pixel_size'])

crystal = SphericalCrystal(
    input['crystal_location']
    ,input['crystal_normal']
    ,input['crystal_orientation']
    ,input['crystal_curvature']
    ,input['crystal_spacing']
    ,input['rocking_curve']
    ,input['reflectivity'] 
    ,input['crystal_width']
    ,input['crystal_height']
    ,input['crystal_pixel_scaling'])
    
source_1 = ExtendedSource(
    input['source_position']
    ,input['source_direction']
    ,input['source_orientation']
    ,input['source_width']
    ,input['source_height']
    ,input['source_depth']
    ,input['source_spread']
    ,input['source_intensity']
    ,input['wavelength'] 
    ,input['source_temp']
    ,input['source_mass']
    ,input['natural_linewidth']) 
    
source_2 = FocusedExtendedSource(
    input['source_position']
    ,input['source_direction']
    ,input['source_orientation']
    ,input['source_width']
    ,input['source_height']
    ,input['source_depth']
    ,input['source_spread']
    ,input['source_intensity']
    ,input['wavelength'] 
    ,input['source_temp']
    ,input['source_mass']
    ,input['natural_linewidth']
    ,input['crystal_location']) 

profiler.stop('Class Setup Time')

output = raytrace_special(source_2, pilatus, crystal)

print('Exporting detector image.')
pilatus.output_image('test_detector.tif')

print('Exporting crystal image.')
crystal.output_image('test_crystal.tif')

profiler.stop('Total Time')


profiler.stopProfiler()
profiler.report()
