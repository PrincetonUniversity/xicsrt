# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

authors
-------
  - James Kring
  - Novimir Pablant
"""

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from xicsrt.util import profiler
profiler.startProfiler()

profiler.start('Total Time')
profiler.start('Import Time')

import numpy as np
from collections import OrderedDict

from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_detectors import Detector
from xicsrt.xics_rt_optics import SphericalCrystal
from xicsrt.xics_rt_raytrace import raytrace, raytrace_special
from xicsrt.xics_rt_tools import source_location_bragg

profiler.stop('Import Time')


"""
Input Section
Contains information about detectors, crystals, sources, etc.

Crystal and detector parameters are taken from the W7-X Op1.2 calibration.
This can be found in the file w7x_ar16.cerdata for shot 180707017.
"""
input = OrderedDict()

# Number of rays to launch.
#
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
input['source_intensity']= int(1e7)
input['number_of_runs'] = 100


# Evetually I should just read all of these numbers directly
# from the CERDATA file. For now I'll manually add these to make
# code shareing with James easier. 
input['wavelength']          = 3.9492 # in angstroms    
input['crystal_location']    = np.array([-8.6004817282842634e+00,   3.2948364023501213e+00,    7.9607746965125376e-02])
input['crystal_normal']      = np.array([ 5.3371892832829204e-01,  -8.4240900512544947e-01,    7.4102453587246514e-02])
input['crystal_orientation'] = np.array([-8.4180325822227686e-01,  -5.3760131424833157e-01,   -4.8498467654403514e-02])
input['crystal_curvature']   = 1.45040
input['crystal_width']       = 0.04
input['crystal_height']      = 0.1  
input['crystal_spacing']     = 2.456760000 #in angstroms        
input['crystal_center']      = (input['crystal_location'] 
                                + (input['crystal_curvature'] * input['crystal_normal']))
input['rocking_curve']       = 0.000068 * 2
input['reflectivity']        = 1
input['crystal_pixel_scaling']       = int(200)

input['detector_location']   = np.array([-8.6696410292958781e+00,  2.1434156177104566e+00,  1.1016580447728094e-01])
input['detector_normal']     = np.array([ 6.3380413140006073e-02,  9.9764908228981930e-01, -2.6062076595761697e-02])
input['detector_orientation']= np.array([-9.9463191023494646e-01,  6.1005316208816135e-02, -8.3580587080038543e-02])
input['pixel_size']          = 0.000172
input['x_size']              = 195
input['y_size']              = 1475

input['source_position'] = source_location_bragg(
    # Distance from Crystal
    3.5,
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

input['source_direction'] = (input['crystal_location'] - input['source_position'])
input['source_direction'] /=  np.linalg.norm(input['source_direction'])

# This direction is rather abitrary and is not (in general)
# in the meridional direction.
input['source_orientation'] = np.cross(np.array([0, 0, 1]), input['source_direction'])
input['source_orientation'] /= np.linalg.norm(input['source_orientation'])


# Angular spread of source in degrees.
# This needs to be matched to the source distance and crystal size.
# At the moment this is too small to fully illuminate the crystal.
input['source_spread']   = 1.0
# Ion temperature in eV
input['source_temp']     = 1800
# Argon mass in AMU.
input['source_mass']     = 39.948
# Naturaly linewith for the Ar16+ w line. Units: 1/s.
input['natural_linewidth'] = 1.129e+14


# These values are arbitrary for now.
input['source_width']   = 0.15
input['source_height']  = 0.75
input['source_depth']   = 1.0



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
    
source = FocusedExtendedSource(
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

#output = raytrace_special(source, pilatus, crystal)
output = raytrace(
    source
    ,pilatus
    ,crystal
    ,number_of_runs=input['number_of_runs']
    ,collect_optics=True)


print('Exporting detector image.')
pilatus.output_image('test_detector.tif')

print('Exporting crystal image.')
crystal.output_image('test_crystal.tif')

profiler.stop('Total Time')


profiler.stopProfiler()
print('')
profiler.report()
