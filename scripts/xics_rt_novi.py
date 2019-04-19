# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring
"""
# This is only needed since I have not actually installed xicsrt.
import sys
sys.path.append('/u/npablant/code/mirproject/xicsrt')


import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from xicsrt.util import profiler
profiler.startProfiler()

profiler.start('Total Time')
profiler.start('Import Time')

import numpy as np
from collections import OrderedDict

import os
import argparse

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

input['ideal_geometry'] = True

# Number of rays to launch.
#
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
input['source_intensity']= int(1e7)
input['number_of_runs'] = 1000


# Argon mass in AMU.
input['source_mass']     = 39.948

# in angstroms
# Line location (angstroms) and naturaly linewith (1/s)
#Ar16+ w line.
input['wavelength']          = 3.9492
input['natural_linewidth'] = 1.129e+14
#Ar16+ z line.
#input['wavelength']          = 3.9944
#input['natural_linewidth'] = 4.439E+06
#input['natural_linewidth'] = 0.0

# Evetually I should just read all of these numbers directly
# from the CERDATA file. For now I'll manually add these to make
# code shareing with James easier. 
input['crystal_location']    = np.array([-8.6004817282842634e+00,   3.2948364023501213e+00,    7.9607746965125376e-02])
input['crystal_normal']      = np.array([ 5.3371892832829204e-01,  -8.4240900512544947e-01,    7.4102453587246514e-02])
input['crystal_orientation'] = np.array([-8.4180325822227686e-01,  -5.3760131424833157e-01,   -4.8498467654403514e-02])
input['crystal_curvature']   = 1.45040
input['crystal_width']       = 0.04
input['crystal_height']      = 0.1  
input['crystal_spacing']     = 2.456760000 #in angstroms        
input['crystal_center']      = (input['crystal_location'] 
                                + (input['crystal_curvature'] * input['crystal_normal']))
# Rocking curve FWHM in radians.
# This is taken from x0h for quarts 1,1,-2,0
# Darwin Curve, sigma: 48.070 urad
# Darwin Curve, pi:    14.043 urad
input['rocking_curve_fwhm']       = 48.070e-6
input['reflectivity']        = 1
input['crystal_pixel_scaling']       = int(200)

# These are the values from the 2018-07-18 calibration.
#input['detector_location']   = np.array([-8.6696410292958781e+00,  2.1434156177104566e+00,  1.1016580447728094e-01])
#input['detector_normal']     = np.array([ 6.3380413140006073e-02,  9.9764908228981930e-01, -2.6062076595761697e-02])
#input['detector_orientation']= np.array([-9.9463191023494646e-01,  6.1005316208816135e-02, -8.3580587080038543e-02])
if input['ideal_geometry']:
    # Put the detector in the ideal location for the w-line.
    
    # This should put the center of the detector on the Roland sphere for 3.9492.
    #input['detector_location']   = np.array([-8.6842790093703197,   2.1326564775438337,    0.11540044021092999])
    #input['detector_normal']     = np.array([ 0.071882926070111167, 0.99694038436909072,  -0.030703663520969083])
    #input['detector_orientation']= np.array([-0.99414358366844624,  0.069122897884802220, -0.083069609598615271])

    # Here the detector has been shifted detector to approximately match the w7-x position.
    input['detector_location']   = np.array([-8.6976229910780312,   2.1334749248660914,   0.11427912305466420])
    input['detector_normal']     = np.array([ 0.071882926070111167, 0.99694038436909072,  -0.030703663520969083])
    input['detector_orientation']= np.array([-0.99414358366844624,  0.069122897884802220, -0.083069609598615271])

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
input['source_spread']   = 1.0
# Ion temperature in eV
input['source_temp']     = 1800


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
    ,input['rocking_curve_fwhm']
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

import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suffix'
        ,help='A suffix to add to the end of the image name.'
        ,type=str)
    parser.add_argument(
        '--path'
        ,default=''
        ,help='The path to store the results.'
        ,type=str)    
    args = parser.parse_args()

    
    #output = raytrace_special(source, pilatus, crystal)
    output = raytrace(
        source
        ,pilatus
        ,crystal
        ,number_of_runs=input['number_of_runs']
        ,collect_optics=True)


    

    # Create the output path if needed.
    if args.path:
        if not os.path.exists(args.path):
           os.mkdir(args.path)
                    
    filename = 'xicsrt_detector'
    if args.suffix:
        filename += '_'+args.suffix
    filename += '.tif'
    filepath = os.path.join(args.path, filename)
    print('Exporting detector image: {}'.format(filepath))
    pilatus.output_image(filepath, rotate=False)

    filename = 'xicsrt_crystal'
    if args.suffix:
        filename += '_'+args.suffix
    filename += '.tif'
    filepath = os.path.join(args.path, filename)
    print('Exporting crystal image:  {}'.format(filepath))
    crystal.output_image(filepath, rotate=False)

    profiler.stop('Total Time')


    profiler.stopProfiler()
    print('')
    profiler.report()
