# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:29:37 2017

Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
This script loads up all of the initial parameters and object classes needed to
execute the xicsrt raytracing pipeline
"""

# This is only needed since I have not actually installed xicsrt
import sys
sys.path.append('/Users/Eugene/PPPL_python_project1')
sys.path.append('/Users/Eugene/PPPL_python_project1/xics_rt_code')             #YVY: Changed filepath

# Start Logging
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Start Profiling
from xicsrt.util import profiler
profiler.startProfiler()

profiler.start('Total Time')
profiler.start('Import Time')

# Begin normal imports
import numpy as np
from collections import OrderedDict

import os
import argparse

from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_detectors import Detector
from xicsrt.xics_rt_optics import SphericalCrystal
from xicsrt.xics_rt_raytrace import raytrace, raytrace_special
from xicsrt.xics_rt_tools import source_location_bragg

# MirPyIDL imports
import mirutil.hdf5

profiler.stop('Import Time')


"""
Input Section
Contains information about detectors, crystals, sources, etc.

Crystal and detector parameters are taken from the W7-X Op1.2 calibration.
This can be found in the file w7x_ar16.cerdata for shot 180707017.
"""
input = OrderedDict()

input['ideal_geometry'] = True

input['system'] = 'w7x_ar16'
input['shot'] = 180707017


# Number of rays to launch.
#
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
input['source_intensity']= int(1e7)
input['number_of_runs'] = 1

# Argon mass in AMU.
input['source_mass']     = 39.948

# in angstroms
# Line location (angstroms) and natural linewith (1/s)
#Ar16+ w line.
input['wavelength']          = 3.9492
input['natural_linewidth']   = 1.129e+14
#Ar16+ x line.
#input['wavelength']          = 3.9660
#input['natural_linewidth']   = 1.673E+12
#Ar16+ y line.
#input['wavelength']          = 3.9695
#input['natural_linewidth']   = 3.212E+08
#Ar16+ z line.
#input['wavelength']          = 3.9944
#input['natural_linewidth']   = 4.439E+06
#input['natural_linewidth'] = 0.0

config_dict = mirutil.hdf5.hdf5ToDict(
        '/Users/Eugene/PPPL_python_project1/w7x_ar16_180707017_geometry.hdf5')

input['crystal_location']    = config_dict['CRYSTAL_LOCATION']
input['crystal_normal']      = config_dict['CRYSTAL_NORMAL']
input['crystal_orientation'] = config_dict['CRYSTAL_ORIENTATION']
input['crystal_curvature']   = config_dict['CRYSTAL_CURVATURE']
input['crystal_spacing']     = config_dict['CRYSTAL_SPACING']
input['crystal_width']       = 0.04
input['crystal_height']      = 0.1  
input['crystal_center']      = (input['crystal_location'] 
                                + (input['crystal_curvature'] * input['crystal_normal']))

# Rocking curve FWHM in radians.
# This is taken from x0h for quartz 1,1,-2,0
# Darwin Curve, sigma: 48.070 urad
# Darwin Curve, pi:    14.043 urad
input['rocking_curve_fwhm']    = 48.070e-6
input['reflectivity']          = 1
input['crystal_pixel_scaling'] = int(200)

input['detector_location']    = config_dict['DETECTOR_LOCATION']
input['detector_normal']      = config_dict['DETECTOR_NORMAL']
input['detector_orientation'] = config_dict['DETECTOR_ORIENTATION']

input['pixel_size']          = 0.000172
input['x_size']              = int(config_dict['X_SIZE'])
input['y_size']              = int(config_dict['Y_SIZE'])

#YY NOTE: Right now the source location is defined relative to the spherical
#crystal. The project parameters have changed, and now the rays start at the 
#souce, bounce off of a graphite mirror (HOPG),  then the crystal, and finally
#hit the detector. We should change this block of code to position the source
#relative to the graphite, and then add code to position the graphite relative
#to the crystal.
#Remove this comment once the graphite object is implemented
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
input['source_temp']     = 1000

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

#
#graphite = ReflectorGraphite(
#    input['graphite_location']
#   ,input['graphite_normal']
#   ,input['graphite_orientation']
#   ,input['graphite_reflectivity']
#   ,inpur['graphite_mosaic_spread']
#   ,input['graphite_crystal_spacing']
#   ,input['graphite_rocking_curve_fwhm']
#   ,input['graphite_width']
#   ,input['graphite_height']
#   ,input['graphite_pixel_scaling'])
#

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

    #output = raytrace_special(source, pilatus, graphite, crystal)
    output = raytrace(
        source
        ,pilatus
#       ,graphite        
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
