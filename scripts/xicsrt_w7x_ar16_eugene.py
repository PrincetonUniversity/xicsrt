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
from xicsrt.xics_rt_optics import SphericalCrystal, MosaicGraphite
from xicsrt.xics_rt_raytrace import raytrace
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
general_input   = OrderedDict()
source_input    = OrderedDict()
crystal_input   = OrderedDict()
graphite_input  = OrderedDict()
detector_input  = OrderedDict()

general_input['ideal_geometry'] = True
general_input['random_seed'] = 123456
general_input['system'] = 'w7x_ar16'
general_input['shot'] = 180707017

# Number of rays to launch.
# A source intensity greater than 1e7 is not recommended due to excessive
# memory usage.
source_input['source_intensity']= int(1e7)
general_input['number_of_runs'] = 10

# Argon mass in AMU.
source_input['source_mass']     = 39.948

# in angstroms
# Line location (angstroms) and natural linewith (1/s)
#Ar16+ w line.
source_input['source_wavelength']   = 3.9492
source_input['source_linewidth']    = 1.129e+14
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

#Load spherical crystal properties from hdf5 file
config_dict = mirutil.hdf5.hdf5ToDict(
        '/Users/Eugene/PPPL_python_project1/w7x_ar16_180707017_geometry.hdf5')

# Rocking curve FWHM in radians.
# This is taken from x0h for quartz 1,1,-2,0
# Darwin Curve, sigma: 48.070 urad
# Darwin Curve, pi:    14.043 urad

crystal_input['crystal_position']       = config_dict['CRYSTAL_LOCATION']
crystal_input['crystal_normal']         = config_dict['CRYSTAL_NORMAL']
crystal_input['crystal_orientation']    = config_dict['CRYSTAL_ORIENTATION']
crystal_input['crystal_curvature']      = config_dict['CRYSTAL_CURVATURE']
crystal_input['crystal_spacing']        = config_dict['CRYSTAL_SPACING']
crystal_input['crystal_width']          = 0.04
crystal_input['crystal_height']         = 0.10 
crystal_input['crystal_center']         = (crystal_input['crystal_position'] 
                                        + (crystal_input['crystal_curvature'] 
                                        *  crystal_input['crystal_normal']))
crystal_input['crystal_reflectivity']   = 1
crystal_input['crystal_rocking_curve']  = 48.070e-6
crystal_input['crystal_pixel_scaling']  = int(200)

#Load mosaic graphite properties
graphite_input['graphite_position']          = config_dict['CRYSTAL_LOCATION']
graphite_input['graphite_normal']            = config_dict['CRYSTAL_NORMAL']
graphite_input['graphite_orientation']       = config_dict['CRYSTAL_ORIENTATION']
graphite_input['graphite_width']             = 0.04
graphite_input['graphite_height']            = 0.10
graphite_input['graphite_reflectivity']      = 1
graphite_input['graphite_mosaic_spread']     = 0.1
graphite_input['graphite_crystal_spacing']   = 3.35
graphite_input['graphite_rocking_curve']     = 48.070e-6
graphite_input['graphite_pixel_scaling']     = int(200)

#Load detector properties
detector_input['detector_position']     = config_dict['DETECTOR_LOCATION']
detector_input['detector_normal']       = config_dict['DETECTOR_NORMAL']
detector_input['detector_orientation']  = config_dict['DETECTOR_ORIENTATION']

detector_input['pixel_size']            = 0.000172
detector_input['horizontal_pixels']     = int(config_dict['X_SIZE'])
detector_input['vertical_pixels']       = int(config_dict['Y_SIZE'])

"""
 YY NOTE: Right now the source location is defined relative to the spherical
 crystal. The project parameters have changed, and now the rays start at the 
 source, bounce off of a graphite mirror (HOPG),  then the crystal, and finally
 hit the detector. We should change this block of code to position the source
 relative to the graphite, and then add code to position the graphite relative
 to the crystal.
"""

source_input['source_position'] = source_location_bragg(
    # Distance from Crystal
    3.5,
    # Offset in meridional direction (typically vertical).
    0,
    # Offset in sagital direction (typically horizontal).
    0,
    crystal_input['crystal_position'],
    crystal_input['crystal_normal'], 
    crystal_input['crystal_curvature'], 
    crystal_input['crystal_spacing'],
    detector_input['detector_position'], 
    source_input['source_wavelength'])
source_input['source_target'] = crystal_input['crystal_position']

source_input['source_direction'] = (crystal_input['crystal_position'] - source_input['source_position'])
source_input['source_direction'] /=  np.linalg.norm(source_input['source_direction'])

# This direction is rather abitrary and is not (in general)
# in the meridional direction.
source_input['source_orientation'] = np.cross(np.array([0, 0, 1]), source_input['source_direction'])
source_input['source_orientation'] /= np.linalg.norm(source_input['source_orientation'])

# Angular spread of source in degrees.
# This needs to be matched to the source distance and crystal size.
source_input['source_spread']   = 1.0
# Ion temperature in eV
source_input['source_temp']     = 1000

# These values are arbitrary for now.
source_input['source_width']   = 0.15
source_input['source_height']  = 0.75
source_input['source_depth']   = 1.0

#pipe all of the configuration settings into their respective objects
profiler.start('Class Setup Time')

pilatus     = Detector(detector_input, general_input)
crystal     = SphericalCrystal(crystal_input, general_input)
graphite    = MosaicGraphite(graphite_input, general_input)
source      = FocusedExtendedSource(source_input, general_input)

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
        ,graphite        
#       ,crystal
        ,number_of_runs = general_input['number_of_runs']
        ,collect_optics = True)

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

    filename = 'xicsrt_graphite'
    if args.suffix:
        filename += '_'+args.suffix
    filename += '.tif'
    filepath = os.path.join(args.path, filename)
    print('Exporting graphite image: {}'.format(filepath))
    graphite.output_image(filepath, rotate=False)

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
