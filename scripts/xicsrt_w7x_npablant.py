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
This script creates a default configuration for a single XICS run of the ITER XRCS-Core.

"""

import numpy as np
from collections import OrderedDict

from xicsrt.util import profiler

def get_config():

    config = OrderedDict()
    config['general_input']   = OrderedDict()
    config['scenario_input']  = OrderedDict()
    config['plasma_input']    = OrderedDict()
    config['source_input']    = OrderedDict()
    config['graphite_input']  = OrderedDict()
    config['crystal_input']   = OrderedDict()
    config['detector_input']  = OrderedDict()

    # -----------------------------------------------------------------------------
    # General raytracer properties

    # set ideal_geometry to False to enable offsets, tilts, and thermal expand
    # set backwards_raytrace to True to swap the detector and source
    # set do_visualizations to toggle the visualizations on or off
    # set do_savefiles to toggle whether the program saves .tif files
    # set do_image_analysis to toggle whether the visualizer performs .tif analysis
    # set do_bragg_checks to False to make the optics into perfect X-Ray mirrors
    # set do_miss_checks to False to prevent optics from masking rays that miss
    # change the random seed to alter the random numbers generated
    # possible scenarios include 'MODEL', 'PLASMA', 'BEAM', 'CRYSTAL', 'GRAPHITE', 'SOURCE'
    # possible rocking curve types include 'STEP', 'GAUSS', and 'FILE'

    config['general_input']['number_of_rays']     = int(1e6)
    config['general_input']['number_of_runs']     = 1

    config['general_input']['ouput_path']         = '/u/npablant/code/mirproject/xicsrt/results/'
    config['general_input']['ideal_geometry']     = True
    config['general_input']['backwards_raytrace'] = False
    config['general_input']['do_visualizations']  = False
    config['general_input']['do_savefiles']       = False
    config['general_input']['do_image_analysis']  = False
    config['general_input']['random_seed']        = 123456
    config['general_input']['scenario']           = 'beam'
    config['general_input']['system']             = 'w7x_ar16'
    config['general_input']['shot']               = 180707017

    # Temperature that the optical elements will be cooled to (kelvin)
    config['general_input']['xics_temp'] = 273.0


    # -----------------------------------------------------------------------------
    # Scenario properties

    # Each of these scenarios corresponds to a script located in xics_rt_tools.py
    # which assembles the optical elements into a specific configuration based on
    # input parameters

    # Load scenario properties
    config['scenario_input']['source_graphite_dist']  = 4.5
    config['scenario_input']['graphite_crystal_dist'] = 8.5
    config['scenario_input']['crystal_detector_dist'] = None
    config['scenario_input']['graphite_offset']       = np.array([0,0,0], dtype = np.float64)
    config['scenario_input']['graphite_tilt']         = np.array([0,0,0], dtype = np.float64)
    config['scenario_input']['crystal_offset']        = np.array([0,0,0], dtype = np.float64)
    config['scenario_input']['crystal_tilt']          = np.array([0,0,0], dtype = np.float64)
    config['scenario_input']['detector_offset']       = np.array([0,0,0], dtype = np.float64)
    config['scenario_input']['detector_tilt']         = np.array([0,0,0], dtype = np.float64)


    # -----------------------------------------------------------------------------
    ## Load plasma properties

    # bundle_count
    #   Increase to improve both plasma modeling and raytracing accuracy.
    # bundle_volume
    #   Increase to improve raytracing accuracy
    #   Decrease to improve plasma modeling
    #
    #   This currently only has an effect of bundle_type is 'voxel'.

    config['source_input']['position']            = np.array([-5.5, 0, 0])
    config['source_input']['normal']              = np.array([0, 0, 1])
    config['source_input']['orientation']         = np.array([0, 1, 0])
    config['source_input']['target']              = np.array([0, 0, 0])
    config['source_input']['width']               = 2.0
    config['source_input']['height']              = 2.0
    config['source_input']['depth']               = 2.0

    config['source_input']['major_radius']        = 6.2
    config['source_input']['minor_radius']        = 2.0

    config['source_input']['time_resolution']     = 0.01
    config['source_input']['bundle_count']        = 20
    config['source_input']['bundle_volume']       = 0.01**3
    config['source_input']['bundle_type']         = 'point'

    config['source_input']['spread']              = 1.0       #Angular spread (degrees)
    config['source_input']['temp']                = 1000      #Ion temperature (eV)
    config['source_input']['mass']                = 39.948    # Xenon mass (AMU)
    config['source_input']['wavelength']          = 3.9492    # Line location (angstroms)
    config['source_input']['linewidth']           = 1.129e+14 # Natural linewith (1/s)

    config['source_input']['plasma_type'] = 'vmec'
    config['source_input']['wout_file'] = '/u/npablant/data/w7x/vmec/webservice/w7x_ref_172/wout.nc'


    # -----------------------------------------------------------------------------
    ## Load spherical crystal properties

    # Rocking curve FWHM in radians
    # This is taken from x0h for quartz 1,1,-2,0
    # Darwin Curve, sigma: 48.070 urad
    # Darwin Curve, pi:    14.043 urad
    # Graphite Rocking Curve FWHM in radians
    # Taken from XOP: 8765 urad

    config['crystal_input']['position']           = [-8.6068906812402943e+00,  3.2920701414857128e+00,  7.3539419063116812e-02]
    config['crystal_input']['normal']             = [ 5.3519444199135369e-01, -8.4134020987066793e-01,  7.5588097716134145e-02]
    config['crystal_input']['orientation']        = [-8.4083033364093662e-01, -5.3917440198461375e-01, -4.7909438253911078e-02]

    config['crystal_input']['width']              = 0.040
    config['crystal_input']['height']             = 0.100
    config['crystal_input']['curvature']          = 1.4503999999999999e+00


    # Rocking curve FWHM in radians.
    # This is taken from x0h for quarts 1,1,-2,0
    # Darwin Curve, sigma: 48.070 urad
    # Darwin Curve, pi:    14.043 urad
    config['crystal_input']['spacing']            = 2.4567600000000001e+00
    config['crystal_input']['reflectivity']       = 1
    config['crystal_input']['rocking_curve']      = 48.070e-6
    config['crystal_input']['pixel_scaling']      = int(200)

    config['crystal_input']['therm_expand']       = 5.9e-6
    #config['crystal_input']['sigma_data']         = '../xicsrt/rocking_curve_germanium_sigma.txt'
    #config['crystal_input']['pi_data']            = '../xicsrt/rocking_curve_germanium_pi.txt'
    config['crystal_input']['sigma_data']         = '/u/npablant/code/mirproject/xicsrt/xicsrt/rocking_curve_germanium_sigma.txt'
    config['crystal_input']['pi_data']            = '/u/npablant/code/mirproject/xicsrt/xicsrt/rocking_curve_germanium_pi.txt'
    config['crystal_input']['mix_factor']         = 1.0

    config['crystal_input']['do_bragg_checks']    = True
    config['crystal_input']['do_miss_checks']     = True
    config['crystal_input']['rocking_curve_type'] = "gaussian"


    # -----------------------------------------------------------------------------
    ## Load detector properties

    config['detector_input']['position']          = [-8.6738784071336230e+00,  2.1399015950319900e+00,  1.0399766774640780e-01]
    config['detector_input']['normal']            = [ 5.9585883616345793e-02,  9.9785215153757567e-01, -2.7214079912620245e-02]
    config['detector_input']['orientation']       = [-9.9464373245879134e-01,  5.7043480061171735e-02, -8.6196791488749647e-02]

    config['detector_input']['pixel_size']        = 0.000172
    config['detector_input']['horizontal_pixels'] = 195
    config['detector_input']['vertical_pixels']   = 1475
    config['detector_input']['width']             = (config['detector_input']['horizontal_pixels']
                                                    * config['detector_input']['pixel_size'])
    config['detector_input']['height']            = (config['detector_input']['vertical_pixels']
                                                    * config['detector_input']['pixel_size'])

    config['detector_input']['do_miss_checks']    = True


    return config




