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

def get_config():
    config = OrderedDict()
    config['general'] = OrderedDict()
    config['sources'] = OrderedDict()
    config['optics']  = OrderedDict()

    # ----------------------------
    # General raytracer properties
    config['general']['number_of_iter']                = 1
    config['general']['number_of_runs']                = 1

    config['general']['pathlist_objects']              = [
        '/u/npablant/code/mirproject/xicsrt/xicsrt/sources'
        ,'/u/npablant/code/mirproject/xicsrt/xicsrt/optics']
    
    config['general']['path_ouput']                    = '/u/npablant/code/mirproject/xicsrt/results/temp'
    config['general']['do_savefiles']                  = False
    config['general']['random_seed']                   = 0


    # -----------------------
    ## Plasma properties

    # bundle_count
    #   Increase to improve both plasma modeling and raytracing accuracy.
    # bundle_volume
    #   Increase to improve raytracing accuracy
    #   Decrease to improve plasma modeling
    #
    #   This only has an effect of bundle_type is 'voxel'.

    config['sources']['plasma'] = OrderedDict()
    config['sources']['plasma']['class_name']          = 'XicsrtPlasmaW7xSimple'
    config['sources']['plasma']['origin']              = [-5.15, 1.95, 0.15]
    config['sources']['plasma']['zaxis']               = [0, 0, 1]
    config['sources']['plasma']['xaxis']               = [0, 1, 0]
    config['sources']['plasma']['target']              = [0, 0, 0]
    config['sources']['plasma']['width']               = 0.5
    config['sources']['plasma']['height']              = 1.7
    config['sources']['plasma']['depth']               = 1.5

    config['sources']['plasma']['use_poisson']         = True
    config['sources']['plasma']['time_resolution']     = 1e-5
    config['sources']['plasma']['bundle_count']        = 1e4
    config['sources']['plasma']['bundle_volume']       = 0.01**3
    config['sources']['plasma']['bundle_type']         = 'point'
    config['sources']['plasma']['max_rays']            = 1e7
    
    config['sources']['plasma']['emissivity_scale']    = 1e14
    config['sources']['plasma']['temperature_scale']   = 1.5e3
    config['sources']['plasma']['velocity_scale']      = 10e3

    config['sources']['plasma']['spread']              = 1.0       # Angular spread (degrees)
    config['sources']['plasma']['mass_number']         = 39.948    # Argon mass (AMU)
    config['sources']['plasma']['wavelength']          = 3.9492    # Line location (angstroms)
    config['sources']['plasma']['linewidth']           = 1.129e+14 # Natural linewith (1/s)
    config['sources']['plasma']['emissivity']          = 1e12      # Emissivity photons/s/m-3
    config['sources']['plasma']['temperature']         = 1000      # Ion temperature (eV)
    config['sources']['plasma']['velocity']            = np.array([0.0,0.0,0.0]) # Velocity in m/s
    config['sources']['plasma']['wout_file']           = '/u/npablant/data/w7x/vmec/webservice/w7x_ref_172/wout.nc'


    # -----------------------------------------------------------------------------
    ## Load spherical crystal properties

    # Rocking curve FWHM in radians
    # This is taken from x0h for quartz 1,1,-2,0
    # Darwin Curve, sigma: 48.070 urad
    # Darwin Curve, pi:    14.043 urad
    # Graphite Rocking Curve FWHM in radians
    # Taken from XOP: 8765 urad

    config['optics']['crystal'] = OrderedDict()
    config['optics']['crystal']['class_name']          = 'XicsrtOpticCrystalSpherical'
    config['optics']['crystal']['origin'] = [-8.6068906812402943e+00,  3.2920701414857128e+00,  7.3539419063116812e-02]
    config['optics']['crystal']['zaxis']  = [ 5.3519444199135369e-01, -8.4134020987066793e-01,  7.5588097716134145e-02]
    config['optics']['crystal']['xaxis']  = [-8.4083033364093662e-01, -5.3917440198461375e-01, -4.7909438253911078e-02]

    config['optics']['crystal']['width']               = 0.040
    config['optics']['crystal']['height']              = 0.100
    config['optics']['crystal']['radius']              = 1.4503999999999999e+00


    # Rocking curve FWHM in radians.
    # This is taken from x0h for quarts 1,1,-2,0
    # Darwin Curve, sigma: 48.070 urad
    # Darwin Curve, pi:    14.043 urad
    config['optics']['crystal']['crystal_spacing']     = 2.4567600000000001e+00
    config['optics']['crystal']['reflectivity']        = 1
    config['optics']['crystal']['rocking_type']        = 'gaussian'
    config['optics']['crystal']['rocking_fwhm']        = 48.070e-6
    config['optics']['crystal']['pixel_size']          = 0.040/100

    config['optics']['crystal']['do_bragg_check']     = True
    config['optics']['crystal']['do_miss_check']      = True


    # -----------------------------------------------------------------------------
    ## Load detector properties

    config['optics']['detector'] = OrderedDict()
    config['optics']['detector']['class_name']         = 'XicsrtOpticDetector'
    
    config['optics']['detector']['origin'] = [-8.6738784071336230e+00,  2.1399015950319900e+00,  1.0399766774640780e-01]
    config['optics']['detector']['zaxis']  = [ 5.9585883616345793e-02,  9.9785215153757567e-01, -2.7214079912620245e-02]
    config['optics']['detector']['xaxis']  = [-9.9464373245879134e-01,  5.7043480061171735e-02, -8.6196791488749647e-02]

    config['optics']['detector']['pixel_size']         = 0.000172
    config['optics']['detector']['pixel_width']        = 195
    config['optics']['detector']['pixel_height']       = 1475
    config['optics']['detector']['width']              = (config['optics']['detector']['pixel_width']
                                                          * config['optics']['detector']['pixel_size'])
    config['optics']['detector']['height']             = (config['optics']['detector']['pixel_height']
                                                          * config['optics']['detector']['pixel_size'])

    config['optics']['detector']['do_miss_check']     = True

    return config


def initialize(config):

    # Setup our plasma box to be radial.
    config['sources']['plasma']['zaxis'] = config['sources']['plasma']['origin'].copy()
    config['sources']['plasma']['zaxis'] /= np.linalg.norm(config['sources']['plasma']['zaxis'])

    config['sources']['plasma']['xaxis'] = np.cross(config['sources']['plasma']['zaxis'], np.array([0,0,1]))
    config['sources']['plasma']['xaxis'] /= np.linalg.norm(config['sources']['plasma']['xaxis'])

    config['sources']['plasma']['target'] = config['optics']['crystal']['origin'].copy()

    return config
