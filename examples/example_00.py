# -*- coding: utf-8 -*-
"""
Authors:
  | Novimir Antoniuk Pablant <npablant@pppl.gov>

Description:
  A simple example consisting only of a point source and a spherical crystal.
"""

import numpy as np
from collections import OrderedDict

# Import xicsrt modules
from xicsrt import xicsrt_raytrace

config = OrderedDict()

config['general'] = OrderedDict()
config['general']['number_of_iter'] = 5

config['general']['output_path'] = ''
config['general']['save_images'] = False

config['sources'] = OrderedDict()
config['sources']['source'] = OrderedDict()
config['sources']['source']['class_name'] = 'XicsrtSourceDirected'
config['sources']['source']['intensity'] = 1e4
config['sources']['source']['wavelength'] = 3.9492
config['sources']['source']['spread'] = np.radians(0.1)

config['optics'] = OrderedDict()
config['optics']['crystal'] = OrderedDict()
config['optics']['crystal']['class_name'] = 'XicsrtOpticCrystalSpherical'
config['optics']['crystal']['do_miss_check'] = True
config['optics']['crystal']['origin'] = [0.0,0.0,1.0]
config['optics']['crystal']['zaxis'] = [0.00000000, 0.59497864, -0.80374151]
config['optics']['crystal']['width']  = 0.001
config['optics']['crystal']['height'] = 0.001
config['optics']['crystal']['radius'] = 1.0

# Rocking curve FWHM in radians.
# This is taken from x0h for quartz 1,1,-2,0
# Darwin Curve, sigma: 48.070 urad
# Darwin Curve, pi:    14.043 urad
config['optics']['crystal']['crystal_spacing'] = 2.45676
config['optics']['crystal']['rocking_type'] = 'gaussian'
config['optics']['crystal']['rocking_fwhm'] = 48.070e-6

output = xicsrt_raytrace.raytrace(config)
