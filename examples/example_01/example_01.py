# -*- coding: utf-8 -*-
"""
..  Authors:
    Novimir Antoniuk Pablant <npablant@pppl.gov>


A slightly more complicated example with an x-ray Bragg reflection from a
spherical crystal.

This configuration has a point source, a spherical crystal, and a detector.
"""

import numpy as np
import xicsrt
xicsrt.warn_version('0.8')

# 1.
config = dict()

# 2.
config['general'] = {}
config['general']['number_of_iter'] = 5
config['general']['save_images'] = False

# 3.
config['sources'] = {}
config['sources']['source'] = {}
config['sources']['source']['class_name'] = 'XicsrtSourceDirected'
config['sources']['source']['intensity'] = 1e4
config['sources']['source']['wavelength'] = 3.9492
config['sources']['source']['spread'] = np.radians(10.0)
config['sources']['source']['xsize'] = 0.00
config['sources']['source']['ysize'] = 0.00
config['sources']['source']['zsize'] = 0.00

# 4.
config['optics'] = {}
config['optics']['crystal'] = {}
config['optics']['crystal']['class_name'] = 'XicsrtOpticCrystalSpherical'
config['optics']['crystal']['check_size'] = True
config['optics']['crystal']['origin'] = [0.0, 0.0,         0.80374151]
config['optics']['crystal']['zaxis']  = [0.0, 0.59497864, -0.80374151]
config['optics']['crystal']['xsize']  = 0.2
config['optics']['crystal']['ysize']  = 0.2
config['optics']['crystal']['radius'] = 1.0

# Rocking curve FWHM in radians.
# This is taken from x0h for quartz 1,1,-2,0
# Darwin Curve, sigma: 48.070 urad
# Darwin Curve, pi:    14.043 urad
config['optics']['crystal']['crystal_spacing'] = 2.45676
config['optics']['crystal']['rocking_type'] = 'gaussian'
config['optics']['crystal']['rocking_fwhm'] = 48.070e-6

# 5.
config['optics']['detector'] = {}
config['optics']['detector']['class_name'] = 'XicsrtOpticDetector'
config['optics']['detector']['origin'] = [0.0,  0.76871290, 0.56904832]
config['optics']['detector']['zaxis']  = [0.0, -0.95641806, 0.29200084]
config['optics']['detector']['xsize']  = 0.4
config['optics']['detector']['ysize']  = 0.2

# 6.
results = xicsrt.raytrace(config)