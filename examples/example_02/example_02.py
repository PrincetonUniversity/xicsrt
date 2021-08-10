# -*- coding: utf-8 -*-
"""
..  Authors:
    Novimir Antoniuk Pablant <npablant@pppl.gov>

An example showing how to define a complex aperture.
"""

import numpy as np
import xicsrt
xicsrt.warn_version('0.8')

config = {}

config['general'] = {}
config['general']['number_of_iter'] = 5
config['general']['save_images'] = False
config['general']['random_seed'] = 0

config['sources'] = {}
config['sources']['source'] = {}
config['sources']['source']['class_name'] = 'XicsrtSourceDirected'
config['sources']['source']['intensity'] = 1e3
config['sources']['source']['wavelength'] = 3.9492
config['sources']['source']['angular_dist'] = 'isotropic_xy'
config['sources']['source']['spread'] = np.radians(6.0)

config['optics'] = {}
config['optics']['aperture'] = {}
config['optics']['aperture']['class_name'] = 'XicsrtOpticAperture'
config['optics']['aperture']['origin'] = [0.0, 0.0, 0.8]
config['optics']['aperture']['zaxis'] = [0.0, 0.0, -1]
config['optics']['aperture']['aperture']=[
    {'shape':'circle', 'size':[0.075], 'logic':'and'},
    {'shape':'circle', 'size':[0.065], 'origin':[-0.010, -0.01],  'logic':'not'},
    {'shape':'circle', 'size':[0.048], 'origin':[-0.027, -0.01],  'logic':'or'},
    {'shape':'circle', 'size':[0.044], 'origin':[-0.032, -0.015], 'logic':'not'},
    {'shape':'circle', 'size':[0.034], 'origin':[-0.041, -0.013], 'logic':'or'},
    {'shape':'circle', 'size':[0.032], 'origin':[-0.045, -0.018], 'logic':'not'},
    {'shape':'circle', 'size':[0.025], 'origin':[-0.038, -0.020], 'logic':'or'},
    ]

config['optics']['detector'] = {}
config['optics']['detector']['class_name'] = 'XicsrtOpticDetector'
config['optics']['detector']['origin'] = [0.0, 0.0, 1.0]
config['optics']['detector']['zaxis'] = [0.0, 0.0, -1]
config['optics']['detector']['xsize'] = 0.2
config['optics']['detector']['ysize'] = 0.2


results = xicsrt.raytrace(config)