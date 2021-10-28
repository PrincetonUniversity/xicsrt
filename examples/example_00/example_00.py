# -*- coding: utf-8 -*-
"""
..  Authors:
    Novimir Antoniuk Pablant <npablant@pppl.gov>

A simple example consisting only of a point source and a spherical crystal.

Description
-----------

1.
Create a new user configuration dictionary.

The entries that we put into this config will overwrite the defaults
that are defined within xicsrt. The config can potentially contain the
following sections:

- general
- sources
- optics
- filters
- scenario

2.
Create a section that contains the general raytracer configuration.

number_of_iter
  Perform raytracing the given number of times. The output from all
  the iterations will be combined. Performing multiple iterations allows
  a large number of rays to be traced without running into memory limits.
save_images
  If set to true, images will be saved to the output directory (which
  we have not specified in this example.

3.
Create the section that contains the sources.
Then define a source, cleverly named 'source'.

class_name
  The type of source object to create.
intensity
  The number of rays to launch in each iteration.
wavelength
  The nominal wavelength of the source emission.
spread
   The angular spread of the source (in radians).

4.
Create the section that contains the optics.
In this case we only define one optic: a detector.

class_name
  The type of optic object to create.
origin
  The location of this optic.
zaxis
  The direction the optics is pointing. For all of the standard
  optics that come with xicrt, the zaxis is the normal direction.
xsize
  The size of the optic along the xaxis.
  Corresponds to the 'width' of the optic.
ysize
  The size of the optic along the yaxis.
  Corresponds to the 'height' of the optic.

5.
Finally we pass the configuration to the XICSRT raytracer to perform
the actual raytracing. The `results` is a dictionary with the full
trace history along with images at the detector.
"""

import numpy as np
import xicsrt
xicsrt.warn_version('0.8')

# 1.
config = {}

# 2.
config['general'] = {}
config['general']['number_of_iter'] = 5
config['general']['save_images'] = False

# 3.
config['sources'] = {}
config['sources']['source'] = {}
config['sources']['source']['class_name'] = 'XicsrtSourceDirected'
config['sources']['source']['intensity'] = 1e3
config['sources']['source']['wavelength'] = 3.9492
config['sources']['source']['spread'] = np.radians(5.0)

# 4.
config['optics'] = {}
config['optics']['detector'] = {}
config['optics']['detector']['class_name'] = 'XicsrtOpticDetector'
config['optics']['detector']['origin'] = [0.0, 0.0, 1.0]
config['optics']['detector']['zaxis']  = [0.0, 0.0, -1]
config['optics']['detector']['xsize']  = 0.2
config['optics']['detector']['ysize']  = 0.2

# 5.
results = xicsrt.raytrace(config)

