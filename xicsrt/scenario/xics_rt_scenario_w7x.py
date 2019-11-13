# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir pablant <npablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------

"""

from xicsrt.util import profiler

profiler.start('Import Time')
import numpy as np

from xicsrt.xics_rt_raytrace   import raytrace
from xicsrt import xics_rt_input

from xicsrt.xics_rt_detectors  import Detector
from xicsrt.xics_rt_optics     import SphericalCrystal
from xicsrt.plasma.xics_rt_plasma_w7x_simple import W7xSimplePlasma

profiler.stop('Import Time')

def initialize(config):

    # Setup our plasma box to be radial.
    config['source_input']['normal'] = config['source_input']['position']
    config['source_input']['normal'] /= np.linalg.norm(config['source_input']['normal'])

    config['source_input']['orientation'] = np.cross(config['source_input']['normal'], np.array([0,0,1]))
    config['source_input']['orientation'] /= np.linalg.norm(config['source_input']['orientation'])

    config['source_input']['target'] = config['crystal_input']['position']

    xics_rt_input.config_to_numpy(config)
    return config


def run(config):

    # Initialize the random seed.
    np.random.seed(config['general_input']['random_seed'])

    profiler.start('Class Setup Time')
    detector = Detector(config['detector_input'])
    crystal = SphericalCrystal(config['crystal_input'])
    plasma = W7xSimplePlasma(config['source_input'])

    profiler.stop('Class Setup Time')

    scenario = str.lower(config['general_input']['scenario'])

    ## Raytrace Runs

    output, meta = raytrace(
        plasma
        ,detector
        ,crystal
        ,number_of_runs=config['general_input']['number_of_runs']
        ,collect_optics=True)

    return output, meta