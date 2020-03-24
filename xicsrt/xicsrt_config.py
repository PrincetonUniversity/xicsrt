# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir pablant <npablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  - James Kring <jdk0026@tigermail.auburn.edu>
"""


import numpy as np
import logging
from collections import OrderedDict


def get_default_config(self):
    config = OrderedDict()

    config['general'] = OrderedDict()
    config['general']['number_of_iterations'] = 1
    config['general']['number_of_runs'] = 1

    config['sources'] = OrderedDict()
    
    config['optics'] = OrderedDict()
    
    return config
