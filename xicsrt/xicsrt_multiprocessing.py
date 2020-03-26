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

from copy import deepcopy
from collections import OrderedDict

from multiprocessing import Pool

from xicsrt.util import profiler

from xicsrt import xicsrt_config
from xicsrt.xicsrt_raytrace import *

def raytrace_multiprocessing(config):
    """
    Perform a series of ray tracing runs.

    Each raytracing run will perform the requested number of iterations.
    Each run will produce a single output image.
    """
    profiler.start('raytrace_multiprocessing')
    
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    
    num_runs = config['general']['number_of_runs']
    random_seed = config['general']['random_seed']

    mp_result_list = []
    output_list = []

    with Pool() as pool:
        # loop through each configuration in the configuration input file
        # and add a new run into the pool.
        for ii in range(num_runs):
            logging.info('Adding run to pool: {} of {}'.format(ii + 1, num_runs))

            # Make a copy of the configuration.              
            config_temp = deepcopy(config)
            
            # Make sure each run uses a unique random seed.
            if random_seed is not None:
                random_seed += ii
            config_temp['general']['random_seed'] = random_seed
                
            arg = (config,)
            mp_result = pool.apply_async(raytrace, arg)
            mp_result_list.append(mp_result)
        pool.close()
        pool.join()

    profiler.start('multiprocessing: gathering')
    # Gather all the results together.
    for mp_result in mp_result_list:
        output = mp_result.get()
        output_list.append(output)
    profiler.stop('multiprocessing: gathering')

    output = combine_raytrace(output_list)

    profiler.stop('raytrace_multiprocessing')
    return output
