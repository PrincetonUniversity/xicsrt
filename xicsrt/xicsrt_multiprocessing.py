# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir pablant <npablant@pppl.gov>
"""

from multiprocessing import Pool
from xicsrt.xicsrt_raytrace import *

log = mirlogging.getLogger(__name__)

def raytrace(config, processes=None):
    """
    Perform a series of ray tracing runs using the `multiprocessing` cpu pool.

    Each run will rebuild all objects, reset the random seed and then
    perform the requested number of iterations.

    If the option 'save_images' is set, then images will be saved
    at the completion of each run.

    Also see :func:`~xicsrt.xicsrt_raytrace.raytrace` for a single process
    version of this routine.
    """
    profiler.start('raytrace_multiprocessing')
    
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    check_config(config)
    
    num_runs = config['general']['number_of_runs']
    random_seed = config['general']['random_seed']

    mp_result_list = []
    output_list = []

    with Pool(processes) as pool:
        # loop through each configuration in the configuration input file
        # and add a new run into the pool.
        for ii in range(num_runs):
            log.info('Adding run to pool: {} of {}'.format(ii + 1, num_runs))

            # Make a copy of the configuration.              
            config_run = deepcopy(config)         
            config_run['general']['output_run_suffix'] = '{:04d}'.format(ii)
            
            # Make sure each run uses a unique random seed.
            if random_seed is not None:
                random_seed += ii
            config_run['general']['random_seed'] = random_seed
                
            arg = (config_run, True)
            mp_result = pool.apply_async(raytrace_single, arg)
            mp_result_list.append(mp_result)
        pool.close()
        pool.join()

    profiler.start('mp: gathering')
    # Gather all the results together.
    for mp_result in mp_result_list:
        output = mp_result.get()
        output_list.append(output)
    profiler.stop('mp: gathering')

    output = combine_raytrace(output_list)

    # Reset the configuration options that were unique to the individual runs.
    output['config']['general']['output_run_suffix'] = config['general']['output_run_suffix']
    output['config']['general']['random_seed'] = config['general']['output_run_suffix']

    if config['general']['save_config']:
        xicsrt_io.save_config(output['config'])
    if config['general']['save_images']:
        xicsrt_io.save_images(output)
    if config['general']['save_results']:
        xicsrt_io.save_results(output)
    if config['general']['print_results']:
        print_raytrace(output)
        
    profiler.stop('raytrace_multiprocessing')
    return output
