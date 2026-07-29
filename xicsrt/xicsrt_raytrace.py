# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Opus 5).
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    James Kring <jdk0026@tigermail.auburn.edu>

Entry point to XICSRT.
Contains the main functions that are called to perform raytracing.
"""

import numpy as np

import os
import resource
import sys

from copy import deepcopy

from xicsrt.util import mirlogging
from xicsrt.util import profiler

from xicsrt import xicsrt_config
from xicsrt import xicsrt_io
from xicsrt.objects._Dispatcher import Dispatcher
from xicsrt.objects._RayArray import RayArray

m_log = mirlogging.getLogger(__name__)

def raytrace(config):
    """
    Perform a series of ray tracing runs.

    Each run will rebuild all objects, reset the random seed and then
    perform the requested number of iterations.

    If the option 'save_images' is set, then images will be saved
    at the completion of each run. The saving of these run images
    is one reason to use this routine rather than just increasing
    the number of iterations: periodic outputs during long computations.

    Also see :func:`~xicsrt.xicsrt_multiprocessing.raytrace` for a
    multiprocessing version of this routine.
    """
    profiler.start('raytrace')
    
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    check_config(config)

    # Make local copies of some options.
    num_runs = config['general']['number_of_runs']
    random_seed = config['general']['random_seed']
    
    output_list = []
    
    for ii in range(num_runs):
        m_log.info('Starting run: {} of {}'.format(ii + 1, num_runs))
        config_run = deepcopy(config)
        config_run['general']['output_run_suffix'] = '{:04d}'.format(ii)

        # Make sure each run uses a unique random seed.
        if random_seed is not None:
            random_seed += ii
        config_run['general']['random_seed'] = random_seed
        
        iteration = raytrace_single(config_run, _internal=True)
        # These runs execute in this process, so their timings are already in
        # the profiler global. Merging them again would double count.
        iteration.pop('profiler', None)
        output_list.append(iteration)

    output = combine_raytrace(output_list, consume_input=True)

    # Reset the configuration options that were unique to the individual runs.
    output['config']['general']['output_run_suffix'] = config['general']['output_run_suffix']
    output['config']['general']['random_seed'] = config['general']['random_seed']

    if config['general']['save_config']:
        xicsrt_io.save_config(output['config'])
    if config['general']['save_images']:
        xicsrt_io.save_images(output)
    if config['general']['save_results']:
        xicsrt_io.save_results(output)
    if config['general']['print_results']:
        print_raytrace(output)

    profiler.stop('raytrace')
    return output


def raytrace_single(config, _internal=False):
    """
    Perform a single raytrace run consisting of multiple iterations.

    If history is enabled, sort the rays into those that are detected and
    those that are lost (found and lost). The found ray history will be
    returned in full. The lost ray history will be truncated to allow
    analysis of lost ray pattern while still limiting memory usage.

    private keywords
    ================
    _internal : bool (False)
      Used when calling this function from `raytrace` as part of the execution
      of multiple runs. Controls how `history_max_lost` is handled along with
      how `save_config` and `save_results` are interpreted.
    """
    profiler.start('raytrace_single')

    # Update the default config with the user config.
    config = xicsrt_config.config_to_numpy(config)
    config = xicsrt_config.get_config(config)
    check_config(config)

    m_log.info('Seeding np.random with {}'.format(config['general']['random_seed']))
    np.random.seed(config['general']['random_seed'])

    # A dedicated generator for the lost-ray subsampling, kept separate from
    # the global stream that generates the rays. See `_sort_raytrace`.
    rng_lost = np.random.default_rng(config['general']['random_seed'])

    num_iter = config['general']['number_of_iter']
    max_lost_iter = int(config['general']['history_max_lost']/num_iter)

    if _internal:
        max_lost_iter = max_lost_iter//config['general']['number_of_runs']

    # Save at least one lost ray even if this exceeds history_max_lost.
    max_lost_iter = max(int(max_lost_iter), 1)

    # Setup the dispatchers.
    if 'filters' in config:
        m_log.debug("Creating filters")
        filters = Dispatcher(config, 'filters')
        filters.instantiate()
        filters.setup()
        filters.initialize()
        config['filters'] = filters.get_config()
    else:
        filters = None

    m_log.debug("Creating sources")
    sources = Dispatcher(config, 'sources')
    sources.instantiate()
    sources.apply_filters(filters)
    sources.setup()
    sources.check_param()
    sources.initialize()
    config['sources'] = sources.get_config()

    m_log.debug("Creating optics")
    optics = Dispatcher(config, 'optics')
    optics.instantiate()
    optics.apply_filters(filters)
    optics.setup()
    optics.check_param()
    optics.initialize()
    config['optics'] = optics.get_config()

    # Do the actual raytracing
    output_list = []
    for ii in range(num_iter):
        m_log.info('Starting iteration: {} of {}'.format(ii + 1, num_iter))

        single = _raytrace_iter(config, sources, optics)
        sorted = _sort_raytrace(single, max_lost=max_lost_iter, rng=rng_lost)
        _log_iter_diagnostics(sorted, single['history'])
        output_list.append(sorted)

        # Release the full-width history for this iteration now that it has
        # been reduced to found + sampled-lost rays.
        #
        # The dispatchers hold a deepcopy of the rays at every element, and
        # `single` holds references to those same arrays. Without this the
        # previous iteration's full history stays alive while the next
        # iteration allocates its own, roughly doubling the peak. Nothing
        # downstream reads them: `sorted` already owns fancy-indexed copies.
        sources.history.clear()
        optics.history.clear()
        del single

    output = combine_raytrace(output_list, consume_input=True)

    if _internal is False:
        if config['general']['print_results']:
            print_raytrace(output)
        if config['general']['save_config']:
            xicsrt_io.save_config(output['config'])
        if config['general']['save_results']:
            xicsrt_io.save_results(output)

    if config['general']['save_images']:
        xicsrt_io.save_images(output)

    profiler.stop('raytrace_single')

    if _internal and profiler.isEnabled():
        # The profiler results are a per-process global, so when this run is
        # executed in a multiprocessing worker the timings are invisible to
        # the parent unless they are shipped back explicitly. This key is
        # consumed by `xicsrt_multiprocessing.raytrace` and is dropped by
        # `combine_raytrace`, so it never reaches a saved results file.
        output['profiler'] = profiler.getResults()

    return output


def get_peak_rss():
    """
    Return the peak resident set size of this process in bytes.

    `resource.getrusage` reports `ru_maxrss` in kilobytes on Linux but in
    bytes on macOS, so the units are normalized here.

    This function was AI generated using Claude (Opus 5).
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return maxrss
    return maxrss * 1024


def get_history_bytes(history):
    """
    Return the total number of bytes held by a ray history dictionary.

    This function was AI generated using Claude (Opus 5).
    """
    total = 0
    for key_opt in history:
        for key_ray in history[key_opt]:
            total += history[key_opt][key_ray].nbytes
    return total


def _log_iter_diagnostics(sorted_output, history):
    """
    Log per-iteration ray counts and memory usage.

    These diagnostics exist because a long raytracing run gives the user very
    little insight into where memory is going; see F006. Everything logged
    here is derived from data that already exists, so this adds no measurable
    cost and does not touch the random number stream.

    This function was AI generated using Claude (Opus 5).
    """
    if not m_log.isEnabledFor(mirlogging.INFO):
        return

    peak_rss = get_peak_rss()

    if len(sorted_output['found']['history']) == 0:
        # With keep_history disabled the counts come from the metadata.
        key_opt_list = list(sorted_output['total']['meta'].keys())
        num_found = sorted_output['total']['meta'][key_opt_list[-1]]['num_out']
        m_log.info(
            'Iteration found: {}, history: disabled, peak rss: {:0.1f} MB'
            ''.format(num_found, peak_rss / 1024 ** 2))
        return

    key_opt_last = list(sorted_output['found']['history'].keys())[-1]
    num_found = len(sorted_output['found']['history'][key_opt_last]['mask'])
    num_lost = len(sorted_output['lost']['history'][key_opt_last]['mask'])

    full_bytes = get_history_bytes(history)
    kept_bytes = (get_history_bytes(sorted_output['found']['history'])
                  + get_history_bytes(sorted_output['lost']['history']))

    m_log.info(
        'Iteration found: {}, lost retained: {}, history: {:0.1f} MB traced '
        '-> {:0.1f} MB kept, peak rss: {:0.1f} MB'.format(
            num_found, num_lost,
            full_bytes / 1024 ** 2,
            kept_bytes / 1024 ** 2,
            peak_rss / 1024 ** 2))


def _raytrace_iter(config, sources, optics):
    """ 
    Perform a single iteration of raytracing with the given sources and optics.
    The returned rays are unsorted.
    """
    profiler.start('_raytrace_iter')

    # Setup local names for a few config entries.
    # This is only to make the code below more readable.
    keep_meta    = config['general']['keep_meta']
    keep_images  = config['general']['keep_images']
    keep_history = config['general']['keep_history']

    m_log.debug('Generating rays')
    rays = sources.generate_rays(keep_history=keep_history)
    m_log.debug('Raytracing optics')
    rays = optics.trace(rays, keep_history=keep_history, keep_images=keep_images)

    # Combine sources and optics outputs.
    meta    = dict()
    image   = dict()
    history = dict()
    
    if keep_meta:
        for key in sources.meta:
            meta[key] = sources.meta[key]
        for key in optics.meta:
            meta[key] = optics.meta[key]
    
    if keep_images:
        for key in sources.image:
            image[key] = sources.image[key]
        for key in optics.image:
            image[key] = optics.image[key]    
    
    if keep_history:
        for key in sources.history:
            history[key] = sources.history[key]
        for key in optics.history:
            history[key] = optics.history[key]

    output = dict()
    output['config'] = config
    output['meta'] = meta
    output['image'] = image
    output['history'] = history
    
    profiler.stop('_raytrace_iter')
    return output


def _sort_raytrace(input, max_lost=None, rng=None):
    """
    Sort the rays into 'lost' and 'found' rays, then truncate
    the number of lost rays.

    Parameters
    ----------
    input : dict
      The unsorted output of a single raytracing iteration.

    max_lost : int (1000)
      The maximum number of lost rays to retain.

    rng : numpy.random.Generator (None)
      The generator used to choose which lost rays to retain. A dedicated
      generator is used rather than the global `np.random` stream so that
      subsampling the lost rays cannot perturb the stream that produced the
      rays themselves; see the programming notes below. If None, a generator
      seeded from OS entropy is created, and the retained lost rays will not
      be reproducible. Both engine callers supply a seeded generator.

    Programming Notes
    -----------------

    `np.random.shuffle` draws a number of values from the global random
    stream that depends on the number of lost rays. Since this function runs
    inside the iteration loop, any use of the global stream here couples the
    lost-ray bookkeeping to the rays generated by every later iteration. A
    dedicated `Generator` removes that coupling entirely: the found rays are
    then bit-identical regardless of how many rays were lost or how many are
    retained.
    """
    if max_lost is None:
        max_lost = 1000
    if rng is None:
        rng = np.random.default_rng()

    profiler.start('_sort_raytrace')

    output = dict()
    output['config'] = input['config']
    output['total'] = dict()
    output['total']['meta'] = dict()
    output['total']['image'] = dict()
    output['found'] = dict()
    output['found']['meta'] = dict()
    output['found']['history'] = dict()
    output['lost'] = dict()
    output['lost']['meta'] = dict()
    output['lost']['history'] = dict()

    output['total']['meta'] = input['meta']
    output['total']['image'] = input['image']

    if len(input['history']) > 0:
        key_opt_list = list(input['history'].keys())
        key_opt_last = key_opt_list[-1]

        mask_last = input['history'][key_opt_last]['mask']
        w_found = np.flatnonzero(mask_last)
        w_lost = np.flatnonzero(np.invert(mask_last))

        # Save only a portion of the lost rays so that our lost history does
        # not become too large. Choosing the retained rays directly avoids
        # shuffling an index array of every lost ray, which for a typical
        # x-ray trace means shuffling millions of entries to keep a few
        # hundred.
        max_lost = min(max_lost, len(w_lost))
        w_lost = rng.choice(w_lost, size=max_lost, replace=False)

        for key_opt in key_opt_list:
            output['found']['history'][key_opt] = dict()
            output['lost']['history'][key_opt] = dict()

            for key_ray in input['history'][key_opt]:
                output['found']['history'][key_opt][key_ray] = input['history'][key_opt][key_ray][w_found]
                output['lost']['history'][key_opt][key_ray] = input['history'][key_opt][key_ray][w_lost]

    profiler.stop('_sort_raytrace')

    return output


def combine_raytrace(input_list,
                     keep_images=True,
                     components=None,
                     consume_input=False):
    """
    Produce a combined results dictionary from a list of raytrace results.

    Keywords
    --------

    keep_images: bool (True)
        Control whether to combine images.
        Useful when combining outputs which use different optics sizes.

    components: list (None)
        A list of specific components to combine.
        Useful when not all components are needed in the final output.

    consume_input: bool (False)
        If True, each input history is discarded as soon as it has been
        copied into the output. This roughly halves the peak memory of the
        combine, since the inputs and the combined output are otherwise both
        fully resident when the last history is copied. The inputs are left
        unusable, so this is only appropriate for internal callers that own
        their input list. Used by the engine when combining iterations and
        runs.

    Example
    -------
    results_1 = xicsrt.raytrace(config_1)
    results_2 = xicsrt.raytrace(config_2)
    results = xicsrt_raytrace.combine_raytrace([results_1, results_2])
    """
    profiler.start('combine_raytrace')

    output = dict()
    output['config'] = input_list[0]['config']
    output['total'] = dict()
    output['total']['meta'] = dict()
    output['total']['image'] = dict()
    output['found'] = dict()
    output['found']['meta'] = dict()
    output['found']['history'] = dict()
    output['lost'] = dict()
    output['lost']['meta'] = dict()
    output['lost']['history'] = dict()

    num_iter = len(input_list)

    if components is None:
        key_opt_list = list(input_list[0]['total']['meta'].keys())
    else:
        key_opt_list = components

    key_opt_last = key_opt_list[-1]

    # Combine the meta data.
    for key_opt in key_opt_list:
        output['total']['meta'][key_opt] = dict()
        key_meta_list = list(input_list[0]['total']['meta'][key_opt].keys())
        for key_meta in key_meta_list:
            output['total']['meta'][key_opt][key_meta] = 0
            for ii_iter in range(num_iter):
                output['total']['meta'][key_opt][key_meta] += input_list[ii_iter]['total']['meta'][key_opt][key_meta]

    # Combine the images.
    if keep_images:
        for key_opt in key_opt_list:
            if key_opt in input_list[0]['total']['image']:
                if input_list[0]['total']['image'][key_opt] is not None:
                    # Check shape compatibility
                    shape = input_list[0]['total']['image'][key_opt].shape
                    shape_match = True
                    for ii_iter in range(num_iter):
                        if not input_list[ii_iter]['total']['image'][key_opt].shape == shape:
                            shape_match = False
                    if shape_match:
                        # Shapes match, combine images
                        output['total']['image'][key_opt] = np.zeros(input_list[0]['total']['image'][key_opt].shape)
                        for ii_iter in range(num_iter):
                            output['total']['image'][key_opt] += input_list[ii_iter]['total']['image'][key_opt]
                    else:
                        m_log.warning('Image dimensions do not match. Cannot combine images.')
                        output['total']['image'][key_opt] = None
                else:
                    output['total']['image'][key_opt] = None

    # Combine all the histories.
    #
    # The output arrays are keyed off the *input* ray keys rather than off
    # `RayArray.zeros`, which only defines origin/direction/mask/wavelength
    # and therefore used to silently discard the 'weight' array. They are
    # allocated with `np.empty` and fully overwritten below, which avoids
    # materializing every array twice (once zeroed, once copied).
    if len(input_list[0]['found']['history']) > 0:
        final_num_found = 0
        final_num_lost = 0
        for ii_run in range(num_iter):
            final_num_found += len(input_list[ii_run]['found']['history'][key_opt_last]['mask'])
            final_num_lost += len(input_list[ii_run]['lost']['history'][key_opt_last]['mask'])

        for key_opt in key_opt_list:
            output['found']['history'][key_opt] = RayArray()
            output['lost']['history'][key_opt] = RayArray()

            for key_ray, array in input_list[0]['found']['history'][key_opt].items():
                shape_found = (final_num_found,) + array.shape[1:]
                shape_lost = (final_num_lost,) + array.shape[1:]
                output['found']['history'][key_opt][key_ray] = np.empty(
                    shape_found, dtype=array.dtype)
                output['lost']['history'][key_opt][key_ray] = np.empty(
                    shape_lost, dtype=array.dtype)

        index_found = 0
        index_lost = 0
        for ii_run in range(num_iter):
            num_found = len(input_list[ii_run]['found']['history'][key_opt_last]['mask'])
            num_lost = len(input_list[ii_run]['lost']['history'][key_opt_last]['mask'])

            for key_opt in key_opt_list:
                for key_ray in output['found']['history'][key_opt]:
                    output['found']['history'][key_opt][key_ray][index_found:index_found + num_found] = (
                        input_list[ii_run]['found']['history'][key_opt][key_ray][:])
                    output['lost']['history'][key_opt][key_ray][index_lost:index_lost + num_lost] = (
                        input_list[ii_run]['lost']['history'][key_opt][key_ray][:])

            index_found += num_found
            index_lost += num_lost

            # Free each input history as it is consumed, so that the inputs
            # and the combined output are not both fully resident at the end.
            if consume_input:
                input_list[ii_run]['found']['history'].clear()
                input_list[ii_run]['lost']['history'].clear()

    profiler.stop('combine_raytrace')
    return output


def check_config(config):
    """
    Check the general section of the configuration dictionary.
    """

    # Check if anything needs to be saved.
    do_save = False
    for key in config['general']:
        if 'save' in key:
            if config['general'][key]:
                do_save = True

    if do_save:
        if not xicsrt_io.path_exists(config['general']['output_path']):
            if not config['general']['make_directories']:
                raise Exception('Output directory does not exist. Create directory or set make_directories to True.')


def print_raytrace(results):
    """
    Print out some information and statistics from the raytracing results.
    """

    key_opt_list = list(results['total']['meta'].keys())
    num_source = results['total']['meta'][key_opt_list[0]]['num_out']
    num_detector = results['total']['meta'][key_opt_list[-1]]['num_out']
    
    print('')
    print('Rays Generated: {:6.3e}'.format(num_source))
    print('Rays Detected:  {:6.3e}'.format(num_detector))
    print('Efficiency:     {:6.3e} ± {:3.1e} ({:7.5f}%)'.format(
        num_detector / num_source,
        np.sqrt(num_detector) / num_source,
        num_detector / num_source * 100))
    print('')

