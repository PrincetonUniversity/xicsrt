# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Entry point of the jaxrt engine.

Contains the main `raytrace` orchestration: the runs and iterations
loops, jax random key threading, and assembly of the results
dictionary. The structure mirrors :mod:`xicsrt.xicsrt_raytrace`, and
the results sorting/combining routines from the numpy engine are
reused directly so the output dictionary is identical in structure.

Random key threading
--------------------
Each run derives an integer seed (user seed + run number, or a fresh
random seed) and creates a jax PRNG key from it. The key is split once
per iteration, and inside the trace each element receives its own
subkey. This is statistically equivalent to (but not bit-identical
with) the numpy engine. Because each run has an independent seed
offset, results from separate jobs (e.g. SLURM arrays) combine exactly
as with the numpy engine.

jit boundary
------------
A single function containing source generation plus the full optic
chain is jit-compiled once per scenario (geometry enters as closure
constants) and then reused for every iteration and run.

This module was AI generated using Claude (Fable 5).
"""

from copy import deepcopy

import jax
import numpy as np

from xicsrt import xicsrt_config
from xicsrt import xicsrt_io
from xicsrt.util import mirlogging
from xicsrt.xicsrt_raytrace import (
    _sort_raytrace, check_config, combine_raytrace, print_raytrace)

from xicsrt.jaxrt import _bounds
from xicsrt.jaxrt import _dispatch
from xicsrt.jaxrt import _images
from xicsrt.jaxrt.sources import _generic

m_log = mirlogging.getLogger(__name__)


def raytrace(config):
    """
    Perform a series of ray tracing runs using the jaxrt engine.

    Accepts the same config dictionary as :func:`xicsrt.raytrace` and
    returns the same results dictionary structure.
    """
    config = xicsrt_config.get_config(config)
    check_config(config)

    num_runs = config['general']['number_of_runs']
    random_seed = config['general']['random_seed']
    if random_seed is None:
        # Draw a random base seed so that separate unseeded calls are
        # independent, while runs within this call remain related by
        # fixed offsets.
        random_seed = int(np.random.default_rng().integers(2**31))

    output_list = []
    for ii in range(num_runs):
        m_log.info('Starting run: {} of {}'.format(ii + 1, num_runs))
        config_run = deepcopy(config)
        config_run['general']['output_run_suffix'] = '{:04d}'.format(ii)
        config_run['general']['random_seed'] = random_seed + ii

        output_run = raytrace_single(config_run, _internal=True)
        output_list.append(output_run)

    output = combine_raytrace(output_list)

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

    return output


def raytrace_single(config, _internal=False):
    """
    Perform a single raytrace run consisting of multiple iterations.
    """
    config = xicsrt_config.config_to_numpy(config)
    config = xicsrt_config.get_config(config)
    check_config(config)

    seed = config['general']['random_seed']
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))

    # np.random is used by the numpy element setup (e.g. source Poisson
    # draws, which jaxrt overrides) and by the lost-ray subsampling in
    # _sort_raytrace.
    m_log.info('Seeding np.random with {}'.format(seed))
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    num_iter = config['general']['number_of_iter']
    max_lost_iter = int(config['general']['history_max_lost'] / num_iter)
    if _internal:
        max_lost_iter = max_lost_iter // config['general']['number_of_runs']
    # Save at least one lost ray even if this exceeds history_max_lost.
    max_lost_iter = max(int(max_lost_iter), 1)

    if config.get('filters'):
        raise NotImplementedError('Filters are not supported by the jaxrt engine.')

    m_log.debug('Creating sources')
    sources = _dispatch.setup_sources(config)
    m_log.debug('Creating optics')
    optics = _dispatch.setup_optics(config)

    trace_iteration = _build_trace(config, sources, optics)

    # Element order matters downstream (_sort_raytrace and
    # print_raytrace use the first/last keys); jax.jit returns pytree
    # dicts with sorted keys, so the trace order is restored explicitly.
    element_order = list(sources) + list(optics)

    output_list = []
    for ii in range(num_iter):
        m_log.info('Starting iteration: {} of {}'.format(ii + 1, num_iter))
        key, key_iter = jax.random.split(key)

        single = trace_iteration(key_iter)
        single = _to_host(config, single, element_order)
        sorted_single = _sort_raytrace(single, max_lost=max_lost_iter)
        output_list.append(sorted_single)

    output = combine_raytrace(output_list)

    if _internal is False:
        if config['general']['print_results']:
            print_raytrace(output)
        if config['general']['save_config']:
            xicsrt_io.save_config(output['config'])
        if config['general']['save_results']:
            xicsrt_io.save_results(output)

    if config['general']['save_images']:
        xicsrt_io.save_images(output)

    return output


def _build_trace(config, sources, optics):
    """
    Build the jit-compiled trace function for one iteration.

    The returned function maps a jax random key to a dict with the
    per-element ray snapshots ('history'), detector images ('image'),
    live-ray counts ('num_out') and the Poisson overflow flag.
    All element geometry enters as closure constants, so the function
    compiles once and is reused for every iteration and run.
    """
    keep_images = config['general']['keep_images']
    keep_history = config['general']['keep_history']

    if len(sources) == 0:
        raise Exception('No ray sources defined.')
    elif len(sources) > 1:
        raise NotImplementedError('Multiple ray sources are not currently supported.')

    source_name = list(sources)[0]
    source = sources[source_name]

    def _trace(key):
        output = {'history': {}, 'image': {}, 'num_out': {}}

        key_source, key_optics = jax.random.split(key)
        rays = _generic.generate(source, key_source)
        output['overflow'] = rays.pop('overflow')

        output['num_out'][source_name] = jax.numpy.sum(rays['mask'])
        if keep_history:
            output['history'][source_name] = rays

        for name, optic in optics.items():
            key_optics, key_element = jax.random.split(key_optics)

            xloc, norm, mask = optic['shape'].intersect(rays, optic['geom'])
            mask = _bounds.check_bounds(xloc, mask, optic['bounds'])
            rays = optic['interact'].interact(
                rays, xloc, norm, mask, optic['phys'], key_element)

            output['num_out'][name] = jax.numpy.sum(rays['mask'])
            if keep_history:
                output['history'][name] = rays
            if keep_images and optic['image'] is not None:
                output['image'][name] = _images.make_image(rays, optic['image'])

        return output

    return jax.jit(_trace)


def _to_host(config, single, element_order):
    """
    Convert the traced output of one iteration to host numpy arrays and
    arrange it in the same structure as the numpy engine's
    `_raytrace_iter` output. Dictionaries are rebuilt in trace order
    (source first, last optic last), which the downstream sorting and
    reporting routines rely on.
    """
    single = jax.device_get(single)

    if single['overflow']:
        raise RuntimeError(
            'Poisson ray count exceeded the fixed array capacity '
            '(mean + 10 sigma). This is a ~1e-23 probability event; if it '
            'occurs repeatedly something is wrong with the configuration.')

    keep_meta = config['general']['keep_meta']
    keep_images = config['general']['keep_images']
    keep_history = config['general']['keep_history']

    output = {'config': config, 'meta': {}, 'image': {}, 'history': {}}

    if keep_meta:
        for name in element_order:
            output['meta'][name] = {'num_out': single['num_out'][name]}

    if keep_images:
        for name in element_order:
            if name in single['image']:
                output['image'][name] = np.asarray(single['image'][name])

    if keep_history:
        for name in element_order:
            output['history'][name] = {
                key: np.asarray(value)
                for key, value in single['history'][name].items()}

    return output
