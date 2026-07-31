# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Sonnet 5).
"""
Tests for the `shuffle_history` option (F012).

Covers:
  * `_sort_raytrace` unit-level behavior on synthetic bundle-blocked data:
    shuffling breaks up contiguous blocks, the found-ray *set* is identical
    with shuffling on or off, and all elements share one permutation.
  * `shuffle_history=False` reproduces the original bundle-blocked order and
    `shuffle_history=True` (the default) does not, using a real plasma
    source end to end through `xicsrt.raytrace`.
  * The lost-ray subsampling (`rng`) is unaffected by whether shuffling is
    enabled, since `rng` and `rng_shuffle` are independent generators.
"""

import numpy as np
import pytest

import xicsrt
from xicsrt.xicsrt_raytrace import _sort_raytrace


def _synthetic_single(num_rays, num_blocks, lost_fraction=0.9):
    """
    Build a synthetic single-iteration raytrace output with two elements,
    where rays are grouped into contiguous per-bundle blocks (as a plasma
    source would produce) and identified by a per-ray 'block' key.
    """
    counts = np.full(num_blocks, num_rays // num_blocks)
    block = np.repeat(np.arange(num_blocks), counts)
    n = block.size

    rng = np.random.default_rng(0)
    mask = rng.random(n) >= lost_fraction

    history = {
        'source': {'block': block.copy(), 'mask': np.ones(n, dtype=bool)},
        'detector': {'block': block.copy(), 'mask': mask},
    }
    single = {
        'config': {'general': {'random_seed': 1}},
        'meta': {},
        'image': {},
        'history': history,
    }
    return single


def _num_runs(x):
    """Number of contiguous constant-value runs in a 1-D array."""
    if x.size == 0:
        return 0
    return 1 + int(np.sum(np.diff(x) != 0))


def test_shuffle_breaks_up_bundle_blocks():
    """
    With shuffle=True the found-ray block order should no longer be
    contiguous; with shuffle=False it should remain exactly as emitted.
    """
    num_blocks = 200
    single = _synthetic_single(num_rays=200 * num_blocks, num_blocks=num_blocks)
    num_found_blocks = len(np.unique(single['history']['detector']['block']))

    out_unshuffled = _sort_raytrace(
        single, max_lost=100, rng=np.random.default_rng(1),
        shuffle=False, rng_shuffle=np.random.default_rng(2))
    block_unshuffled = out_unshuffled['found']['history']['detector']['block']
    assert _num_runs(block_unshuffled) == num_found_blocks

    out_shuffled = _sort_raytrace(
        single, max_lost=100, rng=np.random.default_rng(1),
        shuffle=True, rng_shuffle=np.random.default_rng(2))
    block_shuffled = out_shuffled['found']['history']['detector']['block']
    # A truly randomized order will have vastly more than num_found_blocks
    # runs (close to len(block_shuffled) for many small blocks).
    assert _num_runs(block_shuffled) > 5 * num_found_blocks


def test_shuffle_preserves_found_set():
    """
    Shuffling must be a pure permutation: the multiset of found rays (by
    block id) must be identical whether or not shuffling is enabled.
    """
    single = _synthetic_single(num_rays=20000, num_blocks=50)

    out_a = _sort_raytrace(
        single, max_lost=500, rng=np.random.default_rng(7),
        shuffle=False, rng_shuffle=np.random.default_rng(9))
    out_b = _sort_raytrace(
        single, max_lost=500, rng=np.random.default_rng(7),
        shuffle=True, rng_shuffle=np.random.default_rng(9))

    block_a = np.sort(out_a['found']['history']['detector']['block'])
    block_b = np.sort(out_b['found']['history']['detector']['block'])
    np.testing.assert_array_equal(block_a, block_b)


def test_shuffle_independent_of_lost_selection():
    """
    Turning shuffling on/off must not change which lost rays are retained,
    since `rng` (lost selection) and `rng_shuffle` are independent streams.
    """
    single = _synthetic_single(num_rays=20000, num_blocks=50)

    out_a = _sort_raytrace(
        single, max_lost=200, rng=np.random.default_rng(3),
        shuffle=False, rng_shuffle=np.random.default_rng(4))
    out_b = _sort_raytrace(
        single, max_lost=200, rng=np.random.default_rng(3),
        shuffle=True, rng_shuffle=np.random.default_rng(4))

    np.testing.assert_array_equal(
        out_a['lost']['history']['detector']['block'],
        out_b['lost']['history']['detector']['block'])


def test_shuffle_consistent_across_elements():
    """
    All elements must be reordered by the same permutation, so that ray
    correspondence between elements (e.g. source origin <-> detector hit)
    is preserved.
    """
    single = _synthetic_single(num_rays=20000, num_blocks=50)
    out = _sort_raytrace(
        single, max_lost=200, rng=np.random.default_rng(5),
        shuffle=True, rng_shuffle=np.random.default_rng(6))

    np.testing.assert_array_equal(
        out['found']['history']['source']['block'],
        out['found']['history']['detector']['block'])
    np.testing.assert_array_equal(
        out['lost']['history']['source']['block'],
        out['lost']['history']['detector']['block'])


def _plasma_config(shuffle_history, seed=42):
    """
    A point-bundle plasma scenario: each bundle has a unique origin, so
    per-ray bundle membership in the found history can be recovered exactly
    from the origin array.
    """
    return {
        'general': {
            'number_of_iter': 1,
            'number_of_runs': 1,
            'random_seed': seed,
            'shuffle_history': shuffle_history,
            'print_results': False,
            'save_config': False,
            'save_images': False,
            'save_results': False,
        },
        'sources': {'source': {
            'class_name': 'XicsrtPlasmaCubic',
            'origin': [0.0, 0.0, 0.0],
            'zaxis': [0.0, 0.0, 1.0],
            'xsize': 0.02, 'ysize': 0.02, 'zsize': 0.02,
            'target': [0.0, 0.0, 0.80374151],
            'spread': float(np.radians(5.0)),
            'temperature': 500.0,
            'emissivity': 5e18,
            'mass_number': 40.0,
            'wavelength': 3.9492,
            'time_resolution': 1e-6,
            'bundle_count': 500,
            'bundle_volume': 1e-7,
            'bundle_type': 'point',
            'use_poisson': True,
            'max_rays': int(1e7),
        }},
        'optics': {
            'crystal': {
                'class_name': 'XicsrtOpticSphericalCrystal',
                'origin': [0.0, 0.0, 0.80374151],
                'zaxis': [0.0, 0.59497864, -0.80374151],
                'xsize': 0.2, 'ysize': 0.2, 'radius': 1.0,
                'crystal_spacing': 2.45676,
                'rocking_type': 'gaussian',
                'rocking_fwhm': 1e-4,
            },
            'detector': {
                'class_name': 'XicsrtOpticDetector',
                'origin': [0.0, 0.76871290, 0.56904832],
                'zaxis': [0.0, -0.95641806, 0.29200084],
                'xsize': 0.4, 'ysize': 0.4,
            },
        },
    }


def _bundle_ids(origin):
    _, inv = np.unique(origin, axis=0, return_inverse=True)
    return inv


def test_end_to_end_plasma_history_order():
    """
    With a real plasma source, `shuffle_history=False` must reproduce the
    original bundle-blocked found-ray order; `shuffle_history=True` (the
    default) must not.
    """
    results_off = xicsrt.raytrace(_plasma_config(shuffle_history=False))
    origin_off = results_off['found']['history']['source']['origin']
    assert len(origin_off) > 100, 'test scenario should find a reasonable number of rays'
    ids_off = _bundle_ids(origin_off)
    num_bundles = len(np.unique(ids_off))
    assert _num_runs(ids_off) == num_bundles

    results_on = xicsrt.raytrace(_plasma_config(shuffle_history=True))
    origin_on = results_on['found']['history']['source']['origin']
    ids_on = _bundle_ids(origin_on)
    assert _num_runs(ids_on) > 5 * num_bundles

    # Pure permutation: identical found-ray set (by origin), same count.
    assert len(origin_on) == len(origin_off)
    np.testing.assert_array_equal(
        np.unique(origin_off, axis=0), np.unique(origin_on, axis=0))
