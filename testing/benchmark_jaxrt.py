# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Benchmark the jaxrt engine against the numpy engine.

Runs the same spherical-crystal spectrometer scenario through both
engines for a range of ray counts and reports wall-clock times. The
first jaxrt iteration includes jit compilation; steady-state per-ray
throughput is what matters for large runs, so the compile time is
reported separately.

Usage:

    python testing/benchmark_jaxrt.py

This script was AI generated using Claude (Fable 5).
"""

import time

import numpy as np

import xicsrt
import xicsrt.jaxrt


def make_config(intensity, num_iter):
    config = {
        'general': {
            'number_of_iter': num_iter,
            'print_results': False,
            'random_seed': 0,
            'keep_history': False,
            'keep_images': True,
        },
        'sources': {
            'source': {
                'class_name': 'XicsrtSourceFocused',
                'intensity': intensity,
                'wavelength': 3.9492,
                'temperature': 1000.0,
                'mass_number': 40.0,
                'linewidth': 1.129e+14,
                'spread': np.radians(2.0),
                'target': [0.0, 0.0, 0.80374151],
            },
        },
        'optics': {
            'crystal': {
                'class_name': 'XicsrtOpticSphericalCrystal',
                'origin': [0.0, 0.0, 0.80374151],
                'zaxis': [0.0, 0.59497864, -0.80374151],
                'xsize': 0.2,
                'ysize': 0.2,
                'radius': 1.0,
                'crystal_spacing': 2.45676,
                'rocking_type': 'gaussian',
                'rocking_fwhm': 48.070e-6,
            },
            'detector': {
                'class_name': 'XicsrtOpticDetector',
                'origin': [0.0, 0.76871290, 0.56904832],
                'zaxis': [0.0, -0.95641806, 0.29200084],
                'xsize': 0.4,
                'ysize': 0.2,
            },
        },
    }
    return config


def benchmark(raytrace, intensity, num_iter):
    config = make_config(intensity, num_iter)
    start = time.perf_counter()
    results = raytrace(config)
    elapsed = time.perf_counter() - start
    num_detected = results['total']['meta']['detector']['num_out']
    return elapsed, num_detected


def main():
    import logging
    logging.disable(logging.INFO)

    num_iter = 5
    print(f'{"rays/iter":>12s} {"numpy [s]":>12s} {"jaxrt [s]":>12s} '
          f'{"jaxrt warm [s]":>15s} {"speedup":>8s}')

    for intensity in [1e4, 1e5, 1e6]:
        t_np, n_np = benchmark(xicsrt.raytrace, intensity, num_iter)
        # First call includes jit compile time.
        t_jx_cold, n_jx = benchmark(xicsrt.jaxrt.raytrace, intensity, num_iter)
        # Second call reuses the persistent jax compilation cache only
        # within a process if shapes match; recompilation happens per
        # call here, so run a two-pass timing within one call instead.
        t_jx_warm, _ = benchmark(xicsrt.jaxrt.raytrace, intensity, num_iter)

        print(f'{intensity:12.0e} {t_np:12.2f} {t_jx_cold:12.2f} '
              f'{t_jx_warm:15.2f} {t_np / t_jx_warm:8.1f}'
              f'   (detected: numpy {n_np}, jaxrt {n_jx})')


if __name__ == '__main__':
    main()
