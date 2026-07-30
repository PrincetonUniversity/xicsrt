# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
F010 verification: statistical comparison of the vectorized plasma
create_sources (new) against the per-bundle source loop (baseline).

Runs the same small plasma scenario in subprocesses (one per source tree
per seed) and compares generated/detected counts and detector image
moments across seeds. The two engines are statistically identical but not
bit-identical, so the comparison is a 5-sigma test on aggregate counts.

Usage:
    python testing/compare_f010_plasma.py --baseline-src PATH [--num-seeds N]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_HARNESS_PATH = os.path.abspath(__file__)
_REPO_PATH = os.path.dirname(os.path.dirname(_HARNESS_PATH))


def _config(seed):
    return {
        'general': {
            'number_of_iter': 5,
            'number_of_runs': 1,
            'random_seed': seed,
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
            'bundle_type': 'voxel',
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


def _run_child(seed):
    """Child-process entry: run one raytrace, print a JSON summary."""
    import xicsrt
    results = xicsrt.raytrace(_config(seed))
    image = np.asarray(results['total']['image']['detector'], dtype=float)
    total = image.sum()
    out = {
        'generated': int(results['total']['meta']['source']['num_out']),
        'detected': int(results['total']['meta']['detector']['num_out']),
    }
    if total > 0:
        ix = np.arange(image.shape[0])
        iy = np.arange(image.shape[1])
        out['cx'] = float((image.sum(axis=1) * ix).sum() / total)
        out['cy'] = float((image.sum(axis=0) * iy).sum() / total)
    print('F010JSON ' + json.dumps(out))


def _run_in_src(src, seed):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.abspath(src) + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable, _HARNESS_PATH, 'child', '--seed', str(seed)]
    proc = subprocess.run(cmd, env=env, cwd=tempfile.gettempdir(),
                          capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if line.startswith('F010JSON '):
            return json.loads(line[len('F010JSON '):])
    raise RuntimeError('No summary line from child:\n' + proc.stdout + proc.stderr)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    pp = sub.add_parser('child')
    pp.add_argument('--seed', type=int, required=True)

    pp = sub.add_parser('run')
    pp.add_argument('--baseline-src', required=True)
    pp.add_argument('--num-seeds', type=int, default=5)

    args = parser.parse_args()

    if args.command == 'child':
        _run_child(args.seed)
        return 0

    seeds = [1000 + ii for ii in range(args.num_seeds)]
    base, new = [], []
    for seed in seeds:
        print(f'seed {seed}: baseline ...', flush=True)
        base.append(_run_in_src(args.baseline_src, seed))
        print(f'seed {seed}: new ...', flush=True)
        new.append(_run_in_src(_REPO_PATH, seed))

    failures = []
    for key in ('generated', 'detected'):
        bb = np.array([rr[key] for rr in base], dtype=float)
        nn = np.array([rr[key] for rr in new], dtype=float)
        # Poisson-dominated counts: 5-sigma comparison of totals.
        tot_b, tot_n = bb.sum(), nn.sum()
        sigma = np.sqrt(tot_b + tot_n)
        diff = abs(tot_b - tot_n)
        print(f'{key:>10}: baseline {tot_b:.0f}, new {tot_n:.0f}, '
              f'diff {diff:.0f} ({diff/sigma:.2f} sigma)')
        if diff > 5 * sigma:
            failures.append(f'{key}: differs by {diff/sigma:.1f} sigma')

    cx_b = np.mean([rr['cx'] for rr in base if 'cx' in rr])
    cx_n = np.mean([rr['cx'] for rr in new if 'cx' in rr])
    cy_b = np.mean([rr['cy'] for rr in base if 'cy' in rr])
    cy_n = np.mean([rr['cy'] for rr in new if 'cy' in rr])
    print(f'centroid x: baseline {cx_b:.2f}, new {cx_n:.2f} (pixels)')
    print(f'centroid y: baseline {cy_b:.2f}, new {cy_n:.2f} (pixels)')
    if abs(cx_b - cx_n) > 2.0 or abs(cy_b - cy_n) > 2.0:
        failures.append('image centroid moved by more than 2 pixels')

    if failures:
        print('\nFAIL')
        for ff in failures:
            print(' ', ff)
        return 1
    print('\nPASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
