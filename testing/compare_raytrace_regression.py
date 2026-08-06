# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Opus 5).
"""
Temporary regression harness for F006 (devel/plans/plan_F006_raytrace_memory.md).

.. Authors
    Novimir Pablant <npablant@pppl.gov>

This module was AI generated using Claude (Opus 5).

Purpose
-------

Verify that the F006 raytrace memory / instrumentation changes do not perturb
the raytracing results. Two tiers are implemented, following the user decision
recorded in Finding 6 of the plan:

Tier A -- exact (bit-identical)
    Every `found` history array (all elements, all ray keys), every image, and
    every meta count must compare equal under `np.testing.assert_array_equal`
    (exact, not `allclose`).

Tier B -- statistical (lost rays only)
    The lost-ray subsample is allowed to change, but must remain correct: the
    right number of rays is retained, every retained ray is genuinely lost, no
    ray is retained twice, and the selection is uniform over the lost set
    (chi-square across repeated trials).

Every scenario runs with `number_of_iter = 3`. This is deliberate: it is the
global-RNG-divergence detector. A change that perturbs the global `np.random`
stream will still produce a matching iteration 1 but will diverge on
iterations 2 and 3. A single-iteration comparison gives a false pass.

This file is a development tool. It is not intended for `master`.

Self-test results (2026-07-28, measured)
----------------------------------------

The `selftest` subcommand mutates a captured payload. Separately, real source
mutations were injected into the working tree and checked end to end:

* A 1 ULP perturbation of one ray origin inside `Dispatcher.trace` is caught
  by Tier A (exit status 1) in the `history`, `runs` and `mp` variants.
* A large global-RNG stream shift (`np.random.random(1000)` inside
  `_sort_raytrace`) is caught by all of `history`, `runs` and `mp`.

Two non-obvious MT19937 properties were found while validating the above, and
both matter when interpreting a *passing* result:

1. `np.random.shuffle` draws a whole 32-bit buffer at a time, so the resulting
   stream position is quantized. Injecting a *small* number of extra global
   draws (1-3 `np.random.random()` calls) before the shuffle changes which
   lost rays are picked but leaves the post-shuffle state, and therefore every
   subsequent iteration, bit-identical. Such a perturbation is invisible to
   Tier A by construction. Only a shift large enough to cross a buffer refill
   boundary desynchronizes later iterations.
2. On the *global* `RandomState`, `np.random.choice(a, size, replace=False)`
   consumes exactly the same stream as `np.random.shuffle(arange(N))` and
   leaves an identical state. Swapping one for the other on the global stream
   is therefore already iteration-safe.

Neither property weakens the case for fix 4's dedicated `np.random.Generator`,
which sidesteps the whole question: it provably cannot touch the global
stream, so the found rays are bit-identical for any `max_lost` and any N.

Usage
-----

Create a baseline worktree once::

    git worktree add /tmp/xicsrt_baseline e89a3bd

Run the full comparison of the working tree against that baseline::

    python testing/compare_raytrace_regression.py run \\
        --baseline-src /tmp/xicsrt_baseline

Other subcommands::

    dump     --out FILE [--src PATH] [--variant NAME]
    compare  --baseline FILE --new FILE [--allow-added-ray-keys]
    tierb    [--src PATH]
    selftest [--src PATH]

`--src PATH` runs the trace in a subprocess with `PYTHONPATH` pointed at an
alternate xicsrt source tree (and a working directory outside the repository,
so that the repository copy is not picked up first). This is how the baseline
and the working tree are run against byte-identical inputs.

After a change that intentionally alters the global RNG stream (fix 4), the
old baseline can no longer be matched on iterations 2+. Re-baseline with::

    python testing/compare_raytrace_regression.py dump --out ref_fix4.pickle

and then compare later changes against that dump with `compare`.
"""

import argparse
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
from copy import deepcopy

import numpy as np

# The directory containing this file's repository, used to locate the harness
# itself when re-invoking under an alternate source tree.
_HARNESS_PATH = os.path.abspath(__file__)
_REPO_PATH = os.path.dirname(os.path.dirname(_HARNESS_PATH))


# ----------------------------------------------------------------------------
# Scenario configurations
# ----------------------------------------------------------------------------

def _base_config(number_of_iter=3):
    """
    A small plasma + spherical crystal + detector scenario.

    A plasma source is used rather than a simple source so that the bundle
    loop, the Poisson draws, and the concatenation path in
    `XicsrtPlasmaGeneric.create_sources` are all exercised. `bundle_count` is
    kept small (200) so that a full comparison runs in a few seconds.
    """
    config = {}
    config['general'] = {
        'number_of_iter': number_of_iter,
        'number_of_runs': 1,
        'random_seed': 12345,
        'keep_meta': True,
        'keep_images': True,
        'keep_history': True,
        'history_max_lost': 10000,
        # Disabled so that a Tier A exact comparison can be made against a
        # baseline source tree that predates the found-ray shuffle (F012).
        # `get_config` merges unrecognized 'general' keys non-strictly, so
        # this is silently ignored (and has no effect) on such a baseline.
        'shuffle_history': False,
        'print_results': False,
        'save_config': False,
        'save_images': False,
        'save_results': False,
    }
    config['sources'] = {
        'source': {
            'class_name': 'XicsrtPlasmaCubic',
            'origin': [0.0, 0.0, 0.0],
            'zaxis': [0.0, 0.0, 1.0],
            'xsize': 0.02,
            'ysize': 0.02,
            'zsize': 0.02,
            'target': [0.0, 0.0, 0.80374151],
            'spread': np.radians(2.0),
            'temperature': 500.0,
            'emissivity': 1e18,
            'mass_number': 40.0,
            'wavelength': 3.9492,
            'time_resolution': 1e-6,
            'bundle_count': 200,
            'bundle_volume': 1e-7,
            'max_rays': int(1e7),
        },
    }
    config['optics'] = {
        'crystal': {
            'class_name': 'XicsrtOpticSphericalCrystal',
            'origin': [0.0, 0.0, 0.80374151],
            'zaxis': [0.0, 0.59497864, -0.80374151],
            'xsize': 0.2,
            'ysize': 0.2,
            'radius': 1.0,
            'crystal_spacing': 2.45676,
            'rocking_type': 'gaussian',
            'rocking_fwhm': 1e-4,
        },
        'detector': {
            'class_name': 'XicsrtOpticDetector',
            'origin': [0.0, 0.76871290, 0.56904832],
            'zaxis': [0.0, -0.95641806, 0.29200084],
            'xsize': 0.4,
            'ysize': 0.4,
        },
    }
    return config


def get_variant_config(variant):
    """
    Return the config and driver name for a named scenario variant.
    """
    if variant == 'history':
        # The main Tier A scenario: full history, three iterations.
        return _base_config(), 'raytrace'

    if variant == 'nohistory':
        # Exercises the `len(history) == 0` guard paths that the production
        # run at issue in F006 actually hits.
        config = _base_config()
        config['general']['keep_history'] = False
        return config, 'raytrace'

    if variant == 'runs':
        # Multiple runs in a single process: exercises the `_internal`
        # max_lost path and combine_raytrace across runs.
        config = _base_config()
        config['general']['number_of_runs'] = 2
        return config, 'raytrace'

    if variant == 'mp':
        # Multiple runs through the multiprocessing pool.
        config = _base_config()
        config['general']['number_of_runs'] = 2
        return config, 'raytrace_mp'

    raise ValueError('Unknown variant: {}'.format(variant))


VARIANTS = ['history', 'nohistory', 'runs', 'mp']


# ----------------------------------------------------------------------------
# Result capture
# ----------------------------------------------------------------------------

def _extract(results):
    """
    Reduce a raytrace results dict to the comparable quantities.

    The config is deliberately excluded: it holds absolute paths and a few
    fields that the drivers rewrite, none of which are physics.
    """
    out = {'meta': {}, 'image': {}, 'found': {}, 'lost': {}}

    for key_opt, meta in results['total']['meta'].items():
        out['meta'][key_opt] = {kk: vv for kk, vv in meta.items()}

    for key_opt, image in results['total']['image'].items():
        out['image'][key_opt] = None if image is None else np.asarray(image)

    for group in ('found', 'lost'):
        for key_opt, hist in results[group]['history'].items():
            out[group][key_opt] = {kk: np.asarray(vv) for kk, vv in hist.items()}

    return out


def run_variant(variant):
    """
    Run a scenario variant in this process and return the extracted results.
    """
    import xicsrt

    config, driver = get_variant_config(variant)
    if driver == 'raytrace':
        results = xicsrt.raytrace(config)
    elif driver == 'raytrace_mp':
        results = xicsrt.raytrace_mp(config, processes=2)
    else:
        raise ValueError('Unknown driver: {}'.format(driver))

    return _extract(results)


def dump_variants(variants, filepath):
    """
    Run each variant in this process and pickle the results to `filepath`.
    """
    import xicsrt

    payload = {
        'xicsrt_path': os.path.dirname(os.path.abspath(xicsrt.__file__)),
        'variants': {},
    }
    for variant in variants:
        print('  running variant: {}'.format(variant), flush=True)
        payload['variants'][variant] = run_variant(variant)

    with open(filepath, 'wb') as ff:
        pickle.dump(payload, ff)

    return payload


def dump_variants_in_src(variants, filepath, src=None):
    """
    Run `dump_variants` in a subprocess against an alternate xicsrt tree.

    If `src` is None the current process is used directly. Otherwise a
    subprocess is spawned with `PYTHONPATH` set to `src` and a working
    directory outside the repository, so that `src` is the xicsrt that gets
    imported rather than the repository copy.
    """
    if src is None:
        return dump_variants(variants, filepath)

    src = os.path.abspath(src)
    env = dict(os.environ)
    env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')

    cmd = [
        sys.executable, _HARNESS_PATH, 'dump',
        '--out', os.path.abspath(filepath),
        '--variant', *variants,
    ]
    # The working directory must not be the repository, otherwise '' on
    # sys.path would shadow `src`.
    subprocess.run(cmd, env=env, cwd=tempfile.gettempdir(), check=True)

    with open(filepath, 'rb') as ff:
        return pickle.load(ff)


# ----------------------------------------------------------------------------
# Tier A -- exact comparison
# ----------------------------------------------------------------------------

def _compare_arrays(name, base, new, failures):
    base = np.asarray(base)
    new = np.asarray(new)

    if base.shape != new.shape:
        failures.append('{}: shape {} != {}'.format(name, base.shape, new.shape))
        return
    if base.dtype != new.dtype:
        failures.append('{}: dtype {} != {}'.format(name, base.dtype, new.dtype))
        return

    try:
        np.testing.assert_array_equal(base, new)
    except AssertionError:
        num_bad = int(np.count_nonzero(base != new))
        worst = 0.0
        if base.size and np.issubdtype(base.dtype, np.floating):
            worst = float(np.nanmax(np.abs(base - new)))
        failures.append(
            '{}: {} of {} elements differ (max abs diff {:g})'.format(
                name, num_bad, base.size, worst))


def compare_variant(base, new, variant, allow_added_ray_keys=False):
    """
    Tier A comparison of one variant. Returns a list of failure strings.
    """
    failures = []
    prefix = variant

    # Meta counts.
    if set(base['meta']) != set(new['meta']):
        failures.append('{}: meta element keys {} != {}'.format(
            prefix, sorted(base['meta']), sorted(new['meta'])))
    for key_opt in sorted(set(base['meta']) & set(new['meta'])):
        base_meta = base['meta'][key_opt]
        new_meta = new['meta'][key_opt]
        if set(base_meta) != set(new_meta):
            failures.append('{}: meta[{}] keys {} != {}'.format(
                prefix, key_opt, sorted(base_meta), sorted(new_meta)))
        for key in sorted(set(base_meta) & set(new_meta)):
            if base_meta[key] != new_meta[key]:
                failures.append('{}: meta[{}][{}] {} != {}'.format(
                    prefix, key_opt, key, base_meta[key], new_meta[key]))

    # Images.
    if set(base['image']) != set(new['image']):
        failures.append('{}: image keys {} != {}'.format(
            prefix, sorted(base['image']), sorted(new['image'])))
    for key_opt in sorted(set(base['image']) & set(new['image'])):
        base_img = base['image'][key_opt]
        new_img = new['image'][key_opt]
        if (base_img is None) != (new_img is None):
            failures.append('{}: image[{}] None mismatch ({} vs {})'.format(
                prefix, key_opt, base_img is None, new_img is None))
        elif base_img is not None:
            _compare_arrays(
                '{}: image[{}]'.format(prefix, key_opt),
                base_img, new_img, failures)

    # Found history: exact, every element, every ray key.
    if set(base['found']) != set(new['found']):
        failures.append('{}: found history elements {} != {}'.format(
            prefix, sorted(base['found']), sorted(new['found'])))
    for key_opt in sorted(set(base['found']) & set(new['found'])):
        base_hist = base['found'][key_opt]
        new_hist = new['found'][key_opt]

        missing = sorted(set(base_hist) - set(new_hist))
        added = sorted(set(new_hist) - set(base_hist))
        if missing:
            failures.append('{}: found[{}] missing ray keys {}'.format(
                prefix, key_opt, missing))
        if added and not allow_added_ray_keys:
            failures.append('{}: found[{}] unexpected new ray keys {}'.format(
                prefix, key_opt, added))
        elif added:
            print('  note: {}: found[{}] gained ray keys {}'.format(
                prefix, key_opt, added))

        for key_ray in sorted(set(base_hist) & set(new_hist)):
            _compare_arrays(
                '{}: found[{}][{}]'.format(prefix, key_opt, key_ray),
                base_hist[key_ray], new_hist[key_ray], failures)

    return failures


def check_lost_structure(new, variant):
    """
    Tier B, structural part: checks that can be made from the output alone.

    Every retained lost ray must be masked off at the final element, and the
    retained count must not exceed the number of found rays plus lost rays.
    """
    failures = []
    if not new['lost']:
        return failures

    key_opt_last = list(new['lost'].keys())[-1]
    mask = new['lost'][key_opt_last]['mask']
    num_bad = int(np.count_nonzero(mask))
    if num_bad > 0:
        failures.append(
            '{}: {} of {} retained lost rays are marked found at {}'.format(
                variant, num_bad, mask.size, key_opt_last))

    # All elements must retain the same number of lost rays.
    sizes = {kk: vv['mask'].size for kk, vv in new['lost'].items()}
    if len(set(sizes.values())) > 1:
        failures.append('{}: lost history lengths disagree: {}'.format(
            variant, sizes))

    return failures


def compare_payloads(base_payload, new_payload, allow_added_ray_keys=False):
    """
    Compare two dump payloads across all shared variants.
    """
    failures = []
    variants = [vv for vv in VARIANTS if vv in base_payload['variants']
                and vv in new_payload['variants']]
    if not variants:
        raise RuntimeError('No shared variants between the two dumps.')

    for variant in variants:
        base = base_payload['variants'][variant]
        new = new_payload['variants'][variant]
        failures.extend(compare_variant(
            base, new, variant, allow_added_ray_keys=allow_added_ray_keys))
        failures.extend(check_lost_structure(new, variant))

    return failures


# ----------------------------------------------------------------------------
# Tier B -- statistical lost-ray tests
# ----------------------------------------------------------------------------

def _synthetic_single(num_rays, lost_fraction, seed=0):
    """
    Build a minimal `_raytrace_iter`-shaped dict with a known lost set.

    A synthetic 'index' ray key is included so that the identity of each
    retained lost ray can be recovered from the output of `_sort_raytrace`.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(num_rays) >= lost_fraction

    history = {
        'source': {
            'index': np.arange(num_rays),
            'mask': np.ones(num_rays, dtype=bool),
        },
        'detector': {
            'index': np.arange(num_rays),
            'mask': mask,
        },
    }
    single = {
        'config': {'general': {'random_seed': 12345}},
        'meta': {},
        'image': {},
        'history': history,
    }
    return single, np.flatnonzero(~mask)


def _call_sort(sort_fn, single, max_lost, trial):
    """
    Call `_sort_raytrace`, handling both the old and new signatures.

    The baseline implementation draws from the global `np.random` stream; the
    new one takes a dedicated `np.random.Generator`. Each trial is given an
    independent seed either way.

    Where supported, the found-ray shuffle (F012) is explicitly disabled so
    that Tier B continues to test only what it was designed for: the
    lost-ray subsampling. The shuffle itself is covered by
    `tests/test_history_shuffle.py`.
    """
    params = inspect.signature(sort_fn).parameters
    if 'rng' in params:
        kwargs = dict(rng=np.random.default_rng(trial))
        if 'shuffle' in params:
            kwargs['shuffle'] = False
        return sort_fn(single, max_lost=max_lost, **kwargs)
    np.random.seed(trial)
    return sort_fn(single, max_lost=max_lost)


def run_tier_b(num_rays=20000, lost_fraction=0.99, max_lost=20, num_trials=2000):
    """
    Statistical validation of the lost-ray subsampling.

    Checks correctness (retained count, genuinely lost, no duplicates) on
    every trial, then a chi-square uniformity test over all trials.
    """
    from scipy import stats

    from xicsrt.xicsrt_raytrace import _sort_raytrace

    failures = []

    single, lost_index = _synthetic_single(num_rays, lost_fraction)
    num_lost = lost_index.size
    lost_set = set(lost_index.tolist())
    print('  synthetic rays: {}, lost: {}, retained per trial: {}'.format(
        num_rays, num_lost, max_lost))

    # Position of each lost ray within the lost set, for the histogram.
    rank_of_index = {vv: ii for ii, vv in enumerate(lost_index.tolist())}
    counts = np.zeros(num_lost, dtype=np.int64)

    for trial in range(num_trials):
        out = _call_sort(_sort_raytrace, deepcopy(single), max_lost, trial)
        selected = out['lost']['history']['detector']['index']

        if selected.size != min(max_lost, num_lost):
            failures.append('trial {}: retained {} rays, expected {}'.format(
                trial, selected.size, min(max_lost, num_lost)))
            break
        if np.unique(selected).size != selected.size:
            failures.append('trial {}: duplicate rays retained'.format(trial))
            break
        not_lost = [int(ii) for ii in selected if int(ii) not in lost_set]
        if not_lost:
            failures.append(
                'trial {}: retained rays that are not lost: {}'.format(
                    trial, not_lost[:10]))
            break
        if np.any(out['lost']['history']['detector']['mask']):
            failures.append('trial {}: retained ray has mask True'.format(trial))
            break

        # The two elements must be indexed consistently.
        selected_src = out['lost']['history']['source']['index']
        if not np.array_equal(selected, selected_src):
            failures.append(
                'trial {}: element histories are inconsistently indexed'.format(
                    trial))
            break

        for ii in selected:
            counts[rank_of_index[int(ii)]] += 1

    if failures:
        return failures

    expected = num_trials * max_lost / num_lost
    chi2, pvalue = stats.chisquare(counts)
    print('  chi2 = {:.1f}, dof = {}, expected/bin = {:.1f}, p = {:.4f}'.format(
        chi2, num_lost - 1, expected, pvalue))
    if pvalue < 1e-3:
        failures.append(
            'lost-ray selection is not uniform: chi2={:.1f} dof={} p={:.3g}'
            .format(chi2, num_lost - 1, pvalue))

    # Also verify that the found rays are unaffected and complete.
    out = _call_sort(_sort_raytrace, deepcopy(single), max_lost, 0)
    found = out['found']['history']['detector']['index']
    expected_found = np.flatnonzero(single['history']['detector']['mask'])
    if not np.array_equal(found, expected_found):
        failures.append('found rays are not the full, ordered found set')

    return failures


def run_tier_b_in_src(src=None):
    """
    Run Tier B, optionally against an alternate xicsrt source tree.
    """
    if src is None:
        return run_tier_b()

    src = os.path.abspath(src)
    env = dict(os.environ)
    env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable, _HARNESS_PATH, 'tierb']
    proc = subprocess.run(env=env, cwd=tempfile.gettempdir(), args=cmd)
    return [] if proc.returncode == 0 else ['Tier B failed in {}'.format(src)]


# ----------------------------------------------------------------------------
# Harness self-test
# ----------------------------------------------------------------------------

def run_selftest(payload):
    """
    Confirm that the comparator actually catches perturbations.

    An untested test harness is worthless. Each mutation below must produce at
    least one failure; the unmutated comparison must produce none.
    """
    results = []

    clean = compare_payloads(payload, deepcopy(payload))
    results.append(('unmutated compare is clean', len(clean) == 0, clean))

    # 1 ULP on a single found-ray coordinate.
    mutated = deepcopy(payload)
    arr = mutated['variants']['history']['found']['detector']['origin']
    original = arr[0, 0]
    arr[0, 0] = np.nextafter(original, np.inf)
    assert arr[0, 0] != original, 'nextafter did not change the value'
    print('  1 ULP mutation: {!r} -> {!r} (delta {:g})'.format(
        original, arr[0, 0], arr[0, 0] - original))
    failed = compare_payloads(payload, mutated)
    results.append(('1 ULP found-ray origin is detected', len(failed) > 0, failed))

    # 1 ULP on a wavelength.
    mutated = deepcopy(payload)
    arr = mutated['variants']['history']['found']['crystal']['wavelength']
    arr[0] = np.nextafter(arr[0], np.inf)
    failed = compare_payloads(payload, mutated)
    results.append(('1 ULP found-ray wavelength is detected', len(failed) > 0, failed))

    # A single flipped mask bit.
    mutated = deepcopy(payload)
    arr = mutated['variants']['history']['found']['crystal']['mask']
    arr[0] = ~arr[0]
    failed = compare_payloads(payload, mutated)
    results.append(('flipped mask bit is detected', len(failed) > 0, failed))

    # A single image count.
    mutated = deepcopy(payload)
    img = mutated['variants']['history']['image']['detector']
    idx = np.unravel_index(int(np.argmax(img)), img.shape)
    img[idx] += 1
    failed = compare_payloads(payload, mutated)
    results.append(('image count change is detected', len(failed) > 0, failed))

    # A single meta count.
    mutated = deepcopy(payload)
    mutated['variants']['history']['meta']['detector']['num_out'] += 1
    failed = compare_payloads(payload, mutated)
    results.append(('meta count change is detected', len(failed) > 0, failed))

    # A dropped ray key.
    mutated = deepcopy(payload)
    del mutated['variants']['history']['found']['detector']['wavelength']
    failed = compare_payloads(payload, mutated)
    results.append(('dropped ray key is detected', len(failed) > 0, failed))

    # An added ray key (must fail by default, pass when allowed). A name that
    # is not already part of the ray dict is required, otherwise this mutates
    # nothing; 'weight' is present as of F006 fix 7.
    mutated = deepcopy(payload)
    num = mutated['variants']['history']['found']['detector']['mask'].size
    for key_opt in mutated['variants']['history']['found']:
        assert 'selftest_added' not in mutated['variants']['history']['found'][key_opt]
        mutated['variants']['history']['found'][key_opt]['selftest_added'] = (
            np.ones(num))
    failed = compare_payloads(payload, mutated)
    results.append(('added ray key is detected', len(failed) > 0, failed))
    allowed = compare_payloads(payload, mutated, allow_added_ray_keys=True)
    results.append(('added ray key can be allowed', len(allowed) == 0, allowed))

    # A lost ray that is not actually lost.
    mutated = deepcopy(payload)
    mutated['variants']['history']['lost']['detector']['mask'][0] = True
    failed = check_lost_structure(mutated['variants']['history'], 'history')
    results.append(('non-lost ray in lost history is detected', len(failed) > 0,
                    failed))

    # Iterations 2+ divergence must be visible: perturbing only rays that
    # cannot be from iteration 1 still trips the comparator.
    mutated = deepcopy(payload)
    arr = mutated['variants']['history']['found']['detector']['origin']
    arr[-1, 2] = np.nextafter(arr[-1, 2], np.inf)
    failed = compare_payloads(payload, mutated)
    results.append(('1 ULP on the last found ray is detected', len(failed) > 0,
                    failed))

    ok = True
    for name, passed, detail in results:
        print('  [{}] {}'.format('PASS' if passed else 'FAIL', name))
        if not passed:
            ok = False
            for line in detail[:5]:
                print('        unexpected: {}'.format(line))
    return ok


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def _report(failures, label):
    if failures:
        print('')
        print('{}: FAIL ({} problems)'.format(label, len(failures)))
        for line in failures[:40]:
            print('  {}'.format(line))
        if len(failures) > 40:
            print('  ... and {} more'.format(len(failures) - 40))
        return 1
    print('')
    print('{}: PASS'.format(label))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    pp = sub.add_parser('dump', help='Run the scenarios and pickle the results.')
    pp.add_argument('--out', required=True)
    pp.add_argument('--src', default=None)
    pp.add_argument('--variant', nargs='+', default=VARIANTS, choices=VARIANTS)

    pp = sub.add_parser('compare', help='Compare two dumps.')
    pp.add_argument('--baseline', required=True)
    pp.add_argument('--new', required=True)
    pp.add_argument('--allow-added-ray-keys', action='store_true')

    pp = sub.add_parser('run', help='Dump both trees and compare.')
    pp.add_argument('--baseline-src', default=None)
    pp.add_argument('--baseline-dump', default=None)
    pp.add_argument('--new-src', default=None)
    pp.add_argument('--variant', nargs='+', default=VARIANTS, choices=VARIANTS)
    pp.add_argument('--allow-added-ray-keys', action='store_true')
    pp.add_argument('--save-new', default=None)
    pp.add_argument('--skip-tier-b', action='store_true')

    pp = sub.add_parser('tierb', help='Statistical lost-ray tests.')
    pp.add_argument('--src', default=None)

    pp = sub.add_parser('selftest', help='Verify that the harness detects changes.')
    pp.add_argument('--src', default=None)

    args = parser.parse_args()

    if args.command == 'dump':
        if args.src is None:
            dump_variants(args.variant, args.out)
        else:
            dump_variants_in_src(args.variant, args.out, src=args.src)
        print('wrote {}'.format(args.out))
        return 0

    if args.command == 'compare':
        with open(args.baseline, 'rb') as ff:
            base_payload = pickle.load(ff)
        with open(args.new, 'rb') as ff:
            new_payload = pickle.load(ff)
        failures = compare_payloads(
            base_payload, new_payload,
            allow_added_ray_keys=args.allow_added_ray_keys)
        return _report(failures, 'Tier A')

    if args.command == 'tierb':
        failures = run_tier_b()
        return _report(failures, 'Tier B')

    if args.command == 'selftest':
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'selftest.pickle')
            print('generating a reference dump for the self-test')
            payload = dump_variants_in_src(['history'], path, src=args.src)
            print('running self-test mutations')
            ok = run_selftest(payload)
        print('')
        print('harness self-test: {}'.format('PASS' if ok else 'FAIL'))
        return 0 if ok else 1

    if args.command == 'run':
        if args.baseline_src is None and args.baseline_dump is None:
            parser.error('run requires --baseline-src or --baseline-dump')

        with tempfile.TemporaryDirectory() as tmp:
            if args.baseline_dump is not None:
                with open(args.baseline_dump, 'rb') as ff:
                    base_payload = pickle.load(ff)
                print('baseline dump: {}'.format(args.baseline_dump))
            else:
                print('running baseline: {}'.format(args.baseline_src))
                base_payload = dump_variants_in_src(
                    args.variant, os.path.join(tmp, 'base.pickle'),
                    src=args.baseline_src)

            new_path = args.save_new or os.path.join(tmp, 'new.pickle')
            print('running new: {}'.format(args.new_src or _REPO_PATH))
            new_payload = dump_variants_in_src(
                args.variant, new_path, src=args.new_src)

            failures = compare_payloads(
                base_payload, new_payload,
                allow_added_ray_keys=args.allow_added_ray_keys)
            status = _report(failures, 'Tier A')

            if not args.skip_tier_b:
                print('')
                print('running Tier B')
                tier_b = run_tier_b()
                status |= _report(tier_b, 'Tier B')

        return status

    parser.error('unhandled command')


if __name__ == '__main__':
    sys.exit(main())
