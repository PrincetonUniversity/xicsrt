# -*- coding: utf-8 -*-
"""
Tests for XicsrtPlasmaBundleSource.

This file includes AI generated code using Claude (Sonnet 5).

These tests cover:
  * equivalence of XicsrtPlasmaBundleSource (using the default
    XicsrtSourceFocused per bundle) against XicsrtPlasmaCubic, which models
    the identical physical scenario with the vectorized
    XicsrtPlasmaGeneric.create_sources: exact ray counts in the
    deterministic case, counting-noise agreement with Poisson statistics,
    KS agreement of the per-ray distributions, matching ray dict
    structure, and matching behavior for bundle masking, spread_radius and
    bundle_type,
  * coupling an arbitrary user ray source (one that defines none of the
    standard plasma options and adds a novel option of its own) to the
    bundle framework from a plugin directory (general.pathlist), which
    exercises the obj.param['pathlist'] plumbing added in the Dispatcher,
  * dispatching an alternate built-in source class with extra options
    supplied through bundle_source_config,
  * error paths: unknown bundle_source_class, wavelength_dist='multi_voigt',
    and angular_dist values with no known solid angle formula.
"""

import numpy as np
import pytest
from scipy import stats

from xicsrt.objects._Dispatcher import Dispatcher
from xicsrt import xicsrt_config


def _scenario_config(class_name, seed=1, bundle_count=400, **extra):
    """
    A small plasma cube scenario shared by both plasma classes, so that
    XicsrtPlasmaCubic and XicsrtPlasmaBundleSource can be compared directly.
    """
    source = dict(
        class_name=class_name,
        origin=[0.0, 0.0, 0.0],
        zaxis=[0.0, 0.0, 1.0],
        xsize=0.02,
        ysize=0.02,
        zsize=0.02,
        target=[0.0, 0.0, 0.80374151],
        spread=np.radians(2.0),
        temperature=500.0,
        emissivity=1e18,
        mass_number=40.0,
        wavelength=3.9492,
        time_resolution=1e-6,
        bundle_count=bundle_count,
        bundle_volume=1e-7,
        max_rays=int(1e7),
    )
    source.update(extra)

    config = {
        'general': {'random_seed': seed},
        'sources': {'source': source},
    }
    return xicsrt_config.get_config(config)


def _generate(config):
    sources = Dispatcher(config, 'sources')
    sources.instantiate()
    sources.setup()
    sources.check_param()
    sources.initialize()
    return sources.generate_rays()


def test_deterministic_ray_count_matches_plasma_cubic():
    """
    With `use_poisson = False` (the default) the per-bundle photon count is
    truncated to an integer, so both classes emit an identical, fully
    deterministic number of rays. This pins the shared intensity
    normalization exactly, with no statistical slack.
    """
    n_cubic = len(_generate(_scenario_config('XicsrtPlasmaCubic'))['mask'])
    n_bundle = len(_generate(_scenario_config('XicsrtPlasmaBundleSource'))['mask'])
    assert n_bundle == n_cubic


def test_poisson_ray_count_matches_plasma_cubic():
    """
    With Poisson statistics enabled the two classes should agree to within
    counting noise.

    Note that `use_poisson = True` is essential here: with the default
    `use_poisson = False` the counts are deterministic and identical, so
    this comparison would not actually exercise the sampling statistics.
    """
    n_seeds = 12
    n_found = {'XicsrtPlasmaCubic': [], 'XicsrtPlasmaBundleSource': []}

    for seed in range(n_seeds):
        for class_name in n_found:
            config = _scenario_config(class_name, seed=seed, use_poisson=True)
            n_found[class_name].append(len(_generate(config)['mask']))

    mean_cubic = np.mean(n_found['XicsrtPlasmaCubic'])
    mean_bundle = np.mean(n_found['XicsrtPlasmaBundleSource'])

    # Standard error on the difference of two Poisson sample means.
    combined_sigma = np.sqrt(mean_cubic / n_seeds + mean_bundle / n_seeds)
    assert abs(mean_cubic - mean_bundle) < 5 * combined_sigma


def test_ray_distributions_match_plasma_cubic():
    """
    The per-ray distributions (origin, direction, wavelength) produced by
    the per-bundle source loop should be drawn from the same underlying
    distributions as the vectorized implementation. Compared with a
    two-sample KS test rather than by comparing summary statistics, so that
    a difference in distribution shape is also caught.
    """
    rays_cubic = _generate(
        _scenario_config('XicsrtPlasmaCubic', bundle_count=2000))
    rays_bundle = _generate(
        _scenario_config('XicsrtPlasmaBundleSource', bundle_count=2000))

    for key in ('origin', 'direction'):
        for axis in range(3):
            result = stats.ks_2samp(
                rays_cubic[key][:, axis], rays_bundle[key][:, axis])
            assert result.pvalue > 0.001, f'{key}[{axis}] p={result.pvalue}'

    result = stats.ks_2samp(rays_cubic['wavelength'], rays_bundle['wavelength'])
    assert result.pvalue > 0.001, f'wavelength p={result.pvalue}'


def test_ray_dict_structure_matches_plasma_cubic():
    """The concatenated ray dict must carry the same keys and dtypes as the
    vectorized implementation, including 'weight'."""
    rays_cubic = _generate(_scenario_config('XicsrtPlasmaCubic', bundle_count=50))
    rays_bundle = _generate(
        _scenario_config('XicsrtPlasmaBundleSource', bundle_count=50))

    assert set(rays_bundle.keys()) == set(rays_cubic.keys())
    for key in rays_cubic:
        assert rays_bundle[key].dtype == rays_cubic[key].dtype, key


def test_bundle_mask_is_honored():
    """Bundles masked out by a bundle filter must not emit any rays."""

    class _MaskAlternate:
        def filter(self, bundle_input):
            bundle_input['mask'][::2] = False
            return bundle_input

    counts = {}
    for class_name in ('XicsrtPlasmaCubic', 'XicsrtPlasmaBundleSource'):
        config = _scenario_config(class_name)
        sources = Dispatcher(config, 'sources')
        sources.instantiate()
        sources.setup()
        sources.check_param()
        sources.initialize()
        sources.objects['source'].filter_objects.append(_MaskAlternate())
        counts[class_name] = len(sources.generate_rays()['mask'])

    assert counts['XicsrtPlasmaBundleSource'] == counts['XicsrtPlasmaCubic']


def test_point_bundle_type_matches_plasma_cubic():
    """
    With `bundle_type = 'point'` every ray of a bundle starts exactly at the
    bundle origin, and the ray count is independent of bundle position, so
    both implementations agree exactly.
    """
    extra = {'bundle_type': 'point'}
    rays_cubic = _generate(_scenario_config('XicsrtPlasmaCubic', **extra))
    rays_bundle = _generate(_scenario_config('XicsrtPlasmaBundleSource', **extra))

    assert len(rays_bundle['mask']) == len(rays_cubic['mask'])

    # Rays collapse onto the bundle centers rather than filling a voxel.
    bundle_count = 400
    assert len(np.unique(rays_bundle['origin'], axis=0)) == bundle_count
    assert len(np.unique(rays_cubic['origin'], axis=0)) == bundle_count


def test_spread_radius_matches_plasma_cubic():
    """
    With `spread_radius` the emission cone (and therefore the photon count)
    of each bundle depends on that bundle's distance to the target, so the
    total ray count depends on the randomly drawn bundle positions.

    The two implementations consume the random stream in a different order,
    so they are statistically equivalent but not bit-identical (the same
    property recorded for F010 phase 1b). Compare the mean over several
    seeds rather than a single realization.
    """
    extra = {'spread_radius': 0.05, 'spread': None}
    n_seeds = 6
    n_found = {'XicsrtPlasmaCubic': [], 'XicsrtPlasmaBundleSource': []}

    for seed in range(n_seeds):
        for class_name in n_found:
            config = _scenario_config(class_name, seed=seed, **extra)
            n_found[class_name].append(len(_generate(config)['mask']))

    mean_cubic = np.mean(n_found['XicsrtPlasmaCubic'])
    mean_bundle = np.mean(n_found['XicsrtPlasmaBundleSource'])

    # The spread of the total count across seeds is far smaller than
    # sqrt(N), since only the bundle positions vary; compare against the
    # observed seed-to-seed scatter instead.
    scatter = np.std(n_found['XicsrtPlasmaCubic']) / np.sqrt(n_seeds)
    assert abs(mean_cubic - mean_bundle) < 5 * max(scatter, 1.0)


def test_velocity_produces_doppler_shift():
    """
    The per-bundle `velocity` must reach the ray source and produce a
    Doppler shift.

    Note that XicsrtPlasmaCubic cannot be used as the comparison here: it
    never copies the configured velocity into bundle_input, so it silently
    produces no shift at all (tracked as F011). This test therefore checks
    the shift against the analytic value instead.
    """
    speed = 3e6
    wavelength = 3.9492
    common = dict(
        temperature=0.0,
        wavelength=wavelength,
        wavelength_dist='monochrome',
        bundle_count=200,
    )

    rays_static = _generate(
        _scenario_config('XicsrtPlasmaBundleSource', **common))
    rays_moving = _generate(
        _scenario_config('XicsrtPlasmaBundleSource',
                         velocity=[0.0, 0.0, speed], **common))

    assert np.allclose(rays_static['wavelength'], wavelength)

    # Rays are aimed at a target on the +z axis and the plasma moves along
    # +z, so the emission is blue shifted by very nearly v/c.
    c = 299792458.0
    expected = wavelength * (1.0 - speed / c)
    assert np.mean(rays_moving['wavelength']) == pytest.approx(
        expected, rel=1e-3)


def test_bundle_source_produces_rays():
    """Smoke test: rays are produced with the expected structure."""
    config = _scenario_config('XicsrtPlasmaBundleSource', bundle_count=50)
    rays = _generate(config)
    assert len(rays['mask']) > 0
    assert rays['origin'].shape == (len(rays['mask']), 3)
    assert rays['direction'].shape == (len(rays['mask']), 3)
    assert np.all(rays['mask'])


def test_bundle_source_class_directed():
    """An alternate built-in source class can be dispatched, with extra
    options supplied through bundle_source_config."""
    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=50,
        bundle_source_class='XicsrtSourceDirected',
        bundle_source_config={'direction': [0.0, 0.0, 1.0]},
    )
    rays = _generate(config)
    assert len(rays['mask']) > 0

    # A directed source aims every bundle along the same fixed direction,
    # rather than focusing on a target, so all directions should be
    # clustered around [0, 0, 1] rather than pointing in varied directions
    # towards a common focus.
    mean_dir = np.mean(rays['direction'], axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    assert mean_dir == pytest.approx([0.0, 0.0, 1.0], abs=0.05)


def test_bundle_source_class_from_plugin_path(tmp_path):
    """
    A user-defined ray source class living in a plugin directory
    (general.pathlist) can be used as bundle_source_class. This exercises
    the obj.param['pathlist'] plumbing added to the Dispatcher.
    """
    plugin_file = tmp_path / '_XicsrtSourceBundleTest.py'
    plugin_file.write_text(
        "from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric\n"
        "\n"
        "class XicsrtSourceBundleTest(XicsrtSourceGeneric):\n"
        "    pass\n"
    )

    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=50,
        bundle_source_class='XicsrtSourceBundleTest',
    )
    config['general']['pathlist'] = [str(tmp_path)]

    rays = _generate(config)
    assert len(rays['mask']) > 0


# A deliberately unusual ray source: it defines neither 'target' nor
# 'spread' nor 'linewidth' (so the forwarded plasma options must be
# silently dropped rather than raising), and it adds a novel option of its
# own that can only reach it through `bundle_source_config`. This is the
# capability that XicsrtPlasmaBundleSource exists to provide.
_WEIRD_SOURCE = '''
import numpy as np
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._GeometryObject import GeometryObject

@dochelper
class XicsrtSourceWeird(GeometryObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_objects = []

    def default_config(self):
        config = super().default_config()
        config['intensity'] = 0.0
        config['use_poisson'] = False
        config['xsize'] = 0.0
        config['ysize'] = 0.0
        config['zsize'] = 0.0
        config['wavelength'] = 1.0
        config['wavelength_scale'] = 1.0
        return config

    def initialize(self):
        super().initialize()
        if self.param['use_poisson']:
            self.param['intensity'] = np.random.poisson(self.param['intensity'])
        self.param['intensity'] = int(self.param['intensity'])

    def generate_rays(self):
        num = self.param['intensity']
        rays = {}
        rays['origin'] = np.tile(self.param['origin'], (num, 1))
        direction = np.zeros((num, 3))
        direction[:, 2] = 1.0
        rays['direction'] = direction
        rays['wavelength'] = np.full(
            num, self.param['wavelength'] * self.param['wavelength_scale'])
        rays['weight'] = np.ones(num)
        rays['mask'] = np.ones(num, dtype=bool)
        return rays
'''


def test_arbitrary_user_source_with_novel_options(tmp_path):
    """
    A user source that does not define the standard plasma options, and
    that has a novel option of its own, can be coupled to the bundle
    framework. Unsupported plasma options must be dropped silently
    (strict=False forwarding) rather than raising.
    """
    (tmp_path / '_XicsrtSourceWeird.py').write_text(_WEIRD_SOURCE)

    bundle_count = 50
    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=bundle_count,
        wavelength=3.0,
        bundle_source_class='XicsrtSourceWeird',
        bundle_source_config={'wavelength_scale': 2.0},
    )
    config['general']['pathlist'] = [str(tmp_path)]

    rays = _generate(config)

    # The novel option reached the source through bundle_source_config.
    assert np.allclose(rays['wavelength'], 6.0)

    # Every bundle emitted from its own origin, so the plasma still drives
    # the bundle geometry even for a source it knows nothing about.
    assert len(np.unique(rays['origin'], axis=0)) == bundle_count


def test_unknown_bundle_source_class_raises():
    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=50,
        bundle_source_class='XicsrtSourceDoesNotExist',
    )
    with pytest.raises(Exception, match='Could not find'):
        _generate(config)


def test_multi_voigt_not_supported():
    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=50,
        wavelength_dist='multi_voigt',
    )
    with pytest.raises(NotImplementedError):
        _generate(config)


def test_flat_angular_dist_has_no_solid_angle():
    """
    Unlike XicsrtPlasmaGeneric (which always assumes isotropic emission),
    XicsrtPlasmaBundleSource looks up the solid angle formula matching
    angular_dist and must raise rather than silently using the wrong
    normalization for a distribution with no known formula.
    """
    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=50,
        angular_dist='flat',
    )
    with pytest.raises(Exception, match='Solid angle'):
        _generate(config)


def test_spread_radius_requires_isotropic():
    config = _scenario_config(
        'XicsrtPlasmaBundleSource',
        bundle_count=50,
        angular_dist='isotropic_xy',
        spread_radius=0.01,
    )
    with pytest.raises(NotImplementedError):
        _generate(config)
