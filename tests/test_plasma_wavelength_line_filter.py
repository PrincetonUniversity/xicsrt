# -*- coding: utf-8 -*-
"""
Tests for the `wavelength_line_range` line filter on `XicsrtPlasmaGeneric`.

This file includes AI generated code using Claude (Sonnet 5).

`wavelength_line_range` filters the *list of lines* consumed by the
`multi_voigt` wavelength distribution (as returned by
`get_line_parameters`), not the sampled Voigt distribution of any
individual line. These tests cover:
  * default (`None`) includes every configured line,
  * a range that excludes some lines removes exactly those lines from the
    sampled wavelengths (while an included line's own Voigt tail is left
    untouched),
  * a range that excludes every line raises a clear `ValueError`.
"""

import numpy as np

import pytest

from xicsrt.objects._Dispatcher import Dispatcher
from xicsrt import xicsrt_config

LINE_LOCATIONS = np.array([3.94, 3.95, 3.96])
LINE_INTENSITIES = np.array([1.0, 1.0, 1.0])
LINE_SIGMAS = np.array([1e-4, 1e-4, 1e-4])
LINE_GAMMAS = np.array([0.0, 0.0, 0.0])


def _scenario_config(seed=1, bundle_count=400, **extra):
    """A small cubic plasma emitting a fixed three-line multi_voigt spectrum."""
    source = dict(
        class_name='XicsrtPlasmaCubic',
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
        wavelength_dist='multi_voigt',
        line_locations=LINE_LOCATIONS,
        line_intensities=LINE_INTENSITIES,
        line_sigmas=LINE_SIGMAS,
        line_gammas=LINE_GAMMAS,
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


def _line_of(wavelength):
    """Nearest configured line center for each sampled wavelength."""
    return LINE_LOCATIONS[np.argmin(
        np.abs(wavelength[:, None] - LINE_LOCATIONS[None, :]), axis=1)]


def test_default_includes_all_lines():
    """With `wavelength_line_range = None` every configured line is sampled."""
    rays = _generate(_scenario_config())
    lines_seen = np.unique(_line_of(rays['wavelength']))
    np.testing.assert_allclose(lines_seen, LINE_LOCATIONS)


def test_range_excludes_lines_outside_it():
    """A range covering only the first two lines removes the third."""
    rays = _generate(_scenario_config(wavelength_line_range=[3.93, 3.955]))
    lines_seen = np.unique(_line_of(rays['wavelength']))
    np.testing.assert_allclose(lines_seen, LINE_LOCATIONS[:2])
    assert np.max(rays['wavelength']) < 3.96


def test_range_does_not_truncate_included_line_tails():
    """
    A kept line's own Voigt tail must not be truncated by the range: with
    a wide Lorentzian tail on a kept line, samples may fall outside the
    range bounds even though the line center is inside them.
    """
    config = _scenario_config(
        wavelength_line_range=[3.939, 3.941],
        line_locations=np.array([3.94]),
        line_intensities=np.array([1.0]),
        line_sigmas=np.array([0.0]),
        line_gammas=np.array([0.01]),
    )
    rays = _generate(config)
    # A gamma=0.01 Cauchy tail on a single line easily produces samples
    # outside the [3.939, 3.941] line-selection range.
    assert np.any(rays['wavelength'] < 3.939) or np.any(rays['wavelength'] > 3.941)


def test_range_excluding_all_lines_raises():
    with pytest.raises(ValueError, match='excludes all lines'):
        _generate(_scenario_config(wavelength_line_range=[10.0, 11.0]))


# ---------------------------------------------------------------------------
# F035: rays['label'] for wavelength_dist='multi_voigt' (batched plasma path)
# ---------------------------------------------------------------------------

def test_label_present_and_matches_line():
    """
    `rays['label']` must be present, match the nearest configured line
    index for every ray, and cover every configured line at this sample
    size.
    """
    rays = _generate(_scenario_config())
    assert 'label' in rays
    assert rays['label'].shape == rays['wavelength'].shape
    assert np.issubdtype(rays['label'].dtype, np.integer)

    expected_label = np.argmin(
        np.abs(rays['wavelength'][:, None] - LINE_LOCATIONS[None, :]), axis=1)
    np.testing.assert_array_equal(rays['label'], expected_label)
    assert set(np.unique(rays['label'])) == {0, 1, 2}


def test_label_reindexed_after_line_filter():
    """
    When `wavelength_line_range` drops a line, `label` must index into the
    *filtered* line list (0/1 for the two kept lines), not the original
    three-line list.
    """
    rays = _generate(_scenario_config(wavelength_line_range=[3.93, 3.955]))
    kept_locations = LINE_LOCATIONS[:2]

    assert set(np.unique(rays['label'])) <= {0, 1}
    expected_label = np.argmin(
        np.abs(rays['wavelength'][:, None] - kept_locations[None, :]), axis=1)
    np.testing.assert_array_equal(rays['label'], expected_label)


def test_label_absent_for_other_wavelength_dist():
    """`label` must not be set for a non-multi_voigt wavelength_dist."""
    rays = _generate(_scenario_config(
        wavelength_dist='monochrome', wavelength=3.95))
    assert 'label' not in rays
