# -*- coding: utf-8 -*-
"""
Tests for the random spline-profile generators in xicsrt.tools.xicsrt_spline.

This file includes AI generated code using Claude (Opus 5, Fable 5).

These tests cover:
  * knot spacing and endpoint constraints,
  * mask lengths matching n_knots for any n_knots (regression for the
    hardcoded length-5 literals),
  * physical unit ranges (temperature in eV, velocity in m/s),
  * emissivity normalization (peak of exactly 1.0),
  * spline_from_profile behavior including zero core derivative and NaN
    outside the knot range,
  * JSON round trip of the profile dictionaries through
    xicsrt_io.save_config/load_config,
  * reproducibility with a fixed seed.
"""

import numpy as np
import pytest

from xicsrt.tools import xicsrt_spline

ALL_GENERATORS = [
    xicsrt_spline.generate_random_emissivity,
    xicsrt_spline.generate_random_temp,
    xicsrt_spline.generate_random_ion_temp,
    xicsrt_spline.generate_random_electron_temp,
    xicsrt_spline.generate_random_perpendicular_velocity,
    xicsrt_spline.generate_random_parallel_velocity,
]


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_profile_structure(generator):
    """Every generator returns a well-formed profile dictionary."""
    profile = generator(seed=0)

    n_knots = len(profile["x_knots"])
    assert len(profile["y_knots"]) == n_knots
    assert len(profile["deriv_zero"]) == n_knots
    assert len(profile["x_free"]) == n_knots
    assert len(profile["y_free"]) == n_knots
    assert isinstance(profile["profile_type"], str)

    x_knots = np.asarray(profile["x_knots"])
    assert x_knots[0] == 0.0
    assert x_knots[-1] == 1.0
    assert np.all(np.diff(x_knots) > 0.0)


def test_profile_json_round_trip(tmp_path):
    """Profile dictionaries survive the real config save/load path.

    Serialization is handled by xicsrt_io: numpy arrays are converted to
    lists on save (_convert_from_numpy) and back to arrays on load
    (_convert_to_numpy). This exercises the full save_config/load_config
    round trip with the profiles embedded in a config dict.
    """
    from xicsrt import xicsrt_io

    profiles = {
        'profile_emissivity': xicsrt_spline.generate_random_emissivity(seed=1),
        'profile_ion_temp': xicsrt_spline.generate_random_ion_temp(seed=2),
        'profile_electron_temp':
            xicsrt_spline.generate_random_electron_temp(seed=3),
        'profile_perpendicular_velocity':
            xicsrt_spline.generate_random_perpendicular_velocity(seed=4),
        'profile_parallel_velocity':
            xicsrt_spline.generate_random_parallel_velocity(seed=5),
    }
    config = {'general': {}, 'sources': {'plasma': dict(profiles)}}

    filename = str(tmp_path / 'config.json')
    xicsrt_io.save_config(config, filename=filename, overwrite=True)
    loaded = xicsrt_io.load_config(filename)

    for name, profile in profiles.items():
        loaded_profile = loaded['sources']['plasma'][name]
        assert loaded_profile['profile_type'] == profile['profile_type']
        for key in ('x_knots', 'y_knots', 'deriv_zero', 'x_free', 'y_free'):
            np.testing.assert_array_equal(
                np.asarray(loaded_profile[key]), profile[key])


@pytest.mark.parametrize(
    "generator",
    [
        xicsrt_spline.generate_random_emissivity,
        xicsrt_spline.generate_random_temp,
        xicsrt_spline.generate_random_ion_temp,
        xicsrt_spline.generate_random_electron_temp,
        xicsrt_spline.generate_random_parallel_velocity,
    ],
)
@pytest.mark.parametrize("n_knots", [3, 5, 7, 9])
def test_mask_lengths_track_n_knots(generator, n_knots):
    """Masks are derived from n_knots, not hardcoded length-5 literals."""
    profile = generator(n_knots=n_knots, seed=2)
    assert len(profile["x_knots"]) == n_knots
    assert len(profile["y_knots"]) == n_knots
    assert len(profile["deriv_zero"]) == n_knots
    assert len(profile["x_free"]) == n_knots
    assert len(profile["y_free"]) == n_knots


def test_perpendicular_velocity_requires_five_knots():
    """The perpendicular velocity generator only supports 5 knots."""
    with pytest.raises(ValueError):
        xicsrt_spline.generate_random_perpendicular_velocity(n_knots=7, seed=0)


def test_emissivity_normalized_peak():
    """Emissivity is shape-only: the peak knot value is exactly 1.0."""
    for seed in range(20):
        profile = xicsrt_spline.generate_random_emissivity(seed=seed)
        y_knots = np.asarray(profile["y_knots"])
        assert np.max(y_knots) == 1.0
        assert y_knots[-2] == 0.0
        assert y_knots[-1] == 0.0
        assert np.all(y_knots >= 0.0)


def test_temperature_units_ev():
    """Temperature knots are in eV within the documented default ranges."""
    for seed in range(20):
        ion = xicsrt_spline.generate_random_ion_temp(seed=seed)
        y_ion = np.asarray(ion["y_knots"])
        assert 200.0 <= y_ion[0] <= 5000.0
        assert y_ion[-1] == 0.0

        electron = xicsrt_spline.generate_random_electron_temp(seed=seed)
        y_ele = np.asarray(electron["y_knots"])
        assert 200.0 <= y_ele[0] <= 10000.0
        assert y_ele[-1] == 0.0


def test_temperature_not_normalized():
    """Core temperature varies between seeds (it is a physical label)."""
    cores = [
        np.asarray(xicsrt_spline.generate_random_ion_temp(seed=seed)["y_knots"])[0]
        for seed in range(10)
    ]
    assert np.std(cores) > 0.0


def test_velocity_units_m_per_s():
    """Velocity knots respect the m/s defaults of +/- 20e3."""
    for seed in range(20):
        perp = xicsrt_spline.generate_random_perpendicular_velocity(seed=seed)
        y_perp = np.asarray(perp["y_knots"])
        assert -20e3 <= y_perp[1] <= 0.0
        assert 0.0 <= y_perp[3] <= 20e3
        assert y_perp[0] == 0.0
        assert y_perp[2] == 0.0
        assert y_perp[4] == 0.0

        par = xicsrt_spline.generate_random_parallel_velocity(seed=seed)
        y_par = np.asarray(par["y_knots"])
        assert np.all(np.abs(y_par) <= 20e3)
        assert y_par[-1] == 0.0


def test_velocity_range_override():
    """The velocity ranges are user knobs, not constants."""
    profile = xicsrt_spline.generate_random_parallel_velocity(
        y_min=-5e3, y_max=5e3, seed=3
    )
    y_knots = np.asarray(profile["y_knots"])
    assert np.all(np.abs(y_knots) <= 5e3)

    profile = xicsrt_spline.generate_random_perpendicular_velocity(
        y_min=-1e3, y_max=1e3, seed=3
    )
    y_knots = np.asarray(profile["y_knots"])
    assert np.all(np.abs(y_knots) <= 1e3)


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_reproducible_with_seed(generator):
    """The same seed produces the same profile."""
    profile_a = generator(seed=42)
    profile_b = generator(seed=42)
    assert profile_a.keys() == profile_b.keys()
    assert profile_a['profile_type'] == profile_b['profile_type']
    for key in ('x_knots', 'y_knots', 'deriv_zero', 'x_free', 'y_free'):
        np.testing.assert_array_equal(profile_a[key], profile_b[key])


@pytest.mark.parametrize("generator", ALL_GENERATORS)
def test_spline_from_profile_matches_knots(generator):
    """The interpolator passes through the knots exactly."""
    profile = generator(seed=4)
    spline = xicsrt_spline.spline_from_profile(profile)
    x_knots = np.asarray(profile["x_knots"])
    y_knots = np.asarray(profile["y_knots"])
    np.testing.assert_allclose(spline(x_knots), y_knots, atol=1e-12)


def test_spline_from_profile_zero_core_derivative():
    """deriv_zero forces a zero derivative at the marked knots."""
    profile = xicsrt_spline.generate_random_temp(seed=5, zero_deriv_core=True)
    assert profile["deriv_zero"][0]
    spline = xicsrt_spline.spline_from_profile(profile)
    deriv = spline.derivative()(0.0)
    assert abs(deriv) < 1e-12


def test_spline_from_profile_nan_outside_range():
    """Without extrapolation, evaluation outside [0, 1] returns NaN."""
    profile = xicsrt_spline.generate_random_ion_temp(seed=6)
    spline = xicsrt_spline.spline_from_profile(profile)
    assert np.isnan(spline(1.5))
    assert np.isnan(spline(-0.5))

    # NaN input propagates to NaN output (used for outside-LCFS marking).
    result = spline(np.array([0.5, np.nan]))
    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_min_spacing_too_large_raises():
    """An impossible min_spacing raises rather than silently failing."""
    with pytest.raises(ValueError):
        xicsrt_spline.generate_random_temp(n_knots=9, min_spacing=0.5, seed=0)


# ------------------------------------------------------------------------
# Seed decorrelation (F021)
# ------------------------------------------------------------------------

PROFILE_NAMES = [
    "emissivity",
    "ion_temp",
    "electron_temp",
    "perpendicular_velocity",
    "parallel_velocity",
]


def _profiles_from_seed(seed):
    """Build the five-profile set the way callers are expected to."""
    seeds = xicsrt_spline.profile_seeds(seed, PROFILE_NAMES)
    return {
        "emissivity": xicsrt_spline.generate_random_emissivity(
            seed=seeds["emissivity"]),
        "ion_temp": xicsrt_spline.generate_random_ion_temp(
            seed=seeds["ion_temp"]),
        "electron_temp": xicsrt_spline.generate_random_electron_temp(
            seed=seeds["electron_temp"]),
        "perpendicular_velocity":
            xicsrt_spline.generate_random_perpendicular_velocity(
                seed=seeds["perpendicular_velocity"]),
        "parallel_velocity": xicsrt_spline.generate_random_parallel_velocity(
            seed=seeds["parallel_velocity"]),
    }


def test_profile_seeds_are_distinct():
    """Child seeds are distinct, reproducible, and actually needed.

    The final check pins the library contract that motivates this helper:
    generators given the SAME seed alias onto each other. That behavior is
    correct and intended, and it is exactly why the child seeds exist.
    """
    seeds = xicsrt_spline.profile_seeds(1234, PROFILE_NAMES)
    assert set(seeds) == set(PROFILE_NAMES)

    # Each child seeds a different random stream.
    draws = [
        np.random.default_rng(seeds[name]).uniform() for name in PROFILE_NAMES
    ]
    assert len(set(draws)) == len(PROFILE_NAMES)

    # A fixed integer seed reproduces the children exactly.
    seeds_again = xicsrt_spline.profile_seeds(1234, PROFILE_NAMES)
    draws_again = [
        np.random.default_rng(seeds_again[name]).uniform()
        for name in PROFILE_NAMES
    ]
    assert draws == draws_again

    # A different parent seed gives different children.
    seeds_other = xicsrt_spline.profile_seeds(5678, PROFILE_NAMES)
    draws_other = [
        np.random.default_rng(seeds_other[name]).uniform()
        for name in PROFILE_NAMES
    ]
    assert not set(draws) & set(draws_other)

    # seed=None still gives distinct children within a single call.
    seeds_none = xicsrt_spline.profile_seeds(None, PROFILE_NAMES)
    draws_none = [
        np.random.default_rng(seeds_none[name]).uniform()
        for name in PROFILE_NAMES
    ]
    assert len(set(draws_none)) == len(PROFILE_NAMES)

    # Sharing one seed aliases the generators: this is the defect that
    # profile_seeds exists to avoid.
    ion = xicsrt_spline.generate_random_ion_temp(seed=99)
    electron = xicsrt_spline.generate_random_electron_temp(seed=99)
    np.testing.assert_array_equal(ion["x_knots"], electron["x_knots"])


def test_profile_seeds_decorrelate_profiles():
    """Profiles built from child seeds are mutually uncorrelated.

    Over many parent seeds, scalar labels taken from different profiles
    must not track each other. Before this fix the temperature and velocity
    labels were related by exact affine maps, giving |r| = 1.0.

    Same-profile pairs are deliberately excluded: ordered knots within a
    single profile are legitimately correlated with each other.
    """
    num_seed = 500

    label_profile = []
    label_getter = []
    for name in PROFILE_NAMES:
        label_profile.append(name)
        label_getter.append((name, "x_knots", 1))
        label_profile.append(name)
        label_getter.append((name, "y_knots", 0))

    # y_knots[0] is fixed at zero for perpendicular velocity, so use the
    # randomized ion-root and electron-root knots as its amplitude labels.
    index_perp = label_getter.index(("perpendicular_velocity", "y_knots", 0))
    label_getter[index_perp] = ("perpendicular_velocity", "y_knots", 1)
    label_getter.append(("perpendicular_velocity", "y_knots", 3))
    label_profile.append("perpendicular_velocity")

    rows = []
    for seed in range(num_seed):
        profiles = _profiles_from_seed(seed)
        rows.append([
            np.asarray(profiles[name][key])[index]
            for name, key, index in label_getter
        ])

    labels = np.array(rows).T
    assert np.all(np.std(labels, axis=1) > 0.0)

    corr = np.corrcoef(labels)
    profile_of = np.array(label_profile)
    is_cross = profile_of[:, None] != profile_of[None, :]

    max_cross = np.abs(corr[is_cross]).max()
    assert max_cross < 0.20, (
        f"Profiles are correlated across the set: max |r| = {max_cross:.3f}."
    )
