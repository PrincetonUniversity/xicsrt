# -*- coding: utf-8 -*-
"""
Regression tests pinning the eV convention of the Doppler-sigma formula.

This file includes AI generated code using Claude (Opus 5, Fable 5).

The Doppler broadened line width

    sigma = sqrt(T[eV] / mass_number / amu_kg / c^2 * ev_J) * wavelength

is duplicated in several places (XicsrtSourceGeneric, XicsrtPlasmaGeneric
and the jaxrt wavelength tools). Temperatures are ALWAYS in eV throughout
xicsrt; a keV value sneaking in would silently change sigma by sqrt(1000).
These tests pin the numeric anchor:

    temperature = 1000.0 eV, mass_number = 39.948 (argon),
    wavelength = 3.9492 A  ->  sigma = 6.4740e-4 A

on both the numpy and jaxrt engines.
"""

import numpy as np
import pytest

# The analytic anchor value for the parameters below.
ANCHOR_TEMPERATURE = 1000.0  # eV
ANCHOR_MASS_NUMBER = 39.948  # argon, au
ANCHOR_WAVELENGTH = 3.9492  # Angstrom
ANCHOR_SIGMA = 6.47398297e-4  # Angstrom


def _analytic_sigma():
    import scipy.constants as const

    c = const.physical_constants['speed of light in vacuum'][0]
    amu_kg = const.physical_constants[
        'atomic mass unit-kilogram relationship'][0]
    ev_j = const.physical_constants['electron volt-joule relationship'][0]
    return (np.sqrt(ANCHOR_TEMPERATURE / ANCHOR_MASS_NUMBER / amu_kg
                    / c**2 * ev_j)
            * ANCHOR_WAVELENGTH)


def test_anchor_value():
    """The hardcoded anchor agrees with the analytic formula."""
    np.testing.assert_allclose(_analytic_sigma(), ANCHOR_SIGMA, rtol=1e-8)


def test_source_generic_doppler_sigma():
    """XicsrtSourceGeneric.random_wavelength_normal uses eV."""
    from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric

    config = {
        'temperature': ANCHOR_TEMPERATURE,
        'mass_number': ANCHOR_MASS_NUMBER,
        'wavelength': ANCHOR_WAVELENGTH,
        'linewidth': 0.0,
        'intensity': 1e3,
    }
    source = XicsrtSourceGeneric(config)

    np.random.seed(0)
    samples = source.random_wavelength_normal(200000)
    sigma = np.std(samples)
    np.testing.assert_allclose(sigma, ANCHOR_SIGMA, rtol=0.02)


def test_source_generic_voigt_doppler_sigma():
    """XicsrtSourceGeneric.random_wavelength_voigt uses eV (gamma=0 path)."""
    from xicsrt.sources._XicsrtSourceGeneric import XicsrtSourceGeneric

    config = {
        'temperature': ANCHOR_TEMPERATURE,
        'mass_number': ANCHOR_MASS_NUMBER,
        'wavelength': ANCHOR_WAVELENGTH,
        'linewidth': 0.0,
        'intensity': 1e3,
    }
    source = XicsrtSourceGeneric(config)

    np.random.seed(1)
    samples = source.random_wavelength_voigt(200000)
    sigma = np.std(samples)
    np.testing.assert_allclose(sigma, ANCHOR_SIGMA, rtol=0.02)


def test_plasma_generic_doppler_sigma():
    """XicsrtPlasmaGeneric._generate_wavelengths (voigt) uses eV."""
    from xicsrt.sources._XicsrtPlasmaCubic import XicsrtPlasmaCubic

    n_bundles = 16
    config = {
        'xsize': 0.01, 'ysize': 0.01, 'zsize': 0.01,
        'target': [0.0, 0.0, 1.0],
        'spread': np.radians(1.0),
        'temperature': ANCHOR_TEMPERATURE,
        'mass_number': ANCHOR_MASS_NUMBER,
        'wavelength': ANCHOR_WAVELENGTH,
        'linewidth': 0.0,
        'wavelength_dist': 'voigt',
        'bundle_count': n_bundles,
    }
    plasma = XicsrtPlasmaCubic(config)

    rays_per_bundle = 20000
    bundle_input = {
        'temperature': np.full(n_bundles, ANCHOR_TEMPERATURE),
        'mask': np.ones(n_bundles, dtype=bool),
    }
    m = bundle_input['mask']
    bundle_index = np.repeat(np.arange(n_bundles), rays_per_bundle)

    np.random.seed(2)
    wavelength, line_index = plasma._generate_wavelengths(bundle_input, m, bundle_index)
    assert line_index is None
    sigma = np.std(wavelength)
    np.testing.assert_allclose(sigma, ANCHOR_SIGMA, rtol=0.02)


def test_jaxrt_doppler_sigma():
    """The jaxrt engine's _doppler_sigma uses eV."""
    pytest.importorskip('jax')
    from xicsrt.jaxrt.tools import _wavelength

    param = {
        'temperature': ANCHOR_TEMPERATURE,
        'mass_number': ANCHOR_MASS_NUMBER,
        'wavelength': ANCHOR_WAVELENGTH,
    }
    sigma = _wavelength._doppler_sigma(param)
    np.testing.assert_allclose(sigma, ANCHOR_SIGMA, rtol=1e-8)
