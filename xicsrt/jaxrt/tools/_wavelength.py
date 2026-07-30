# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

JAX wavelength samplers for the jaxrt sources.

The Voigt distributions are sampled directly: a Voigt variate is exactly
``center + Normal(0, sigma) + Cauchy(0, gamma)``, and a multiline spectrum
is a mixture where each ray's line is chosen with probability proportional
to the line intensities. This mirrors the numpy engine's direct samplers
(``xicsrt_voigt.voigt_random`` / ``xicsrt_voigt_multi.multi_voigt_random``)
and is exact — there is no CDF-table truncation or interpolation error.

This module was AI generated using Claude (Fable 5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const


def setup(param):
    """
    Parse the wavelength config options (host side) into static sampler
    parameters.

    Parameters
    ----------
    param : dict
        The `param` dict of an initialized numpy source object.
    """
    dist = str.lower(param['wavelength_dist'])

    if dist == 'monochrome':
        return {'name': 'monochrome',
                'wavelength': float(param['wavelength'])}

    if dist == 'uniform':
        return {'name': 'uniform',
                'range': np.asarray(param['wavelength_range'], dtype=np.float64)}

    if dist == 'voigt':
        return _setup_voigt(param)

    if dist == 'multi_voigt':
        intensities = np.asarray(param['line_intensities'], dtype=np.float64)
        cum = np.cumsum(intensities)
        cum /= cum[-1]
        return {
            'name': 'multi_voigt',
            'locations': jnp.asarray(param['line_locations'], dtype=jnp.float64),
            'sigmas': jnp.asarray(param['line_sigmas'], dtype=jnp.float64),
            'gammas': jnp.asarray(param['line_gammas'], dtype=jnp.float64),
            'cum_intensity': jnp.asarray(cum),
        }

    raise NotImplementedError(
        f"wavelength_dist '{dist}' is not supported by the jaxrt engine.")


def _setup_voigt(param):
    """
    Build sampler parameters for a single Voigt line.

    The special cases follow the numpy engine exactly:
    zero linewidth and temperature -> monochrome;
    zero linewidth -> gaussian.
    The direct sampler handles zero temperature (pure Lorentzian) exactly.
    """
    wavelength = float(param['wavelength'])
    linewidth = float(param['linewidth'])
    temperature = float(param['temperature'])

    if linewidth == 0.0 and temperature == 0.0:
        return {'name': 'monochrome', 'wavelength': wavelength}

    if linewidth == 0.0:
        return {'name': 'gaussian',
                'wavelength': wavelength,
                'sigma': _doppler_sigma(param)}

    c = const.physical_constants['speed of light in vacuum'][0]
    gamma = linewidth * wavelength**2 / (4 * np.pi * c * 1e10)
    sigma = _doppler_sigma(param)

    return {'name': 'voigt',
            'wavelength': wavelength,
            'sigma': sigma,
            'gamma': gamma}


def _doppler_sigma(param):
    """
    Doppler-broadened line width (sigma) from temperature, in Angstroms.
    """
    c = const.physical_constants['speed of light in vacuum'][0]
    amu_kg = const.physical_constants['atomic mass unit-kilogram relationship'][0]
    ev_j = const.physical_constants['electron volt-joule relationship'][0]
    return (np.sqrt(param['temperature'] / param['mass_number'] / amu_kg / c**2 * ev_j)
            * param['wavelength'])


def sample(params, num, key):
    """
    Draw `num` wavelengths from the configured distribution.
    """
    name = params['name']

    if name == 'monochrome':
        return jnp.full(num, params['wavelength'], dtype=jnp.float64)

    if name == 'uniform':
        return jax.random.uniform(
            key, (num,), minval=params['range'][0], maxval=params['range'][1])

    if name == 'gaussian':
        return params['wavelength'] + params['sigma'] * jax.random.normal(key, (num,))

    if name == 'voigt':
        # Direct Voigt sampling: center + Normal(0, sigma) + Cauchy(0, gamma).
        key_n, key_c = jax.random.split(key)
        return (params['wavelength']
                + params['sigma'] * jax.random.normal(key_n, (num,))
                + params['gamma'] * jax.random.cauchy(key_c, (num,)))

    if name == 'multi_voigt':
        # Mixture sampling: pick a line by intensity weight, then draw that
        # line's Voigt variate directly.
        key_u, key_n, key_c = jax.random.split(key, 3)
        uniform = jax.random.uniform(key_u, (num,))
        index = jnp.searchsorted(params['cum_intensity'], uniform)
        return (params['locations'][index]
                + params['sigmas'][index] * jax.random.normal(key_n, (num,))
                + params['gammas'][index] * jax.random.cauchy(key_c, (num,)))

    raise NotImplementedError(f"Wavelength sampler '{name}' unknown.")
