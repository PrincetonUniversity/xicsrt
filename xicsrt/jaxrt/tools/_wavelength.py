# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

JAX wavelength samplers for the jaxrt sources.

The Voigt distributions are sampled by inverse-CDF interpolation. The
CDF tables are built once on the host (in `setup`, with the same numpy
routines used by the numpy engine) and the per-ray sampling inside the
jit'd trace is a uniform draw plus `jnp.interp`. This is statistically
identical to the numpy engine, which uses the same CDF tables and the
same linear interpolation.

This module was AI generated using Claude (Fable 5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from xicsrt.tools import xicsrt_voigt
from xicsrt.tools import xicsrt_voigt_multi


def setup(param):
    """
    Parse the wavelength config options (host side) into static sampler
    parameters, precomputing CDF tables where needed.

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
        cdf_x, cdf, _ = xicsrt_voigt_multi.multi_voigt_cdf_tab(
            param['line_locations'],
            param['line_intensities'],
            param['line_sigmas'],
            param['line_gammas'],
            gridsize=param['multi_gridsize'],
            cutoff=param['multi_cutoff'],
        )
        return {'name': 'cdf', 'cdf_x': jnp.asarray(cdf_x), 'cdf': jnp.asarray(cdf)}

    raise NotImplementedError(
        f"wavelength_dist '{dist}' is not supported by the jaxrt engine.")


def _setup_voigt(param):
    """
    Build sampler parameters for a single Voigt line.

    The special cases follow the numpy engine exactly:
    zero linewidth and temperature -> monochrome;
    zero linewidth -> gaussian;
    zero temperature -> voigt with temperature clamped to 1 eV
    (the numpy engine applies the same temporary clamp).
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

    if temperature == 0.0:
        # See random_wavelength_voigt in the numpy engine: the CDF
        # generator cannot handle zero temperature, so 1 eV is added.
        temperature = 1.0
        param = dict(param, temperature=temperature)

    c = const.physical_constants['speed of light in vacuum'][0]
    gamma = linewidth * wavelength**2 / (4 * np.pi * c * 1e10)
    sigma = _doppler_sigma(param)

    cdf_x, cdf = xicsrt_voigt.voigt_cdf_tab(gamma, sigma)
    return {'name': 'cdf',
            'cdf_x': jnp.asarray(cdf_x) + wavelength,
            'cdf': jnp.asarray(cdf)}


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

    if name == 'cdf':
        # Inverse-CDF sampling with a tabulated CDF, identical to the
        # numpy engine's voigt_random / multi_voigt_random.
        cdf = params['cdf']
        random_y = jax.random.uniform(
            key, (num,), minval=jnp.min(cdf), maxval=jnp.max(cdf))
        return jnp.interp(random_y, cdf, params['cdf_x'])

    raise NotImplementedError(f"Wavelength sampler '{name}' unknown.")
