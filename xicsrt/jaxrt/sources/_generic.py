# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5, Sonnet 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Ray generation for the Generic, Directed and Focused sources (jaxrt).

Port of the numpy engine sources to pure jax functions. The three
source types share all of their behavior except for the aiming of the
emission cone:

Generic
    Emission cone aimed along the source z-axis.
Directed
    Emission cone aimed along a fixed 'direction' for every ray.
Focused
    Emission cone aimed from each ray origin towards a 'target' point.

Poisson statistics (capacity + mask)
------------------------------------
When `use_poisson` is enabled, the true ray count for each iteration is
drawn as N ~ Poisson(intensity), exactly as in the numpy engine. Since
jit compilation requires fixed array shapes, the ray arrays are
allocated with a fixed capacity of mean + 10*sigma and the rays beyond
N are masked from birth. Every element therefore sees exactly N
statistically correct photons. The probability of N exceeding the
capacity is ~1e-23; if it ever happens the engine raises an error
rather than truncating (never silently altering photon statistics).

This module was AI generated using Claude (Fable 5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from xicsrt.jaxrt import _rays
from xicsrt.jaxrt.tools import _spread
from xicsrt.jaxrt.tools import _wavelength


def setup(param, aim):
    """
    Build static source parameters from an initialized numpy source
    `param` dict.

    Parameters
    ----------
    param : dict
        The `param` dict of the initialized numpy source object.
    aim : str
        Emission cone aiming mode: 'zaxis', 'direction' or 'target'.
    """
    if str(param['spatial_dist']) not in ('uniform', 'gaussian'):
        raise NotImplementedError(
            f"spatial_dist '{param['spatial_dist']}' is not supported by "
            "the jaxrt engine.")

    intensity = float(param['intensity'])
    use_poisson = bool(param['use_poisson'])
    if use_poisson:
        # Fixed array capacity of mean + 10 sigma (see module docstring).
        capacity = int(np.ceil(intensity + 10 * np.sqrt(intensity))) + 1
    else:
        if intensity < 1:
            raise ValueError(
                'intensity of less than one encountered. '
                'Turn on poisson statistics.')
        capacity = int(intensity)

    source = {
        'aim': aim,
        'origin': jnp.asarray(param['origin'], dtype=jnp.float64),
        'xaxis': jnp.asarray(param['xaxis'], dtype=jnp.float64),
        'yaxis': jnp.asarray(np.cross(param['zaxis'], param['xaxis']), dtype=jnp.float64),
        'zaxis': jnp.asarray(param['zaxis'], dtype=jnp.float64),
        'xsize': float(param['xsize']),
        'ysize': float(param['ysize']),
        'zsize': float(param['zsize']),
        'spatial_dist': str(param['spatial_dist']),
        'intensity': intensity,
        'use_poisson': use_poisson,
        'capacity': capacity,
        'spread': _spread.setup(param['spread'], param['angular_dist']),
        'wavelength': _wavelength.setup(param),
        'velocity': jnp.asarray(param['velocity'], dtype=jnp.float64),
    }

    if aim == 'direction':
        direction = param['direction']
        if direction is None:
            direction = param['zaxis']
        source['direction'] = jnp.asarray(direction, dtype=jnp.float64)
    elif aim == 'target':
        if param['target'] is None:
            raise ValueError("SourceFocused requires a 'target' to be defined.")
        source['target'] = jnp.asarray(param['target'], dtype=jnp.float64)
    elif aim != 'zaxis':
        raise ValueError(f"Unknown aim mode: {aim}")

    return source


def generate(source, key):
    """
    Generate a bundle of rays from this source.

    Returns the ray bundle. The number of live rays is len(mask) if
    `use_poisson` is off, otherwise a fresh Poisson draw (see module
    docstring). Overflow of the Poisson capacity is detected by the
    engine through the 'overflow' entry.
    """
    num = source['capacity']
    key_n, key_origin, key_direction, key_wavelength = jax.random.split(key, 4)

    rays = _rays.new_rays(num)

    if source['use_poisson']:
        count = jax.random.poisson(key_n, source['intensity'])
        rays['mask'] = jnp.arange(num) < count
        rays['overflow'] = count > num
    else:
        rays['overflow'] = jnp.asarray(False)

    rays['origin'] = _generate_origin(source, num, key_origin)
    rays['direction'] = _generate_direction(source, rays['origin'], key_direction)
    rays['wavelength'], label = _generate_wavelength(
        source, rays['direction'], num, key_wavelength)
    if label is not None:
        rays['label'] = label

    return rays


def _generate_origin(source, num, key):
    """
    Draw ray origins from the configured spatial distribution.
    """
    if source['spatial_dist'] == 'uniform':
        offsets = jax.random.uniform(key, (num, 3), minval=-0.5, maxval=0.5)
        offsets = offsets * jnp.array(
            [source['xsize'], source['ysize'], source['zsize']])
    elif source['spatial_dist'] == 'gaussian':
        # The sizes are interpreted as full-width-at-half-max (fwhm).
        sigma_to_fwhm = 2 * jnp.sqrt(2 * jnp.log(2))
        sigma = jnp.array(
            [source['xsize'], source['ysize'], source['zsize']]) / sigma_to_fwhm
        offsets = sigma * jax.random.normal(key, (num, 3))
    else:
        raise NotImplementedError(
            f"spatial_dist: {source['spatial_dist']} not implemented.")

    # Map x,y,z aligned offsets to the source origin and orientation.
    origin = (source['origin']
              + offsets[:, 0:1] * source['xaxis']
              + offsets[:, 1:2] * source['yaxis']
              + offsets[:, 2:3] * source['zaxis'])
    return origin


def _generate_direction(source, origin, key):
    """
    Draw ray directions: sample the angular spread distribution around
    the z-axis, then rotate into the per-ray emission frame.
    """
    num = origin.shape[0]
    normal = _make_normal(source, origin)

    dir_local = _spread.sample(source['spread'], num, key)

    # Generate basis vectors perpendicular to the emission normal.
    # For isotropic distributions the orientation does not matter; for
    # the xy distributions this reproduces the numpy engine behavior
    # for a source normal directed along the z-axis.
    o_1 = jnp.cross(normal, source['xaxis']) + jnp.cross(normal, source['zaxis'])
    o_1 = _rays.normalize(o_1)
    o_2 = jnp.cross(normal, o_1)
    o_2 = _rays.normalize(o_2)

    # Rotate the local directions into the emission frame:
    # direction = dir_local[0]*o_2 + dir_local[1]*o_1 + dir_local[2]*normal
    direction = (dir_local[:, 0:1] * o_2
                 + dir_local[:, 1:2] * o_1
                 + dir_local[:, 2:3] * normal)
    return direction


def _make_normal(source, origin):
    """
    The per-ray emission cone axis for the configured aiming mode.
    """
    num = origin.shape[0]
    if source['aim'] == 'zaxis':
        axis = source['zaxis']
        normal = jnp.broadcast_to(axis / jnp.linalg.norm(axis), (num, 3))
    elif source['aim'] == 'direction':
        axis = source['direction']
        normal = jnp.broadcast_to(axis / jnp.linalg.norm(axis), (num, 3))
    elif source['aim'] == 'target':
        normal = _rays.normalize(source['target'] - origin)
    else:
        raise ValueError(f"Unknown aim mode: {source['aim']}")
    return normal


def _generate_wavelength(source, direction, num, key):
    """
    Draw ray wavelengths, applying a Doppler shift if the source has a
    bulk velocity.

    Returns
    -------
    wavelength : jnp.ndarray, shape (num,)
    label : jnp.ndarray of int64, shape (num,), or None
        See :any:`xicsrt.jaxrt.tools._wavelength.sample`. The "line index"
        interpretation is specific to `multi_voigt` sampling, not a
        property of `label` in general.
    """
    wavelength, label = _wavelength.sample(source['wavelength'], num, key)

    if bool(np.any(np.asarray(source['velocity']) != 0.0)):
        c = const.physical_constants['speed of light in vacuum'][0]
        wavelength = wavelength * (
            1 - jnp.sum(source['velocity'] * direction, axis=1) / c)

    return wavelength, label
