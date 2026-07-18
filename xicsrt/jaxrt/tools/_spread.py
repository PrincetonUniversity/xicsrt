# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

JAX versions of the angular spread (vector distribution) samplers.

These are ports of the samplers in :mod:`xicsrt.tools.xicsrt_spread` to
pure jax functions with explicit random keys. Each sampler draws `num`
unit vectors distributed around the z-axis; the caller is responsible
for rotating them into the emission frame.

Statistics are identical to the numpy engine samplers: the same
distributions are sampled with the same transformations (including the
exact rejection sampling used for 'isotropic_xy').

This module was AI generated using Claude (Fable 5).
"""

import jax
import jax.numpy as jnp
import numpy as np

from xicsrt.tools import xicsrt_spread


def setup(spread, name):
    """
    Parse the spread config (host-side) into static sampler parameters.

    Reuses the numpy engine parsing helpers so that the interpretation
    of the 'spread' config option is guaranteed identical.
    """
    name = name.lower()
    if name in ('isotropic', 'flat', 'gaussian'):
        theta = np.asarray(xicsrt_spread._parse_spread_single(spread), dtype=np.float64)
    elif name in ('isotropic_xy', 'flat_xy'):
        theta = np.asarray(xicsrt_spread._parse_spread_xy(spread), dtype=np.float64)
    else:
        raise NotImplementedError(
            f"Angular distribution '{name}' is not supported by the jaxrt engine.")
    return {'name': name, 'theta': theta}


def sample(params, num, key):
    """
    Draw `num` unit vectors from the configured angular distribution.
    """
    name = params['name']
    theta = params['theta']
    if name == 'isotropic':
        return _sample_isotropic(theta[0], num, key)
    elif name == 'isotropic_xy':
        return _sample_isotropic_xy(theta, num, key)
    elif name == 'flat':
        return _sample_flat(theta[0], num, key)
    elif name == 'flat_xy':
        return _sample_flat_xy(theta, num, key)
    elif name == 'gaussian':
        return _sample_flat_gaussian(theta[0], num, key)
    else:
        raise NotImplementedError(
            f"Angular distribution '{name}' is not supported by the jaxrt engine.")


def _sample_isotropic(theta, num, key):
    """
    Isotropic (uniform spherical) emission within a cone of half-angle theta.
    """
    key_z, key_phi = jax.random.split(key)
    z = jax.random.uniform(key_z, (num,), minval=jnp.cos(theta), maxval=1.0)
    phi = jax.random.uniform(key_phi, (num,), minval=0.0, maxval=2 * jnp.pi)

    sin_theta = jnp.sqrt(1 - z**2)
    return jnp.stack([sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), z], axis=1)


def _sample_flat(theta, num, key):
    """
    Flat (uniform planar) emission within a cone of half-angle theta.
    """
    key_r, key_a = jax.random.split(key)
    r = jnp.sqrt(jax.random.uniform(key_r, (num,), minval=0.0, maxval=jnp.tan(theta)))
    angle1 = jax.random.uniform(key_a, (num,), minval=0.0, maxval=2 * jnp.pi)
    angle0 = jnp.arctan(r)

    return jnp.stack([
        jnp.cos(angle1) * jnp.sin(angle0),
        jnp.sin(angle1) * jnp.sin(angle0),
        jnp.cos(angle0)], axis=1)


def _sample_flat_xy(theta, num, key):
    """
    Flat (uniform planar) emission within a rectangular truncated-cone.

    theta contains [xmin, xmax, ymin, ymax] half-angles.
    """
    key_x, key_y = jax.random.split(key)
    tan_theta = jnp.tan(theta)
    x = jax.random.uniform(key_x, (num,), minval=tan_theta[0], maxval=tan_theta[1])
    y = jax.random.uniform(key_y, (num,), minval=tan_theta[2], maxval=tan_theta[3])

    angle0 = jnp.arctan(jnp.sqrt(x**2 + y**2))
    angle1 = jnp.arctan2(y, x)

    return jnp.stack([
        jnp.cos(angle1) * jnp.sin(angle0),
        jnp.sin(angle1) * jnp.sin(angle0),
        jnp.cos(angle0)], axis=1)


def _sample_flat_gaussian(theta_hwhm, num, key):
    """
    Gaussian distribution of vectors on a flat plane (z=1), normalized to
    unit vectors. theta_hwhm is the angular half-width-at-half-max.

    For small angles this approximates a Gaussian angular distribution.
    """
    # Convert the angular hwhm to sigma, then to a width on the z=1 plane.
    sigma = theta_hwhm / jnp.sqrt(2 * jnp.log(2))
    plane_sigma = jnp.sin(sigma)

    xy = plane_sigma * jax.random.normal(key, (num, 2))
    vectors = jnp.concatenate([xy, jnp.ones((num, 1))], axis=1)
    return vectors / jnp.linalg.norm(vectors, axis=1)[:, None]


def _sample_isotropic_xy(theta, num, key):
    """
    Isotropic (uniform spherical) emission within a rectangular
    truncated-cone, using exact rejection sampling.

    This reproduces the rejection scheme of the numpy engine: vectors
    are drawn from an isotropic circular cone that covers the requested
    rectangle and are re-drawn until they fall inside. Statistics are
    exactly isotropic over the rectangular spread.
    """
    # Extent of the covering circular cone.
    theta_xmax = jnp.max(jnp.abs(theta[0:2]))
    theta_ymax = jnp.max(jnp.abs(theta[2:]))
    theta_max = jnp.arcsin(jnp.sqrt(jnp.sin(theta_xmax)**2 + jnp.sin(theta_ymax)**2))

    sin_theta = jnp.sin(theta)

    def _inside(vectors):
        # Check whether vectors fall inside the requested rectangular spread.
        vx, vy, vz = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        angle_x = vx / jnp.sqrt(vx**2 + vz**2)
        angle_y = vy / jnp.sqrt(vy**2 + vz**2)
        inside = (angle_x > sin_theta[0]) & (angle_x <= sin_theta[1])
        inside &= (angle_y > sin_theta[2]) & (angle_y <= sin_theta[3])
        return inside

    def _cond(state):
        _, accepted, _ = state
        return ~jnp.all(accepted)

    def _body(state):
        vectors, accepted, key = state
        key, subkey = jax.random.split(key)
        candidates = _sample_isotropic(theta_max, num, subkey)
        take = ~accepted & _inside(candidates)
        vectors = jnp.where(take[:, None], candidates, vectors)
        accepted = accepted | take
        return vectors, accepted, key

    vectors = jnp.zeros((num, 3), dtype=jnp.float64)
    accepted = jnp.zeros(num, dtype=bool)
    vectors, accepted, key = jax.lax.while_loop(
        _cond, _body, (vectors, accepted, key))

    return vectors
