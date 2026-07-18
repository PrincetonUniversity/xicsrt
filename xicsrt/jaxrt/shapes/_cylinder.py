# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Ray intersection with a cylinder (jaxrt).

Port of :class:`xicsrt.optics._ShapeCylinder.ShapeCylinder` to pure
jax functions. The cylinder axis is along the optic x-axis.
Calculations are performed in global coordinates.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp

from xicsrt.jaxrt import _rays


def setup(param):
    """
    Extract the static cylinder geometry from an initialized optic
    param dict. The cylinder center is precomputed by the numpy optic
    class.
    """
    geom = {
        'center': jnp.asarray(param['center'], dtype=jnp.float64),
        'axis': jnp.asarray(param['xaxis'], dtype=jnp.float64),
        'radius': float(param['radius']),
        'convex': bool(param['convex']),
    }
    return geom


def intersect(rays, geom):
    """
    Intersect rays with the cylinder.

    The ray components perpendicular to the cylinder axis define a 2D
    circle intersection, solved with the quadratic formula.
    """
    origin = rays['origin']
    direction = rays['direction']
    axis = geom['axis']

    # Components of the ray direction and offset perpendicular to the
    # cylinder axis.
    dp = origin - geom['center']
    d_perp = direction - _rays.dot(direction, axis[None, :])[:, None] * axis
    o_perp = dp - _rays.dot(dp, axis[None, :])[:, None] * axis

    # Coefficients for A*t**2 + B*t + C = 0.
    A = _rays.dot(d_perp, d_perp)
    B = 2 * _rays.dot(d_perp, o_perp)
    C = _rays.dot(o_perp, o_perp) - geom['radius']**2

    discriminant = B**2 - 4 * A * C

    # Rays with a negative discriminant miss the cylinder.
    mask = rays['mask'] & (discriminant >= 0)

    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t_0 = (-B - sqrt_disc) / (2 * A)
    t_1 = (-B + sqrt_disc) / (2 * A)

    if geom['convex']:
        distance = jnp.minimum(t_0, t_1)
    else:
        distance = jnp.maximum(t_0, t_1)

    xloc = origin + direction * distance[:, None]

    # The surface normal points from the intersection towards the
    # cylinder axis (projected to the plane of the intersection).
    center_proj = geom['center'] - _rays.dot(
        geom['center'] - xloc, axis[None, :])[:, None] * axis
    norm = _rays.normalize(center_proj - xloc)

    # As in the numpy engine, rays without an intersection get nan.
    xloc = _rays.where_vector(mask, xloc, jnp.nan)

    return xloc, norm, mask
