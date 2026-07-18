# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Ray intersection with a sphere (jaxrt).

Port of :class:`xicsrt.optics._ShapeSphere.ShapeSphere` to pure jax
functions. Calculations are performed in global coordinates.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp

from xicsrt.jaxrt import _rays


def setup(param):
    """
    Extract the static sphere geometry from an initialized optic param
    dict. The sphere center is precomputed by the numpy optic class.
    """
    geom = {
        'center': jnp.asarray(param['center'], dtype=jnp.float64),
        'radius': float(param['radius']),
        'convex': bool(param['convex']),
    }
    return geom


def intersect(rays, geom):
    """
    Intersect rays with the sphere.

    Standard geometric ray-sphere intersection: project the vector to
    the sphere center onto the ray, then step forward or backward by
    the half-chord distance depending on the chosen curvature.
    """
    origin = rays['origin']
    direction = rays['direction']

    # L is the vector from the ray origin to the center of the sphere.
    # t_ca is the projection of L along the ray direction.
    L = geom['center'] - origin
    t_ca = _rays.dot(L, direction)

    # d is the minimum distance between the ray and the center.
    d_sq = _rays.dot(L, L) - t_ca**2

    # If d is larger than the radius, the ray misses the sphere.
    mask = rays['mask'] & (d_sq <= geom['radius']**2)

    # t_hc is the distance from the closest approach to the surface.
    t_hc = jnp.sqrt(jnp.maximum(geom['radius']**2 - d_sq, 0.0))

    if geom['convex']:
        distance = t_ca - t_hc
    else:
        distance = t_ca + t_hc

    xloc = origin + direction * distance[:, None]
    norm = _rays.normalize(geom['center'] - xloc)

    # As in the numpy engine, rays without an intersection get nan.
    xloc = _rays.where_vector(mask, xloc, jnp.nan)

    return xloc, norm, mask
