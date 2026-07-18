# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Ray intersection with a plane (jaxrt).

Port of :class:`xicsrt.optics._ShapePlane.ShapePlane` to pure jax
functions. Calculations are performed in global coordinates.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp

from xicsrt.jaxrt import _rays


def setup(param):
    """
    Extract the static plane geometry from an initialized optic param dict.
    """
    geom = {
        'origin': jnp.asarray(param['origin'], dtype=jnp.float64),
        'zaxis': jnp.asarray(param['zaxis'], dtype=jnp.float64),
    }
    return geom


def intersect(rays, geom):
    """
    Intersect rays with the plane.

    Returns the intersection locations, surface normals and updated mask.
    Rays traveling away from the plane (negative distance) are masked out.
    """
    origin = rays['origin']
    direction = rays['direction']

    distance = (jnp.dot(geom['origin'] - origin, geom['zaxis'])
                / jnp.dot(direction, geom['zaxis']))

    mask = rays['mask'] & (distance >= 0)

    xloc = origin + direction * distance[:, None]
    norm = jnp.broadcast_to(geom['zaxis'], xloc.shape)

    # As in the numpy engine, rays without an intersection get nan.
    xloc = _rays.where_vector(mask, xloc, jnp.nan)

    return xloc, norm, mask
