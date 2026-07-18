# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Ray intersection with a torus (jaxrt).

Port of :class:`xicsrt.optics._ShapeTorus.ShapeTorus` to pure jax
functions. The intersection distances are the roots of an analytic
quartic; the appropriate root for the configured curvature (convex
options) is selected by index, exactly as in the numpy engine.

The distance calculation is done in 'torus' coordinates, in which the
center of the geometric torus is the origin and the torus axis is the
local y-axis. The normal calculation is done in global coordinates.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp

from xicsrt.jaxrt import _rays
from xicsrt.jaxrt.tools import _quartic


def setup(param):
    """
    Extract the static torus geometry from an initialized optic param
    dict. The geometric torus radii, center and quartic root index are
    precomputed by the numpy optic class.
    """
    geom = {
        'center': jnp.asarray(param['center'], dtype=jnp.float64),
        'orientation': jnp.asarray(param['orientation'], dtype=jnp.float64),
        'zaxis': jnp.asarray(param['zaxis'], dtype=jnp.float64),
        'xaxis': jnp.asarray(param['xaxis'], dtype=jnp.float64),
        'torus_major': float(param['torus_major']),
        'torus_minor': float(param['torus_minor']),
        'root_idx': int(param['root_idx']),
    }
    return geom


def intersect(rays, geom):
    """
    Intersect rays with the torus.
    """
    distance, mask = _intersect_distance(rays, geom)
    xloc = rays['origin'] + rays['direction'] * distance[:, None]
    norm = _intersect_normal(xloc, geom)

    # As in the numpy engine, rays without an intersection get nan.
    xloc = _rays.where_vector(mask, xloc, jnp.nan)

    return xloc, norm, mask


def _intersect_distance(rays, geom):
    """
    Distance along each ray to the torus intersection, from the roots
    of the analytic quartic equation.
    """
    r_major = geom['torus_major']
    r_minor = geom['torus_minor']

    # Transform rays to torus coordinates (origin at the geometric
    # torus center; rows of 'orientation' are the local axes).
    origin = jnp.einsum(
        'ij,kj->ki', geom['orientation'], rays['origin'] - geom['center'])
    direction = jnp.einsum('ij,kj->ki', geom['orientation'], rays['direction'])

    o_mag_sq = _rays.dot(origin, origin)
    dot_od = _rays.dot(origin, direction)
    r_sq = r_major**2 + r_minor**2

    # The torus axis is the local y-axis.
    axis_idx = 1
    d_axis = direction[:, axis_idx]
    o_axis = origin[:, axis_idx]

    # Quartic coefficients: c0*t^4 + c1*t^3 + c2*t^2 + c3*t + c4 = 0.
    c0 = jnp.ones_like(o_mag_sq)
    c1 = 4 * dot_od
    c2 = 4 * dot_od**2 + 2 * o_mag_sq - 2 * r_sq + 4 * r_major**2 * d_axis**2
    c3 = 4 * dot_od * (o_mag_sq - r_sq) + 8 * r_major**2 * d_axis * o_axis
    c4 = (o_mag_sq**2 - 2 * r_sq * o_mag_sq
          + 4 * r_major**2 * o_axis**2 + (r_major**2 - r_minor**2)**2)

    roots = _quartic.multi_quartic(c0, c1, c2, c3, c4)

    # Neglect complex roots, then select the root corresponding to the
    # configured curvature. The root ordering of the analytic solver is
    # deterministic, exactly as in the numpy engine.
    root = roots[geom['root_idx']]
    is_real = root.imag == 0
    distance = root.real

    mask = rays['mask'] & is_real & jnp.isfinite(distance) & (distance > 0.0)

    return distance, mask


def _intersect_normal(xloc, geom):
    """
    Surface normal at each intersection point (global coordinates).

    The normal points from the local center of curvature (which lies on
    the toroidal axis circle) towards the intersection.
    """
    center = geom['center']
    yaxis = jnp.cross(geom['zaxis'], geom['xaxis'])
    r_major = geom['torus_major']

    pt = xloc - center

    # Project onto the plane perpendicular to the torus axis.
    pt = pt - _rays.dot(pt, yaxis[None, :])[:, None] * yaxis
    pt_norm = _rays.normalize(pt)

    # The nearest point on the toroidal axis circle.
    qq = center + r_major * pt_norm

    norm = _rays.normalize(xloc - qq)
    return norm
