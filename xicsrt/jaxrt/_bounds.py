# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Optic bounds and aperture checking for the jaxrt engine.

Port of `TraceObject.check_bounds` (size and aperture checks) from the
numpy engine to pure jax functions. The intersection locations are
transformed to optic-local coordinates and checked against the optic
extent (xsize/ysize/zsize) and any configured apertures.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp
import numpy as np

from xicsrt.tools import xicsrt_aperture


def setup(param):
    """
    Extract static bounds-checking parameters from an initialized optic
    param dict.

    The optic orientation matrix is stored so intersections can be
    transformed to local coordinates inside the jit'd trace.
    """
    bounds = {
        'origin': jnp.asarray(param['origin'], dtype=jnp.float64),
        'orientation': jnp.asarray(param['orientation'], dtype=jnp.float64),
        'check_size': bool(param['check_size']),
        'sizes': [],
        'apertures': [],
    }

    # Store (axis index, half-size) pairs for the configured sizes.
    for ii, name in enumerate(['xsize', 'ysize', 'zsize']):
        if param[name] is not None:
            bounds['sizes'].append((ii, float(param[name]) / 2))

    if param['check_aperture'] and param['aperture'] is not None:
        for aperture in np.atleast_1d(param['aperture']):
            bounds['apertures'].append(_setup_aperture(aperture))

    return bounds


def _setup_aperture(aperture):
    """
    Normalize a single aperture config dict into static parameters.

    Reuses the numpy engine defaults parsing so interpretation of the
    aperture options is identical.
    """
    aperture = xicsrt_aperture._aperture_defaults(aperture)
    out = {
        'shape': aperture['shape'],
        'logic': aperture['logic'],
        'origin': jnp.asarray(aperture['origin'], dtype=jnp.float64),
    }
    if 'size' in aperture:
        out['size'] = jnp.asarray(aperture['size'], dtype=jnp.float64)
    if 'vertices' in aperture:
        out['vertices'] = jnp.asarray(aperture['vertices'], dtype=jnp.float64)
    return out


def check_bounds(xloc, mask, bounds):
    """
    Mask out rays whose intersection falls outside the optic bounds
    (size checks) or outside the configured apertures.
    """
    # Transform intersections to optic-local coordinates.
    # orientation rows are the local (x, y, z) axes in global coordinates.
    xlocal = jnp.einsum('ij,kj->ki', bounds['orientation'], xloc - bounds['origin'])

    if bounds['check_size']:
        for axis, half_size in bounds['sizes']:
            mask = mask & (jnp.abs(xlocal[:, axis]) < half_size)

    for aperture in bounds['apertures']:
        mask = _apply_aperture(xlocal, mask, aperture)

    return mask


def _apply_aperture(xlocal, mask, aperture):
    """
    Apply a single aperture test to the mask using the aperture logic.
    """
    inside = _aperture_inside(xlocal, aperture)

    logic = aperture['logic']
    if logic == 'and':
        mask = mask & inside
    elif logic == 'not':
        mask = mask & ~inside
    elif logic == 'or':
        mask = mask | inside
    elif logic == 'nand':
        mask = ~(mask & inside)
    elif logic == 'nor':
        mask = ~(mask | inside)
    elif logic == 'xor':
        mask = mask ^ inside
    elif logic == 'xnor':
        mask = ~(mask ^ inside)
    else:
        raise Exception(f'Aperture logic "{logic}" is not known.')

    return mask


def _aperture_inside(xlocal, aperture):
    """
    Boolean test of whether local intersections fall inside an aperture.
    """
    shape = aperture['shape']
    xx = xlocal[:, 0]
    yy = xlocal[:, 1]

    if shape == 'none':
        return jnp.ones(xlocal.shape[0], dtype=bool)

    ox = aperture['origin'][0]
    oy = aperture['origin'][1]

    if shape == 'circle':
        radius = aperture['size'][0]
        return (xx - ox)**2 + (yy - oy)**2 < radius**2

    if shape == 'square':
        half = aperture['size'][0] / 2
        return (jnp.abs(xx - ox) < half) & (jnp.abs(yy - oy) < half)

    if shape == 'rectangle':
        half_x = aperture['size'][0] / 2
        half_y = aperture['size'][1] / 2
        return (jnp.abs(xx - ox) < half_x) & (jnp.abs(yy - oy) < half_y)

    if shape == 'ellipse':
        size_x = aperture['size'][0]
        size_y = aperture['size'][1]
        return ((xx - ox) / size_x)**2 + ((yy - oy) / size_y)**2 < 1

    if shape == 'triangle':
        p0 = aperture['vertices'][0, 0:2] + aperture['origin'][0:2]
        p1 = aperture['vertices'][1, 0:2] + aperture['origin'][0:2]
        p2 = aperture['vertices'][2, 0:2] + aperture['origin'][0:2]
        return _point_in_triangle_2d(xx, yy, p0, p1, p2)

    raise Exception(f'Aperture shape: "{shape}" is not implemented.')


def _point_in_triangle_2d(xx, yy, p0, p1, p2):
    """
    Barycentric point-in-triangle test (2D), matching
    :func:`xicsrt.tools.xicsrt_math.point_in_triangle_2d`.
    """
    area = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0])
                  + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
    aa = 1 / (2 * area) * (p0[1] * p2[0] - p0[0] * p2[1]
                           + (p2[1] - p0[1]) * xx + (p0[0] - p2[0]) * yy)
    bb = 1 / (2 * area) * (p0[0] * p1[1] - p0[1] * p1[0]
                           + (p0[1] - p1[1]) * xx + (p1[0] - p0[0]) * yy)
    cc = 1 - aa - bb
    return (aa >= 0) & (bb >= 0) & (cc >= 0)
