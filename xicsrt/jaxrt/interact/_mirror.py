# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Perfect mirror interaction (jaxrt).

Port of :class:`xicsrt.optics._InteractMirror.InteractMirror` to pure
jax functions.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp

from xicsrt.jaxrt import _rays


def setup(param):
    """
    No static parameters are needed for a perfect mirror.
    """
    return {}


def interact(rays, xloc, norm, mask, phys, key):
    """
    Specular reflection about the surface normal.

    The origin moves to the intersection for all rays (as in the numpy
    engine); the direction is only updated for rays that reflect.
    """
    rays = dict(rays)
    rays['origin'] = xloc
    rays['direction'] = reflect(rays['direction'], norm, mask)
    rays['mask'] = mask
    return rays


def reflect(direction, norm, mask):
    """
    Reflect the masked ray directions about the surface normals:
    D' = D - 2 (D . n) n
    """
    reflected = direction - 2 * _rays.dot(direction, norm)[:, None] * norm
    return _rays.where_vector(mask, reflected, direction)
