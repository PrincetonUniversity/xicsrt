# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Mosaic crystal interaction (jaxrt).

Port of
:class:`xicsrt.optics._InteractMosaicCrystal.InteractMosaicCrystal` to
pure jax functions. X-rays penetrate into the mosaic crystal through
up to `mosaic_depth` crystallite layers. At each layer, every ray that
has not yet reflected samples a crystallite normal from a Gaussian
distribution around the surface normal and tests the rocking curve
against it.

The numpy engine breaks out of the layer loop early when all rays have
reflected; here the loop always runs `mosaic_depth` passes (fixed trip
count for jit), which produces identical statistics since already
reflected rays are excluded from later passes.

This module was AI generated using Claude (Fable 5).
"""

import jax
import jax.numpy as jnp

from xicsrt.jaxrt import _rays
from xicsrt.jaxrt.interact import _crystal
from xicsrt.jaxrt.interact import _mirror
from xicsrt.jaxrt.tools import _spread


def setup(param):
    """
    Build static mosaic crystal parameters from an initialized optic
    param dict.
    """
    phys = _crystal.setup(param)
    phys['mosaic_spread'] = float(param['mosaic_spread'])
    phys['mosaic_depth'] = int(param['mosaic_depth'])
    phys['mosaic_cutoff'] = (
        None if param['mosaic_cutoff'] is None else float(param['mosaic_cutoff']))
    return phys


def interact(rays, xloc, norm, mask, phys, key):
    """
    Model reflections from a mosaic crystal using a multi-layer model.

    This simulates the penetration of x-rays into the crystal until the
    rays either encounter a crystallite that satisfies the Bragg
    condition or get absorbed. This replicates both the mosaic
    'focusing' quality and the expected throughput.
    """
    # If a cutoff is given, remove rays whose angle from the nominal
    # Bragg angle is too large to plausibly reflect.
    if phys['mosaic_cutoff'] is not None:
        bragg_angle, incident_angle = _crystal.angle_calc(rays, norm, phys)
        angle_sigma = phys['mosaic_spread'] / (2 * jnp.sqrt(2 * jnp.log(2)))
        angle_cutoff = jnp.sqrt(
            -jnp.log(phys['mosaic_cutoff']) * 2 * angle_sigma**2)
        mask = mask & (jnp.abs(bragg_angle - incident_angle) < angle_cutoff)

    direction = rays['direction']
    reflected = jnp.zeros(mask.shape, dtype=bool)

    def _layer(ii, state):
        direction, reflected, key = state
        key, key_normals, key_rocking = jax.random.split(key, 3)

        # Rays that are still alive and have not yet reflected.
        active = mask & ~reflected

        norm_mosaic = _mosaic_normals(norm, phys, key_normals)

        # Rocking curve test against the crystallite normals.
        layer_rays = dict(rays, direction=direction)
        accepted = active & _crystal.angle_check(
            layer_rays, norm_mosaic, active, phys, key_rocking)

        direction = _mirror.reflect(direction, norm_mosaic, accepted)
        reflected = reflected | accepted
        return direction, reflected, key

    direction, reflected, key = jax.lax.fori_loop(
        0, phys['mosaic_depth'], _layer, (direction, reflected, key))

    rays = dict(rays)
    rays['origin'] = xloc
    rays['direction'] = direction
    rays['mask'] = mask & reflected
    return rays


def _mosaic_normals(norm, phys, key):
    """
    Sample crystallite normal vectors in a Gaussian distribution around
    the nominal surface normals.
    """
    num = norm.shape[0]
    hwhm = phys['mosaic_spread'] / 2.0

    dir_local = _spread._sample_flat_gaussian(hwhm, num, key)

    # Create two vectors perpendicular to the surface normal and each
    # other; their orientation is otherwise unimportant.
    o_1 = (jnp.cross(norm, jnp.array([1.0, 0.0, 0.0]))
           + jnp.cross(norm, jnp.array([0.0, 0.0, 1.0])))
    o_1 = _rays.normalize(o_1)
    o_2 = jnp.cross(norm, o_1)
    o_2 = _rays.normalize(o_2)

    norm_mosaic = (dir_local[:, 0:1] * o_1
                   + dir_local[:, 1:2] * o_2
                   + dir_local[:, 2:3] * norm)
    return norm_mosaic
