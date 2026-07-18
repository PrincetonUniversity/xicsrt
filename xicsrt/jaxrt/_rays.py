# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

The ray bundle used by the jaxrt engine.

A ray bundle is a plain dict of jax arrays (a pytree), mirroring the
`RayArray` used by the numpy engine:

origin : (num, 3) float64
    Ray starting positions in global coordinates.
direction : (num, 3) float64
    Ray direction unit-vectors in global coordinates.
wavelength : (num,) float64
    Ray wavelengths in Angstroms.
weight : (num,) float64
    Statistical weight of each ray (currently always 1.0).
mask : (num,) bool
    True for rays that are still alive ('found' so far).
    Rays are never removed from the arrays; instead they are masked
    out. This keeps array shapes fixed, which is required for jit
    compilation, and exactly matches the photon statistics of the
    numpy engine (which also keeps lost rays in place).

Masking convention
------------------
The numpy engine updates values only for masked rays using in-place
indexed assignment (``value[m] = ...``). In jax, arrays are immutable,
so the same pattern is written with `jnp.where`:

    value = jnp.where(mask, new_value, value)

The small helpers in this module keep that pattern readable in the
physics code.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp


def new_rays(num):
    """
    Create a new ray bundle of the given fixed size.

    All rays start alive (mask=True), at the origin, with zero
    direction and wavelength. Sources are responsible for filling in
    physical values.
    """
    rays = {
        'origin': jnp.zeros((num, 3), dtype=jnp.float64),
        'direction': jnp.zeros((num, 3), dtype=jnp.float64),
        'wavelength': jnp.zeros(num, dtype=jnp.float64),
        'weight': jnp.ones(num, dtype=jnp.float64),
        'mask': jnp.ones(num, dtype=bool),
    }
    return rays


def where_scalar(mask, value_true, value_false):
    """
    Select per-ray scalar values (shape (num,)) based on a mask.
    """
    return jnp.where(mask, value_true, value_false)


def where_vector(mask, value_true, value_false):
    """
    Select per-ray vector values (shape (num, 3)) based on a mask.

    The mask has shape (num,) and is broadcast across the vector
    components.
    """
    return jnp.where(mask[:, None], value_true, value_false)


def dot(a, b):
    """
    Row-wise dot product of two (num, 3) vector arrays.

    Equivalent to ``np.einsum('ij,ij->i', a, b)`` in the numpy engine.
    """
    return jnp.sum(a * b, axis=1)


def normalize(vectors):
    """
    Normalize an array of vectors with shape (num, 3).
    """
    return vectors / jnp.linalg.norm(vectors, axis=1)[:, None]
