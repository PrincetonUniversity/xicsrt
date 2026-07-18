# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Vectorized analytical quartic solver for the jaxrt engine.

Port of :func:`xicsrt.tools.xicsrt_quartic.multi_quartic` (Ferrari and
Cardano closed-form solutions, adapted from NKrvavica/fqs) to jax. The
roots are returned in exactly the same order as the numpy version,
which the torus shape relies on to select the correct intersection.

The numpy version uses boolean-mask branches for the cubic sub-solver;
here all branches are evaluated and combined with `jnp.where`, with
domain clamping (`maximum`, `clip`) so the unselected branches cannot
produce nan.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp


def _cubic_root(x):
    """
    Real cube root that preserves the sign of the argument.
    """
    return jnp.sign(x) * jnp.abs(x)**(1 / 3)


def _cubic_one_real_root(a0, b0, c0, d0):
    """
    One real root of a cubic equation a0*x^3 + b0*x^2 + c0*x + d0 = 0.

    Port of multi_cubic(..., all_roots=False).
    """
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    third = 1. / 3.
    a13 = a * third
    a2 = a13 * a13

    f = third * b - a2
    g = a13 * (2 * a2 - b) + c
    h = 0.25 * g * g + f * f * f

    # Case 1: all roots real and equal.
    root_equal = -_cubic_root(c)

    # Case 2: all roots real and distinct (h <= 0).
    j = jnp.sqrt(jnp.maximum(-f, 0.0))
    j3 = jnp.where(j > 0, j * j * j, 1.0)
    k = jnp.arccos(jnp.clip(-0.5 * g / j3, -1.0, 1.0))
    root_distinct = 2 * j * jnp.cos(third * k) - a13

    # Case 3: one real root, two complex (h > 0).
    sqrt_h = jnp.sqrt(jnp.maximum(h, 0.0))
    ss = _cubic_root(-0.5 * g + sqrt_h)
    uu = _cubic_root(-0.5 * g - sqrt_h)
    root_one_real = ss + uu - a13

    m1 = (f == 0) & (g == 0) & (h == 0)
    m2 = (~m1) & (h <= 0)

    return jnp.where(m1, root_equal, jnp.where(m2, root_distinct, root_one_real))


def _quadratic_roots(b, c):
    """
    Complex roots of x^2 + b*x + c = 0 (monic), in the same order as
    multi_quadratic in the numpy engine.
    """
    half_b = -0.5 * b
    sqrt_delta = jnp.sqrt(half_b * half_b - c + 0j)
    return half_b - sqrt_delta, half_b + sqrt_delta


def multi_quartic(a0, b0, c0, d0, e0):
    """
    Complex roots of a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0.

    Returns four arrays of complex roots, in the same order as
    :func:`xicsrt.tools.xicsrt_quartic.multi_quartic`.
    """
    a, b, c, d = b0 / a0, c0 / a0, d0 / a0, e0 / a0

    a0q = 0.25 * a
    a02 = a0q * a0q

    # Coefficients of the subsidiary cubic equation.
    p = 3 * a02 - 0.5 * b
    q = a * a02 - b * a0q + 0.5 * c
    r = 3 * a02 * a02 - b * a02 + c * a0q - d

    # One root of the cubic equation.
    z0 = _cubic_one_real_root(
        jnp.ones_like(p), p, r, p * r - 0.5 * q * q)

    s = jnp.sqrt(2 * p + 2 * z0 + 0j)
    s_safe = jnp.where(s == 0, 1.0, s)
    t = jnp.where(s == 0, z0 * z0 + r, -q / s_safe)

    r0, r1 = _quadratic_roots(s, z0 + t)
    r2, r3 = _quadratic_roots(-s, z0 - t)

    return r0 - a0q, r1 - a0q, r2 - a0q, r3 - a0q
