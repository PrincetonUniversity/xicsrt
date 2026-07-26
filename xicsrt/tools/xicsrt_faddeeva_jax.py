# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A JAX port of :mod:`xicsrt.tools.xicsrt_faddeeva`.

This file includes AI generated code using Claude (Opus 4.8, Sonnet 4.6).

This module is an exploratory, opt-in acceleration path for the plain
numpy+scipy object-oriented engine (see F004 in
``devel/features_request.md``). It is NOT used by any code path unless
explicitly selected (see the ``_USE_JAX_VOIGT_MULTI`` toggle in
:mod:`xicsrt.sources._XicsrtSourceGeneric`), and it is unrelated to the
separate :mod:`xicsrt.jaxrt` raytracing engine.

The Weideman (1994) rational-approximation coefficients are identical to
those used by the numpy engine and are reused unchanged (they are computed
with plain numpy, outside of any jit boundary, and cached). Only the
per-call evaluation of ``w(z)`` and the Voigt profile are re-expressed with
``jax.numpy`` so that they can be jit-compiled.

Reference
---------
J. A. C. Weideman, "Computation of the complex error function",
SIAM Journal on Numerical Analysis, 31 (1994) 1497-1518.
"""

import jax
import jax.numpy as jnp

# Enable float64 before any JAX array is created. This is a per-process
# setting; without it JAX silently downcasts to float32, which is not
# sufficient precision for this application (see xics_jax/__init__.py and
# xicsrt/jaxrt/__init__.py, which enable the same setting for the same
# reason).
jax.config.update('jax_enable_x64', True)

from xicsrt.tools.xicsrt_faddeeva import DEFAULT_N, _weideman_coeffs

__all__ = ['DEFAULT_N', 'wofz_weideman', 'voigt_profile']


@jax.jit
def _wofz_weideman_jit(z, L, aa):
    """
    Jit core for the Weideman approximation of w(z).

    This function was AI generated using Claude (Opus 4.8).
    """
    denom = L - 1j * z
    ZZ = (L + 1j * z) / denom
    pp = jnp.polyval(aa, ZZ)
    ww = 2.0 * pp / denom**2 + (1.0 / jnp.sqrt(jnp.pi)) / denom
    return ww


def wofz_weideman(z, N=DEFAULT_N):
    """
    Evaluate the Faddeeva function w(z) using the Weideman approximation.

    This function was AI generated using Claude (Opus 4.8, Sonnet 4.6).

    JAX-jit'd equivalent of
    :func:`xicsrt.tools.xicsrt_faddeeva.wofz_weideman`. The rational
    approximation coefficients (``L``, ``aa``) are computed with plain numpy
    (cached, one-time cost per value of ``N``) and only the array evaluation
    is traced.

    Parameters
    ----------
    z : array_like (complex)
        Points at which to evaluate w(z).
    N : int, optional
        Number of terms in the rational approximation. Default is
        ``DEFAULT_N`` (16).

    Returns
    -------
    jax.Array (complex)
        The approximated values of w(z).
    """
    L, aa = _weideman_coeffs(int(N))
    z = jnp.asarray(z)
    return _wofz_weideman_jit(z, float(L), jnp.asarray(aa))


def voigt_profile(x, location, intensity, sigma, gamma, N=DEFAULT_N):
    """
    Evaluate one or more (broadcast) Voigt profiles on a grid.

    This function was AI generated using Claude (Opus 4.8, Sonnet 4.6).

    JAX-jit'd equivalent of
    :func:`xicsrt.tools.xicsrt_faddeeva.voigt_profile`. See that function
    for parameter documentation; the two are numerically equivalent.

    Parameters
    ----------
    x : array_like
        Wavelength grid.
    location : array_like
        Center of each line.
    intensity : array_like
        Area (strength) scaling of each line.
    sigma : array_like
        Gaussian width of each line.
    gamma : array_like
        Lorentzian width of each line.
    N : int, optional
        Number of terms in the Weideman approximation. Default ``DEFAULT_N``.

    Returns
    -------
    jax.Array
        The Voigt profile(s) evaluated on the grid (same shape as the
        broadcast of the inputs).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    zz = (x - location + 1j * gamma) / (jnp.sqrt(2.0) * sigma)
    yy = wofz_weideman(zz, N=N).real / (jnp.sqrt(2.0 * jnp.pi) * sigma) * intensity
    return yy
