# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A jax-friendly Faddeeva (complex error) function and Voigt profile kernel.

This file includes AI generated code using Claude (Opus 4.8).

The routines here implement the Weideman (1994) rational approximation of the
Faddeeva function ``w(z)``.  Unlike ``scipy.special.wofz``, which is a compiled
C routine that cannot be traced by JAX, everything in this module is expressed
in pure array arithmetic (``+ - * /``, ``numpy.polyval`` and a few elementwise
transcendental calls).  As a result the profile evaluation can later be
accelerated with ``jax.numpy`` simply by swapping the array backend, and it is
differentiable and vectorizable.

Reference
---------
J. A. C. Weideman, "Computation of the complex error function",
SIAM Journal on Numerical Analysis, 31 (1994) 1497-1518.
"""

import functools

import numpy as np

# Default number of terms in the Weideman rational approximation.  N=16 gives
# a maximum absolute error of ~3e-7 in w(z), which is far below the accuracy
# required for CDF construction and random sampling.  Accuracy improves rapidly
# with N (N=24 ~2e-10, N=32 ~1e-13) at modest additional cost.
DEFAULT_N = 16


@functools.lru_cache(maxsize=None)
def _weideman_coeffs(N):
    """
    Compute the coefficients for the Weideman rational approximation of w(z).

    This function was AI generated using Claude (Opus 4.8).

    Parameters
    ----------
    N : int
        Number of terms in the rational approximation.

    Returns
    -------
    L : float
        The scaling constant of the Möbius transform.
    a : numpy.ndarray
        Real coefficient array, ordered for use with ``numpy.polyval``.

    Notes
    -----
    This is a setup-only routine: it uses ``tan``, ``exp`` and an ``fft`` but
    is called at most once per value of ``N`` (results are cached).  The
    coefficients it returns are consumed by :func:`wofz_weideman`, which is
    fully jax-compatible.
    """
    M = 2 * N
    M2 = 2 * M
    kk = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2.0))
    theta = kk * np.pi / M
    tt = L * np.tan(theta / 2.0)
    ff = np.exp(-(tt**2)) * (L**2 + tt**2)
    ff = np.append(0.0, ff)
    aa = np.real(np.fft.fft(np.fft.fftshift(ff))) / M2
    aa = np.flipud(aa[1 : N + 1])
    return L, aa


def wofz_weideman(z, N=DEFAULT_N):
    """
    Evaluate the Faddeeva function w(z) using the Weideman approximation.

    This function was AI generated using Claude (Opus 4.8).

    The Faddeeva function is ``w(z) = exp(-z^2) * erfc(-i z)``.  This is a pure
    arithmetic (jax-friendly) replacement for ``scipy.special.wofz``.

    Parameters
    ----------
    z : array_like (complex)
        Points at which to evaluate w(z).
    N : int, optional
        Number of terms in the rational approximation.  Larger values are more
        accurate.  Default is ``DEFAULT_N`` (16).

    Returns
    -------
    numpy.ndarray (complex)
        The approximated values of w(z).
    """
    L, aa = _weideman_coeffs(int(N))
    z = np.asarray(z)
    denom = L - 1j * z
    ZZ = (L + 1j * z) / denom
    pp = np.polyval(aa, ZZ)
    ww = 2.0 * pp / denom**2 + (1.0 / np.sqrt(np.pi)) / denom
    return ww


def voigt_profile(x, location, intensity, sigma, gamma, N=DEFAULT_N):
    """
    Evaluate one or more (broadcast) Voigt profiles on a grid.

    This function was AI generated using Claude (Opus 4.8).

    The Voigt profile is the real part of the Faddeeva function.  This single
    kernel is shared by both the single-line (:mod:`xicsrt.tools.xicsrt_voigt`)
    and multi-line (:mod:`xicsrt.tools.xicsrt_multi_voigt`) code paths.  All
    parameters broadcast against one another following the usual numpy rules,
    so multiple lines can be evaluated at once by supplying arrays that
    broadcast against ``x``.

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
        Number of terms in the Weideman approximation.  Default ``DEFAULT_N``.

    Returns
    -------
    numpy.ndarray
        The Voigt profile(s) evaluated on the grid (same shape as the
        broadcast of the inputs).
    """
    x = np.asarray(x, dtype=float)
    zz = (x - location + 1j * gamma) / (np.sqrt(2.0) * sigma)
    yy = wofz_weideman(zz, N=N).real / (np.sqrt(2.0 * np.pi) * sigma) * intensity
    return yy
