# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A JAX-accelerated, drop-in replacement for
:mod:`xicsrt.tools.xicsrt_voigt_multi`.

This file includes AI generated code using Claude (Opus 4.8, Sonnet 4.6).

This module is an exploratory, opt-in acceleration path for the plain
numpy+scipy object-oriented (OO) engine. It is intended to be used exactly
like :mod:`xicsrt.tools.xicsrt_voigt_multi` (same function names, arguments
and return values), and is selected only via the ``_USE_JAX_VOIGT_MULTI``
toggle in :mod:`xicsrt.sources._XicsrtSourceGeneric`. It is unrelated to the
separate :mod:`xicsrt.jaxrt` raytracing engine.

Background (see F004 in ``devel/features_request.md`` for the full history):
a first attempt at jax-accelerating this code jit-compiled a new function for
every ray bundle, because each plasma bundle has its own auto-computed
``gridsize`` (temperature-dependent) and its own Poisson-derived ray count
``size``. Both are static/trace arguments to ``jax.jit``, so nearly every
bundle triggered an XLA recompile, and the approach was abandoned as slower
than plain numpy for the (small, ~15-line) spectrum used in that benchmark.

This module avoids both retracing hazards:

* ``size`` never enters the jit boundary. The final uniform draw and
  inverse-CDF interpolation are done in plain numpy on the host, exactly as
  in :mod:`xicsrt.tools.xicsrt_voigt_multi`. This also means sampling is
  still driven by ``numpy.random`` and is therefore reproducible via
  ``numpy.random.seed``, unlike a fully-jax sampler would be.
* ``gridsize`` is rounded up to the next power of two (with a minimum of
  ``GRIDSIZE_BUCKET_MIN``) before being used as a jit static argument. Since
  the number of distinct grid sizes actually used across a run of many
  bundles is then small (a handful of power-of-two "buckets" rather than one
  distinct value per bundle), the number of JIT compilations is bounded
  instead of scaling with the number of bundles.

Whether this module is actually faster than the plain numpy engine depends
strongly on the number of spectral lines being evaluated:

* For a large line list (e.g. the ~184-line Ar16+ spectrum used by the W7-X
  ``ar16_voigt`` wavelength distribution), JIT dispatch overhead is small
  relative to the Faddeeva evaluation, and this module has been benchmarked
  at roughly 2x faster than plain numpy.
* For a small line list (a handful of lines, as in a typical hand-specified
  ``multi_voigt`` configuration), JIT dispatch overhead dominates and plain
  numpy remains faster. This matches the original (small-spectrum) F004
  benchmark.

There is no automatic selection between the two implementations; the caller
decides which to use.
"""

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from xicsrt.tools import xicsrt_faddeeva_jax

# Minimum power-of-two grid size. Numpy's own auto-computed gridsize floor is
# 100 (see xicsrt_voigt_multi.multi_voigt_cdf_tab); 128 is the next power of
# two above that.
GRIDSIZE_BUCKET_MIN = 128


def _bucket_gridsize(gridsize, minimum=GRIDSIZE_BUCKET_MIN):
    """
    Round `gridsize` up to the next power of four (with a floor of `minimum`).

    This function was AI generated using Claude (Opus 4.8, Sonnet 4.6).

    Rounding to a small set of discrete "bucket" sizes bounds the number of
    distinct shapes seen by `jax.jit` across many calls with different
    (data-dependent) gridsize values, avoiding per-call recompilation.
    """
    n = max(int(minimum), int(gridsize))
    bucket = int(2 ** np.ceil(np.log2(n)))
    #print(f"gridsize: {gridsize}, bucket: {bucket}")
    return bucket


@functools.partial(jax.jit, static_argnames=('gridsize', 'N'))
def _voigt_cdf_core(
        wave_min, wave_max, gridsize, line_locations, line_intensities,
        sigmas, gammas, N):
    """
    Jit core: build the wavelength grid, evaluate the summed Voigt spectrum,
    and accumulate the (unnormalized) CDF.

    This function was AI generated using Claude (Opus 4.8, Sonnet 4.6).

    `gridsize` and `N` are static (Python int) arguments, so this function is
    retraced only when one of them changes value; `wave_min`/`wave_max` and
    the line parameter arrays are traced (dynamic) arguments.
    """
    bounds = jnp.linspace(wave_min, wave_max, gridsize + 1)
    cdf_x = (bounds[:-1] + bounds[1:]) / 2

    profiles = xicsrt_faddeeva_jax.voigt_profile(
        cdf_x[None, :],
        line_locations[:, None],
        line_intensities[:, None],
        sigmas[:, None],
        gammas[:, None],
        N=N,
    )
    pdf = profiles.sum(axis=0)

    pdf_dx = pdf * (bounds[1:] - bounds[:-1])
    cdf = jnp.cumsum(pdf_dx)
    cdf = cdf / cdf[-1]

    return bounds[1:], cdf, pdf


def multi_voigt(
        x, line_locations, line_intensities, sigmas, gammas,
        N=xicsrt_faddeeva_jax.DEFAULT_N):
    """
    Evaluates the summed spectrum from multiple Voigt profiles.

    JAX-accelerated drop-in equivalent of
    :func:`xicsrt.tools.xicsrt_voigt_multi.multi_voigt`. See that function
    for full parameter documentation; the two are numerically equivalent.

    Returns
    -------
    y : numpy.ndarray
        Sum of all Voigt profiles evaluated on the wavelength grid.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    line_locations = jnp.asarray(line_locations, dtype=jnp.float64)
    line_intensities = jnp.asarray(line_intensities, dtype=jnp.float64)
    sigmas = jnp.asarray(sigmas, dtype=jnp.float64)
    gammas = jnp.asarray(gammas, dtype=jnp.float64)

    profiles = xicsrt_faddeeva_jax.voigt_profile(
        x[None, :],
        line_locations[:, None],
        line_intensities[:, None],
        sigmas[:, None],
        gammas[:, None],
        N=N,
    )
    y = profiles.sum(axis=0)

    return np.asarray(y)


def multi_voigt_cdf_tab(
        line_locations, line_intensities, sigmas, gammas, gridsize=None,
        cutoff=None, N=xicsrt_faddeeva_jax.DEFAULT_N):
    """
    Numerical CDF table for a spectrum made from multiple Voigt profiles.

    JAX-accelerated drop-in equivalent of
    :func:`xicsrt.tools.xicsrt_voigt_multi.multi_voigt_cdf_tab`. See that
    function for full parameter documentation and for the (unmodified) host-
    side domain/gridsize sizing logic duplicated here.

    The only behavioral difference from the plain-numpy version is that,
    when `gridsize` is not given explicitly, the auto-computed gridsize is
    rounded up to the next power of two (see module docstring) before the
    spectrum is evaluated. This slightly widens the grid spacing relative to
    the plain-numpy version (never coarser than a factor of 2), which is
    negligible for CDF/sampling purposes given the already-conservative
    `min_spacing` heuristic.

    Returns
    -------
    bounds[1:], cdf, pdf : numpy.ndarray
        Same as :func:`xicsrt.tools.xicsrt_voigt_multi.multi_voigt_cdf_tab`.
    """
    if cutoff is None:
        cutoff = 1e-4

    # Set min and max gridsize to match the bucketing strategy.
    # The max gridsize will limit the ability to accurately model narrow peaks.
    # Currently set to 2^13 = 8192
    gridsize_min = 128
    gridsize_max = int(2**13)

    line_locations = np.asarray(line_locations)
    line_intensities = np.asarray(line_intensities)
    sigmas = np.asarray(sigmas)
    gammas = np.asarray(gammas)

    sigma_max = np.max(sigmas)
    gamma_max = np.max(gammas)

    sigma_min = np.min(sigmas)
    gamma_min = np.min(gammas)

    fraction = 0.5

    gauss_hwfm = np.sqrt(2.0 * np.log(1.0 / fraction)) * sigma_min
    lorentz_hwfm = gamma_min * np.sqrt(1.0 / fraction - 1.0)

    hwfm_min = np.sqrt(gauss_hwfm**2 + lorentz_hwfm**2)

    min_spacing = hwfm_min / 5.0

    lorentz_cutoff = gamma_max * np.sqrt(1.0 / cutoff - 1.0)
    gauss_cutoff = np.sqrt(-1 * sigma_max**2 * 2 * np.log(cutoff * sigma_max * np.sqrt(2 * np.pi)))
    value_cutoff = max(lorentz_cutoff, gauss_cutoff)

    wave_min = np.min(line_locations) - value_cutoff
    wave_max = np.max(line_locations) + value_cutoff

    if gridsize is None:
        domain_width = wave_max - wave_min
        gridsize = max(gridsize_min, int(np.ceil(domain_width / min_spacing)))
        if gridsize > gridsize_max:
            gridsize = gridsize_max
            warnings.warn(f"Warning: Computed gridsize is larger than the maximum ({gridsize_max}), truncating.")

    # Bucket to a power of two so that jax.jit sees only a small number of
    # distinct shapes across many calls with different data-dependent
    # gridsize values (see module docstring).
    gridsize = _bucket_gridsize(gridsize)

    bounds_right, cdf, pdf = _voigt_cdf_core(
        float(wave_min), float(wave_max), gridsize,
        jnp.asarray(line_locations, dtype=jnp.float64),
        jnp.asarray(line_intensities, dtype=jnp.float64),
        jnp.asarray(sigmas, dtype=jnp.float64),
        jnp.asarray(gammas, dtype=jnp.float64),
        int(N),
    )

    bounds_right = np.asarray(bounds_right)
    cdf = np.asarray(cdf)
    pdf = np.asarray(pdf)

    if (np.max(cdf) < 0.99):
        raise Exception('Multiline Voigt CDF calculation domain too small.')

    return bounds_right, cdf, pdf


def multi_voigt_random(
        line_locations, line_intensities, sigmas, gammas, size,
        gridsize=None, cutoff=None, N=xicsrt_faddeeva_jax.DEFAULT_N):
    """
    Draw random wavelength samples from a multiline Voigt spectrum.

    JAX-accelerated drop-in equivalent of
    :func:`xicsrt.tools.xicsrt_voigt_multi.multi_voigt_random`. See that
    function for full parameter documentation; the two are statistically
    equivalent (both build the CDF table and then draw from
    ``numpy.random.uniform`` followed by ``numpy.interp``, so results are
    reproducible via ``numpy.random.seed`` exactly as in the plain-numpy
    version).

    Returns
    -------
    random_x : numpy.ndarray
        Randomly sampled wavelengths drawn from the multiline Voigt spectrum.
    """
    cdf_x, cdf, pdf = multi_voigt_cdf_tab(
        line_locations,
        line_intensities,
        sigmas,
        gammas,
        gridsize=gridsize,
        cutoff=cutoff,
        N=N,
    )

    random_y = np.random.uniform(np.min(cdf), np.max(cdf), size)
    random_x = np.interp(random_y, cdf, cdf_x)

    return random_x
