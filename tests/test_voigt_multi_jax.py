# -*- coding: utf-8 -*-
"""
Tests for the JAX-accelerated multiline Voigt tools.

This file includes AI generated code using Claude (Opus 4.8, Sonnet 4.6).

These tests cover:
  * accuracy of the JAX Weideman Faddeeva approximation against
    ``scipy.special.wofz``,
  * agreement of ``xicsrt_voigt_multi_jax`` with the plain-numpy
    ``xicsrt_voigt_multi`` (CDF/PDF, exact-gridsize case),
  * gridsize bucketing behavior (no change in results at a bucket boundary),
  * statistical sanity of the random sampler.

All tests are skipped if ``jax`` is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from scipy.special import wofz  # noqa: E402

from xicsrt.tools import xicsrt_faddeeva_jax  # noqa: E402
from xicsrt.tools import xicsrt_voigt_multi  # noqa: E402
from xicsrt.tools import xicsrt_voigt_multi_jax  # noqa: E402


WOFZ_TOL = {16: 1e-6, 24: 1e-9, 32: 1e-12}


@pytest.mark.parametrize("N", [16, 24, 32])
def test_wofz_weideman_jax_accuracy(N):
    """The JAX Weideman approximation matches scipy.special.wofz."""
    rng = np.random.default_rng(1)
    re = rng.uniform(-8, 8, 5000)
    im = rng.uniform(0.0, 4.0, 5000)
    z = re + 1j * im

    ref = wofz(z)
    approx = np.asarray(xicsrt_faddeeva_jax.wofz_weideman(z, N=N))

    err = np.max(np.abs(approx - ref))
    assert err < WOFZ_TOL[N], f"N={N}: max abs err {err:.3e}"


def test_voigt_profile_jax_matches_numpy():
    """xicsrt_faddeeva_jax.voigt_profile matches the plain-numpy kernel."""
    from xicsrt.tools import xicsrt_faddeeva

    x = np.linspace(-8, 8, 2000)
    location, intensity, sigma, gamma = 0.7, 1.3, 0.25, 0.15

    ref = xicsrt_faddeeva.voigt_profile(x, location, intensity, sigma, gamma)
    approx = np.asarray(
        xicsrt_faddeeva_jax.voigt_profile(x, location, intensity, sigma, gamma))

    assert np.allclose(approx, ref, atol=1e-9)


def test_multi_voigt_jax_matches_numpy():
    """multi_voigt (jax) matches multi_voigt (numpy) on the same grid."""
    x = np.linspace(-10, 10, 3000)
    loc = np.array([-2.0, 0.5, 3.0])
    inten = np.array([1.0, 2.0, 0.5])
    sig = np.array([0.2, 0.3, 0.15])
    gam = np.array([0.1, 0.05, 0.2])

    ref = xicsrt_voigt_multi.multi_voigt(x, loc, inten, sig, gam)
    approx = xicsrt_voigt_multi_jax.multi_voigt(x, loc, inten, sig, gam)

    assert np.allclose(approx, ref, atol=1e-9)


def test_multi_voigt_cdf_tab_jax_matches_numpy_at_exact_gridsize():
    """
    With an explicit (already-power-of-two) gridsize, the jax CDF table
    matches the plain-numpy CDF table evaluated on the identical grid.
    """
    loc = np.array([-1.0, 1.5])
    inten = np.array([1.0, 0.7])
    sig = np.array([0.25, 0.2])
    gam = np.array([0.1, 0.15])

    gridsize = 2048
    assert xicsrt_voigt_multi_jax._bucket_gridsize(gridsize) == gridsize

    cdf_x_ref, cdf_ref, pdf_ref = xicsrt_voigt_multi.multi_voigt_cdf_tab(
        loc, inten, sig, gam, gridsize=gridsize)
    cdf_x_jax, cdf_jax, pdf_jax = xicsrt_voigt_multi_jax.multi_voigt_cdf_tab(
        loc, inten, sig, gam, gridsize=gridsize)

    assert np.allclose(cdf_x_jax, cdf_x_ref, atol=1e-9)
    assert np.allclose(cdf_jax, cdf_ref, atol=1e-8)
    assert np.allclose(pdf_jax, pdf_ref, atol=1e-8)


def test_multi_voigt_cdf_properties_jax():
    """The jax-built CDF is monotonic non-decreasing and normalized to 1."""
    loc = np.array([-1.0, 1.5])
    inten = np.array([1.0, 0.7])
    sig = np.array([0.25, 0.2])
    gam = np.array([0.1, 0.15])

    cdf_x, cdf, pdf = xicsrt_voigt_multi_jax.multi_voigt_cdf_tab(
        loc, inten, sig, gam)

    assert np.all(np.diff(cdf) >= 0), "CDF must be non-decreasing"
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(pdf >= 0), "PDF must be non-negative"
    assert cdf_x.shape == cdf.shape == pdf.shape


@pytest.mark.parametrize("minimum,requested", [
    (128, 100),   # below floor -> bucketed up to minimum
    (128, 128),   # exact power of two -> unchanged
    (128, 129),   # just above a power of two -> rounds up to next bucket
    (128, 4096),  # exact power of two (large) -> unchanged
    (128, 4097),  # just above a large power of two -> rounds up
])
def test_bucket_gridsize(minimum, requested):
    """_bucket_gridsize rounds up to the next power of two, floored at minimum."""
    result = xicsrt_voigt_multi_jax._bucket_gridsize(requested, minimum=minimum)
    assert result >= max(minimum, requested)
    assert result >= minimum
    # result must itself be a power of two
    assert (result & (result - 1)) == 0
    # result must be the *smallest* qualifying power of two
    assert result // 2 < max(minimum, requested) or result == minimum


def test_bucket_gridsize_boundary_gives_consistent_results():
    """
    Two auto-computed gridsize requests that land in the same power-of-two
    bucket produce CDF tables on the identical (bucketed) grid, i.e. the
    bucketing is deterministic and does not depend on the exact requested
    gridsize once bucketed.
    """
    loc = np.array([-1.0, 1.5])
    inten = np.array([1.0, 0.7])
    sig = np.array([0.25, 0.2])
    gam = np.array([0.1, 0.15])

    # 1025 and 2000 both bucket to 2048.
    assert xicsrt_voigt_multi_jax._bucket_gridsize(1025) == 2048
    assert xicsrt_voigt_multi_jax._bucket_gridsize(2000) == 2048

    cdf_x_a, cdf_a, pdf_a = xicsrt_voigt_multi_jax.multi_voigt_cdf_tab(
        loc, inten, sig, gam, gridsize=1025)
    cdf_x_b, cdf_b, pdf_b = xicsrt_voigt_multi_jax.multi_voigt_cdf_tab(
        loc, inten, sig, gam, gridsize=2000)

    assert cdf_x_a.shape == cdf_x_b.shape == (2048,)
    assert np.allclose(cdf_x_a, cdf_x_b)
    assert np.allclose(cdf_a, cdf_b)
    assert np.allclose(pdf_a, pdf_b)


def test_multi_voigt_random_statistics_jax():
    """
    Random samples from the jax sampler reproduce the analytic spectrum
    shape, exactly as tested for the plain-numpy sampler in test_voigt.py.
    """
    loc = np.array([-1.0, 2.0])
    inten = np.array([1.0, 1.0])
    sig = np.array([0.3, 0.3])
    gam = np.array([0.1, 0.1])

    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        samples = xicsrt_voigt_multi_jax.multi_voigt_random(
            loc, inten, sig, gam, size=400000, gridsize=4000)
    finally:
        np.random.set_state(rng_state)

    assert np.mean(samples) == pytest.approx(np.mean(loc), abs=0.02)

    edges = np.linspace(-4, 5, 60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(samples, bins=edges, density=True)

    pdf = xicsrt_voigt_multi_jax.multi_voigt(centers, loc, inten, sig, gam)
    pdf = pdf / np.trapezoid(pdf, centers)

    mask = pdf > 0.02 * pdf.max()
    assert np.allclose(hist[mask], pdf[mask], atol=0.1 * pdf.max())


def test_multi_voigt_random_jax_reproducible_with_numpy_seed():
    """
    multi_voigt_random (jax) draws its random numbers from numpy.random, so
    it is reproducible via numpy.random.seed just like the plain-numpy
    version (unlike a fully-jax sampler seeded from a jax.random key).
    """
    loc = np.array([-1.0, 2.0])
    inten = np.array([1.0, 1.0])
    sig = np.array([0.3, 0.3])
    gam = np.array([0.1, 0.1])

    rng_state = np.random.get_state()
    try:
        np.random.seed(7)
        samples_a = xicsrt_voigt_multi_jax.multi_voigt_random(
            loc, inten, sig, gam, size=100, gridsize=2048)
        np.random.seed(7)
        samples_b = xicsrt_voigt_multi_jax.multi_voigt_random(
            loc, inten, sig, gam, size=100, gridsize=2048)
    finally:
        np.random.set_state(rng_state)

    assert np.array_equal(samples_a, samples_b)
