# -*- coding: utf-8 -*-
"""
Tests for the Voigt profile tools.

This file includes AI generated code using Claude (Opus 4.8, Fable 5).

These tests cover:
  * accuracy of the jax-friendly Weideman Faddeeva approximation against
    ``scipy.special.wofz``,
  * agreement of the new ``voigt`` wrapper with the retained ``voigt_wofz``,
  * broadcast correctness of ``multi_voigt`` vs a sum of single lines,
  * CDF properties and multi/single-line consistency,
  * statistical sanity of the random sampler.
"""

import numpy as np
import pytest
from scipy.special import wofz

from xicsrt.tools import xicsrt_faddeeva
from xicsrt.tools import xicsrt_voigt
from xicsrt.tools import xicsrt_voigt_multi


# Maximum absolute error in w(z) for each number of terms N.  These are
# comfortably above the empirically observed errors but tight enough to catch
# a real regression.
WOFZ_TOL = {16: 1e-6, 24: 1e-9, 32: 1e-12}


@pytest.mark.parametrize("N", [16, 24, 32])
def test_wofz_weideman_accuracy(N):
    """The Weideman approximation matches scipy.special.wofz."""
    rng = np.random.default_rng(1)
    re = rng.uniform(-8, 8, 5000)
    im = rng.uniform(0.0, 4.0, 5000)
    z = re + 1j * im

    ref = wofz(z)
    approx = xicsrt_faddeeva.wofz_weideman(z, N=N)

    err = np.max(np.abs(approx - ref))
    assert err < WOFZ_TOL[N], f"N={N}: max abs err {err:.3e}"


@pytest.mark.parametrize("N", [16, 24, 32])
def test_voigt_profile_accuracy(N):
    """voigt_profile matches a direct scipy wofz Voigt evaluation."""
    x = np.linspace(-8, 8, 2000)
    location, intensity, sigma, gamma = 0.7, 1.3, 0.25, 0.15

    z = (x - location + 1j * gamma) / np.sqrt(2) / sigma
    ref = wofz(z).real / np.sqrt(2 * np.pi) / sigma * intensity

    approx = xicsrt_faddeeva.voigt_profile(
        x, location, intensity, sigma, gamma, N=N)

    err = np.max(np.abs(approx - ref))
    assert err < WOFZ_TOL[N] * intensity / sigma


def test_voigt_matches_voigt_wofz():
    """The new voigt() wrapper agrees with the retained voigt_wofz()."""
    x = np.linspace(-5, 6, 1500)
    kw = dict(intensity=2.0, location=1.0, sigma=0.3, gamma=0.1)

    new = xicsrt_voigt.voigt(x, **kw)
    old = xicsrt_voigt.voigt_wofz(x, **kw)

    assert np.allclose(new, old, atol=1e-6)


def test_multi_voigt_equals_sum_of_singles():
    """multi_voigt equals the sum of individual voigt() evaluations."""
    x = np.linspace(-10, 10, 3000)
    loc = np.array([-2.0, 0.5, 3.0])
    inten = np.array([1.0, 2.0, 0.5])
    sig = np.array([0.2, 0.3, 0.15])
    gam = np.array([0.1, 0.05, 0.2])

    summed = np.zeros_like(x)
    for ii in range(len(loc)):
        summed += xicsrt_voigt.voigt(
            x, intensity=inten[ii], location=loc[ii],
            sigma=sig[ii], gamma=gam[ii])

    multi = xicsrt_voigt_multi.multi_voigt(x, loc, inten, sig, gam)

    assert np.allclose(multi, summed, atol=1e-12)


def test_multi_voigt_cdf_properties():
    """The CDF is monotonic non-decreasing and normalized to 1."""
    loc = np.array([-1.0, 1.5])
    inten = np.array([1.0, 0.7])
    sig = np.array([0.25, 0.2])
    gam = np.array([0.1, 0.15])

    cdf_x, cdf, pdf = xicsrt_voigt_multi.multi_voigt_cdf_tab(
        loc, inten, sig, gam)

    assert np.all(np.diff(cdf) >= 0), "CDF must be non-decreasing"
    assert cdf[-1] == pytest.approx(1.0)
    assert np.all(pdf >= 0), "PDF must be non-negative"
    assert cdf_x.shape == cdf.shape == pdf.shape


def test_multi_voigt_random_statistics():
    """Random samples reproduce the analytic spectrum shape.

    The direct sampler draws exact (untruncated) Lorentzian tails, so
    neither ``mean`` nor ``std`` converge usefully.  Instead we verify
    (a) the sample median sits at the symmetry point of the two identical
    lines and (b) a histogram of the samples matches the analytic PDF.
    """
    loc = np.array([-1.0, 2.0])
    inten = np.array([1.0, 1.0])
    sig = np.array([0.3, 0.3])
    gam = np.array([0.1, 0.1])

    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        samples = xicsrt_voigt_multi.multi_voigt_random(
            loc, inten, sig, gam, size=400000)
    finally:
        np.random.set_state(rng_state)

    # By symmetry the median sits halfway between the two identical lines.
    # (The mean of a Cauchy-tailed sample does not converge.)
    assert np.median(samples) == pytest.approx(np.mean(loc), abs=0.02)

    # Histogram of the samples should match the normalized analytic PDF.
    edges = np.linspace(-4, 5, 60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(samples, bins=edges, density=True)

    pdf = xicsrt_voigt_multi.multi_voigt(centers, loc, inten, sig, gam)
    pdf = pdf / np.trapezoid(pdf, centers)

    # Compare where the PDF is appreciable to avoid noisy empty-tail bins.
    mask = pdf > 0.02 * pdf.max()
    assert np.allclose(hist[mask], pdf[mask], atol=0.1 * pdf.max())


def test_default_N_is_16():
    """The tunable default should be N=16 as specified for F001."""
    assert xicsrt_faddeeva.DEFAULT_N == 16
