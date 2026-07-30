#!/usr/bin/env python
# coding: utf-8

# This 'multi_voigt' code extends the single-line Voigt evaluation to a sum of
# multiple Voigt profiles at arbitrary line centers.
#
# Both this module and 'xicsrt_voigt' now share a single, jax-friendly Voigt
# kernel defined in 'xicsrt_faddeeva.voigt_profile'.  That kernel is fully
# broadcastable, so the multi-line spectrum is evaluated as a single vectorized
# operation (no per-line Python loop).
#
# This file includes AI generated code using Claude (Opus 4.8, Sonnet 4.6).


import numpy as np
import warnings

from xicsrt.tools import xicsrt_faddeeva


def multi_voigt(x, line_locations, line_intensities, sigmas, gammas,
                N=xicsrt_faddeeva.DEFAULT_N):
    """
    Evaluates the summed spectrum from multiple Voigt profiles.

    This is evaluated as a single vectorized (broadcast) call into the shared
    :func:`xicsrt.tools.xicsrt_faddeeva.voigt_profile` kernel, then summed over
    the line axis.

    Parameters: 
        x : wavelength grid
        line_locations : center wavelength of each spectral line
        line_intensities : intensity (area) of each spectral line
        sigmas : Gaussian width of each spectral line
        gammas : Lorentzian width of each spectral line
        N : number of terms in the Weideman approximation

    Returns: 
        y : sum of all Voigt profiles evaluated on the wavelength gird

    """

    x = np.asarray(x, dtype=float)
    line_locations = np.asarray(line_locations, dtype=float)
    line_intensities = np.asarray(line_intensities, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    # Broadcast the (n_lines,) line parameters against the (n_grid,) grid to
    # form a (n_lines, n_grid) array, then sum over the line axis.
    profiles = xicsrt_faddeeva.voigt_profile(
        x[None, :],
        line_locations[:, None],
        line_intensities[:, None],
        sigmas[:, None],
        gammas[:, None],
        N=N,
    )
    y = profiles.sum(axis=0)

    return y



def multi_voigt_cdf_tab(line_locations, line_intensities, sigmas, gammas, gridsize=None, cutoff=None, N=xicsrt_faddeeva.DEFAULT_N):
    """
    Numerical CDF table for a spectrum made from multiple Voight profiles.
    This follows the structure of 'voigt_cdf_tab()': 
        1. Create wavelength-bin bounds.
        2. Evaluate the spectral distribution on the grid.
        3. Multiply the bin width to approximate the area.
        4. Cumulatively sum those areas to create the CDF.
        5. Normalize the CDF so that it ends at 1.
    This function builds one CDF for the sum of many Voigt lines.
    """

    if cutoff is None: 
        cutoff = 1e-4

    # Set min and max gridsize.
    # The min is set based on a single voigt, and also to match the jax version.
    # The max will limit the ability to accurately model very narrow peaks.
    # Currently set to 2^13 = 8192
    gridsize_min = 128
    gridsize_max = int(2**13)

    # Converts inputs to Numpy arrays 
    line_locations = np.asarray(line_locations)
    line_intensities = np.asarray(line_intensities)
    sigmas = np.asarray(sigmas)
    gammas = np.asarray(gammas)

    # Using the largest sigma and gamma value for determining lorentz and gauss cutoffs
    sigma_max = np.max(sigmas)
    gamma_max = np.max(gammas)

    # Using smallest sigma and gamma value for determining 'min_spacing' and 'value'
    sigma_min = np.min(sigmas)
    gamma_min = np.min(gammas)

    fraction = 0.5

    # Gaussian and Lorentzian half-width estimates copied from original function
    gauss_hwfm = np.sqrt(2.0 * np.log(1.0 / fraction)) * sigma_min
    lorentz_hwfm = gamma_min * np.sqrt(1.0 / fraction - 1.0)

    # Estimate the half-width of the narrowest Voigt profile
    hwfm_min = np.sqrt(gauss_hwfm**2 + lorentz_hwfm**2)

    # 'min_spacing' depends on the smallest sigma and gamma.
    min_spacing = hwfm_min / 5.0

    # Determine a cutoff value using max sigma and gamma.
    lorentz_cutoff = gamma_max * np.sqrt(1.0 / cutoff - 1.0)
    gauss_cutoff = np.sqrt(-1 * sigma_max**2 * 2 * np.log(cutoff * sigma_max * np.sqrt(2 * np.pi)))
    value_cutoff = max(lorentz_cutoff, gauss_cutoff)

    # For multiple lines, we make one grid that covers all line centers.
    # Adds enough room on both sides for the line tails.
    wave_min = np.min(line_locations) - value_cutoff
    wave_max = np.max(line_locations) + value_cutoff

    # Instead of using a fixed grid size, we use 'min_spacing' to compute it 
    if gridsize is None: 
        domain_width = wave_max - wave_min
        gridsize = max(gridsize_min, int(np.ceil(domain_width / min_spacing)))
        if gridsize > gridsize_max:
            gridsize = gridsize_max
            warnings.warn(f"Warning: Computed gridsize is larger than the maximum ({gridsize_max}), truncating.")

    bounds = np.linspace(wave_min, wave_max, gridsize + 1)

    # 'cdf_x' is the center of each wavelength bin
    cdf_x = (bounds[:-1] + bounds[1:]) / 2

    # Evaluating the summed multiline Voigt spectrum
    pdf = multi_voigt(cdf_x, line_locations, line_intensities, sigmas, gammas, N=N)

    # Approximating the area in each wavelength bin.
    # Matches original function for rectangle-style CDF construction.
    pdf_dx = pdf * (bounds[1:] - bounds[:-1])

    # Cumulative sum of bin areas gives the CDF
    cdf = np.cumsum(pdf_dx)

    # Normalizing the CDF so that the total accumulated area equals 1.
    cdf = cdf / cdf[-1]

    if (np.max(cdf) < 0.99):
        raise Exception('Multiline Voigt CDF calculation domain too small.')

    # returns all three arrays: 
    # (1) bounds[1:] : righthand boundary of each wavelength bin
    # (2) cdf : cumulative distribution function
    # (3) pdf : summed multiline Voigt spectrum
    return bounds[1:], cdf, pdf


def multi_voigt_random(line_locations, line_intensities, sigmas, gammas, size):
    """
    Draw random wavelength samples from a multiline Voigt spectrum.

    A multiline Voigt spectrum is a mixture distribution: each sample is
    drawn from one line, chosen with probability proportional to that
    line's intensity, and each line's Voigt profile is sampled exactly as
    the sum of a Normal(0, sigma) and a Cauchy(0, gamma) variate centered
    at the line location. This direct-sampling approach is exact — unlike
    the tabulated inverse-CDF method (see :func:`multi_voigt_cdf_tab`)
    there is no domain truncation and no interpolation error.

    Parameters
    ----------
    line_locations : array_like, shape (n_lines,)
        Center wavelength of each Voigt line.
    line_intensities : array_like, shape (n_lines,)
        Intensity (area) of each Voigt line. Need not be normalized.
    sigmas : array_like, shape (n_lines,)
        Gaussian standard deviation of each Voigt line.
    gammas : array_like, shape (n_lines,)
        Lorentzian half-width-at-half-max of each Voigt line.
    size : int
        Number of random wavelength samples to generate.

    Returns
    -------
    numpy.ndarray
        Randomly sampled wavelengths, shape (size,).

    Notes
    -----
    Draws from the global ``numpy.random`` state. Results are statistically
    identical to, but not bit-identical with, the old CDF-table sampler for
    a given seed.

    This function was AI generated using Claude (Fable 5).
    """
    line_locations = np.asarray(line_locations, dtype=float)
    line_intensities = np.asarray(line_intensities, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    # Choose a line for each sample with probability proportional to the
    # line intensities (mixture weights).
    cum = np.cumsum(line_intensities)
    cum /= cum[-1]
    line_index = np.searchsorted(cum, np.random.uniform(0.0, 1.0, size))

    # Draw the Voigt variate for each sample's chosen line:
    # location + Normal(0, sigma) + Cauchy(0, gamma).
    random_x = line_locations[line_index]
    random_x += np.random.normal(0.0, 1.0, size) * sigmas[line_index]
    random_x += np.random.standard_cauchy(size) * gammas[line_index]

    return random_x


def multi_voigt_random_batched(
        line_locations, line_intensities, sigmas, gammas, bundle_index):
    """
    Draw one wavelength per ray from per-bundle multiline Voigt spectra.

    This is the batched form of :func:`multi_voigt_random`: every bundle
    has its own set of line parameters (for example because the line
    intensities and Doppler widths depend on the local plasma temperature),
    and each ray samples from the spectrum of the bundle it belongs to.
    All rays from all bundles are sampled in a single vectorized call.

    The sampling is exact mixture sampling, identical in distribution to
    calling :func:`multi_voigt_random` once per bundle.

    Parameters
    ----------
    line_locations : array_like, shape (n_bundles, n_lines)
        Center wavelength of each Voigt line, per bundle.
    line_intensities : array_like, shape (n_bundles, n_lines)
        Intensity (area) of each Voigt line, per bundle. Need not be
        normalized; each bundle row is normalized independently.
    sigmas : array_like, shape (n_bundles, n_lines)
        Gaussian standard deviation of each Voigt line, per bundle.
    gammas : array_like, shape (n_bundles, n_lines)
        Lorentzian half-width-at-half-max of each Voigt line, per bundle.
    bundle_index : array_like, shape (n_rays,)
        For each ray, the index of the bundle (row) whose spectrum it
        samples from. Values must be in ``range(n_bundles)``.

    Returns
    -------
    numpy.ndarray
        Randomly sampled wavelengths, shape (n_rays,).

    Notes
    -----
    The per-ray line selection uses a single :func:`numpy.searchsorted`
    over a flattened, row-offset cumulative-intensity array, avoiding both
    a Python loop over bundles and an (n_rays, n_lines) temporary. Adding
    the row offset costs a few bits of floating-point resolution in the
    mixture weights (relative quantization ~1e-11 at 1e5 bundles), which
    is statistically undetectable at any achievable sample size.

    This function was AI generated using Claude (Fable 5).
    """
    line_locations = np.asarray(line_locations, dtype=float)
    line_intensities = np.asarray(line_intensities, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    bundle_index = np.asarray(bundle_index)

    n_bundles, n_lines = line_intensities.shape
    n_rays = bundle_index.shape[0]

    # Normalized cumulative mixture weights for every bundle row.
    cum = np.cumsum(line_intensities, axis=1)
    cum /= cum[:, -1:]

    # Row-offset trick: add 2*row to each row of the (0, 1] cumulative
    # weights so the flattened array is globally sorted, then search for
    # (uniform + 2*bundle_index) to select a line within each ray's own
    # bundle row with a single searchsorted call.
    offsets = 2.0 * np.arange(n_bundles)
    cum_flat = (cum + offsets[:, None]).ravel()
    keys = np.random.uniform(0.0, 1.0, n_rays) + offsets[bundle_index]
    line_index = np.searchsorted(cum_flat, keys) - bundle_index * n_lines

    # Draw the Voigt variate for each ray's chosen (bundle, line):
    # location + Normal(0, sigma) + Cauchy(0, gamma).
    flat_index = bundle_index * n_lines + line_index
    random_x = line_locations.ravel()[flat_index]
    random_x += np.random.normal(0.0, 1.0, n_rays) * sigmas.ravel()[flat_index]
    random_x += np.random.standard_cauchy(n_rays) * gammas.ravel()[flat_index]

    return random_x

