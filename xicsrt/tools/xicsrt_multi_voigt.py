#!/usr/bin/env python
# coding: utf-8

# This 'multi_voigt' code uses the foundation of 'xicsrt_voigt' and extends the functions to be used on multiple lines.
# 
# We cannot directly use those functions from 'xicsrt_voigt' because it is based on the assumption that there is only one Voigt profile, which is centered at zero. We rewrite the functions to be compatible with multiple Voigt profiles/lines.


import numpy as np
from scipy.special import wofz

# Setup the module path.
import sys
sys.path.append(r"C:\Users\leila\Documents\Visual Studio\pppl_xics_2026\xicsrt")

def voigt(x, intensity=None, location=None, sigma=None, gamma=None):
    """
    Evaluates one Voigt profile. 

    Parameters:
        x : wavelength grid
        intensity : strength/area scaling of the line
        location : center of the wavelength line
        sigma : Gaussian width
        gamma : Lorentzian width

    Returns: 
        y: intensity of this one Voigt line at every x value
    """

    z = (x - location + 1j*gamma)/np.sqrt(2)/sigma
    y = wofz(z).real/np.sqrt(2*np.pi)/sigma * intensity

    return y



def multi_voigt(x, line_locations, line_intensities, sigmas, gammas):
    """
    Evaluates the summed spectrum from multiple Voigt profiles. 

    Parameters: 
        x : wavelength grid
        line_locations : center wavelength of each spectral line
        line_intensities : intensity (area) of each spectral line
        sigmas : Gaussian width of each spectral line
        gammas : Lorentzian width of each spectral line

    Returns: 
        y : sum of all Voigt profiles evaluated on the wavelength gird

    """

    x = np.asarray(x)
    y = np.zeros_like(x, dtype = float)

    for location, intensity, sigma, gamma, in zip(
            line_locations, 
            line_intensities, 
            sigmas, 
            gammas):
        y += voigt(
            x, 
            intensity = intensity, 
            location = location, 
            sigma = sigma, 
            gamma = gamma,
        )

    return y



def multi_voigt_cdf_tab(line_locations, line_intensities, sigmas, gammas, gridsize=None, cutoff=None):
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

    # The 'voigt_cdf_tab()' function scheme worked will with a minimum of 100 points.
    # With that function it was possible to go as low as 50 points, but accuracy was not great.
    gridsize_min = 100

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
        # adding a sanity check to see if the computed gridsize is too large
        if gridsize > 1_000_000:
            print(f"Warning: Computed gridsize ({gridsize}) is very large.")

    bounds = np.linspace(wave_min, wave_max, gridsize + 1)

    # 'cdf_x' is the center of each wavelength bin
    cdf_x = (bounds[:-1] + bounds[1:]) / 2

    # Evaluating the summed multiline Voigt spectrum
    pdf = np.zeros_like(cdf_x)
    pdf = multi_voigt(cdf_x, line_locations, line_intensities, sigmas, gammas,)

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


def multi_voigt_random(line_locations, line_intensities, sigmas, gammas, size, gridsize=None, cutoff=None):
    """
    Draw random wavelength samples from a multiline Voigt spectrum.

    Parameters:
        line_locations : center wavelength of each Voigt line
        line_intensities : intensity of each Voigt line
        sigmas : Gaussian width of each Voigt line
        gammas : Lorentzian width of each Voigt line
        size : number of random wavelength samples to generate
        gridsize : number of wavelength grid points used to build the CDF
        cutoff : relative intensity cutoff used to determine the wavelength domain of CDF

    Returns:
        random_x : randomly sampled wavelengths drawn from the multiline Voigt spectrum

    Notes: 
        Is linear interpolation sufficient for this?
        Should we use quadratic interpolation instead?
        Keep in mind that linear interpolation is faster, but quadratic may behave better/worse and slower
    """

    cdf_x, cdf, spectrum = multi_voigt_cdf_tab(
        line_locations,
        line_intensities,
        sigmas,
        gammas,
        gridsize=gridsize,
        cutoff=cutoff,
    )

    # Generate uniformly distributed random probability values
    random_y = np.random.uniform(np.min(cdf), np.max(cdf), size)

    # uses the inverse CDF to convert the probabilities into wavelengths
    random_x = np.interp(random_y, cdf, cdf_x)

    return random_x

