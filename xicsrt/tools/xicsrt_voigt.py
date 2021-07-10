# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A set of routines for related to Voigt distributions.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import wofz

def voigt(
        x
        ,intensity=None
        ,location=None
        ,sigma=None
        ,gamma=None):
    """
    The Voigt function is also the real part of  w(z) = exp(-z^2) erfc(iz), 
    the complex probability function, which is also known as the Faddeeva 
    function. Scipy has implemented this function under the name wofz()
    """
    
    z = (x - location + 1j*gamma)/np.sqrt(2)/sigma
    y = wofz(z).real/np.sqrt(2*np.pi)/sigma * intensity
    return y


def voigt_cdf_tab(gamma, sigma, gridsize=None, cutoff=None):

    # This is a numerical method to calculate the cumulative distribution fuction
    # for a voigt profile.  This works reasonably well, but is limited both by
    # the sampling resolution, and the chosen bounds.
    # 
    # In this case the CDF is calculated with a variable grid density to help
    # mitigate those effects.
    #
    # There are a couple of possibilities to speed this up:
    #  1. It may be possible to optimize the grid spacing by using
    #     a fuction that increases faster away from zero.
    #  2. The CDF is symetric so only calculation up to x=0 is needed.
    #  3. For some applicaiton I might be able to use a psudo-voigt
    #     calculation that may be faster than the wofz implementation
    #     (at the expense of accuracy.)
    

    if gridsize is None: gridsize = 1000
    if cutoff is None: cutoff = 1e-5
        
    # The current scheme works well with a minimum of 100 points.
    # It is possible to go as low as 50 points, but accuracy is not great.
    gridsize_min = 100

    fraction = 0.5
    gauss_hwfm = np.sqrt(2.0*np.log(1.0/fraction))*sigma
    lorentz_hwfm = gamma*np.sqrt(1.0/fraction - 1.0)
    # This is always larger than the voigt hwfm (half width at percentile).
    hwfm_max = np.sqrt(gauss_hwfm**2 + lorentz_hwfm**2)

    min_spacing = hwfm_max/5.0
    value = gridsize_min/2*min_spacing

    # Determine a cutoff value.
    lorentz_cutoff = gamma*np.sqrt(1.0/cutoff - 1.0)
    gauss_cutoff = np.sqrt(-1 * sigma**2 * 2 * np.log(cutoff*sigma*np.sqrt(2*np.pi)))
    value_cutoff = max(lorentz_cutoff, gauss_cutoff)
    base = np.exp(1/10 * np.log(value_cutoff/value))

    bounds = np.linspace(-value, value, gridsize+1)
    bounds = bounds*base**np.abs(bounds/value*10)
    cdf_x = (bounds[:-1]+bounds[1:])/2
        
    # We must used a properly normalized voigt here (intensity=1.0)
    cdf_y = voigt(
        cdf_x
        ,intensity=1.0
        ,location=0.0
        ,sigma=sigma
        ,gamma=gamma)

    cdf_ydx = (cdf_y*(bounds[1:]-bounds[:-1]))
    cdf = np.cumsum(cdf_ydx)

    # These checks are only useful if the user changes the number
    # of calculated points.
    if (np.sum((cdf > 0.25) & (cdf < 0.75)) < 3):
        raise Exception('Voight CDF calculation does not have enough resolution.')
    if (np.max(cdf) < 0.99):
        raise Exception('Voight CDF calculation domain too small.')
    
    return bounds[1:], cdf


def voigt_cdf_interp(gamma, sigma, gridsize=None):
    x, cdf = voigt_cdf_tab(gamma, sigma, gridsize)
    interp = interp1d(x, cdf, kind='quadratic')
    return interp


def voigt_invcdf_interp(gamma, sigma, gridsize=None):
    x, cdf = voigt_cdf_tab(gamma, sigma, gridsize)
    interp = interp1d(cdf, x, kind='quadratic')
    return interp


def voigt_cdf_numeric(x, gamma, sigma, gridsize=None):
    cdf_x, cdf = voigt_cdf_tab(gamma, sigma, gridsize)
    y = np.interp(x, cdf_x, cdf)
    return y


def voigt_invcdf_numeric(x, gamma, sigma, gridsize=None):
    cdf_x, cdf = voigt_cdf_tab(gamma, sigma, gridsize)
    y = np.interp(x, cdf, cdf_x, left=-np.inf, right=np.inf)
    return y


def voigt_random(gamma, sigma, size, **kwargs):
    """
    Draw random samples from a Voigt distribution.
    
    The tails of the distribution will be clipped;
    the clipping level can be adjusted with the cutoff keyword.
    The default values is 1e-5.
    """
    cdf_x, cdf = voigt_cdf_tab(gamma, sigma, **kwargs)
    random_y = np.random.uniform(np.min(cdf), np.max(cdf), size)
    random_x = np.interp(random_y, cdf, cdf_x)
    return random_x
    
