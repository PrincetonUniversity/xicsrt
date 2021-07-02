# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

A set of algorithms to generate vector distributions.

.. Note::
    The term 'spread' here denotes the angular range of emission. The term
    'divergence' is not used because I normally think of a divergence as a
    gaussian distributed probability distribution of angles. This type of
    distribution is available, but for generality and consistency 'spread'
    will be used throughout.
"""

import numpy as np


def vector_distribution(spread, number, name=None):
    """
    A convenience function to retrieve vector distributions by name.

    Parameters
    ----------
    spread : float or array [radians]
      Can be a scalar or an array. See individual distributions for format
      and definitions.

    number : int
      The number of vectors to generate.

    name : string ('isotropic')
      The name of the vector distribution. Available names:
      'isotropic', 'isotropic_xy', 'flat', 'flat_xy', 'gaussian'.

    Returns
    -------
    ndarray
      A numpy array of shape (number, 3) containing the generated unit vectors.
    """
    if name is None: name = 'isotropic'

    name = name.lower()
    if name == 'isotropic':
        func = vector_dist_isotropic
    elif name == 'isotropic_xy':
        func = vector_dist_isotropic_xy
    elif name == 'flat':
        func = vector_dist_flat
    elif name == 'flat_xy':
        func = vector_dist_flat_xy
    elif name == 'gaussian':
        func = vector_dist_gaussian
    else:
        raise Exception(f'Distribution "{name}" is not known.')

    return func(spread, number)

def solid_angle(spread, name=None):
    """
    A convenience function to retrieve solid angles that correspond to
    the various vector distributions.

    Units: [sr]
    """
    if name is None: name = 'isotropic'

    name = name.lower()
    if name == 'isotropic':
        func = solid_angle_isotropic
    elif name == 'isotropic_xy':
        func = solid_angle_isotropic_xy
    else:
        raise Exception(f'Solid angle calculation for "{name}" is not available.')

    return func(spread)

def vector_dist_isotropic(spread, number):
    """
    Return unit vectors from an isotropic (uniform spherical) distribution that
    fall within an angular spread (divergence) of theta.

    The ray cone is aligned along the z-axis.

    Parameters
    ----------
    spread : float [radians]
      The half-angle of the emitted cone of vectors (axis to edge).

    number : int
      The number of vectors to generate.

    Returns
    -------
    ndarray
        A numpy array of shape (number, 3) containing the generated unit vectors.
    """
    theta = _parse_spread_single(spread)

    z = np.random.uniform(np.cos(theta), 1, number)
    phi = np.random.uniform(0, 2*np.pi, number)

    output = np.empty((number, 3))
    output[:, 0] = np.sqrt(1 - z**2) * np.cos(phi)
    output[:, 1] = np.sqrt(1 - z**2) * np.sin(phi)
    output[:, 2] = z

    return output

def solid_angle_isotropic(spread):
    """
    Calculate the solid angle for the vector_dist_isotropic distribution.

    Parameters
    ----------
    spread : float [radians]
      The half-angle of cone of vectors (axis to edge).

    Returns
    -------
    solid_angle
      Units: [sr]
    """
    theta = _parse_spread_single(spread)
    solid_angle = 4 * np.pi * np.sin(theta[0]/2)**2
    return solid_angle

def vector_dist_isotropic_xy(spread, number):
    """
    Return random unit vectors from an isotroptic (uniform spherical) distribution
    that fall within a given x and y angular spread.

    The truncated-cone of vectors is aligned along the z-axis.

    .. Note::
        This routine repeatedly filters from a circular distribution, which is
        accurate but not efficent. Efficiency goes down for more unequal values
        of the x and y spread.

    .. Todo::
        Replace vector_dist_isotropic_xy with a more efficent calculation.
        A possible approach is to calculate the 2D Joint Cumulative
        Distribution Function for isotropic emission on a flat plane.

    Parameters
    ----------
    spread : float or array [radians]
      | The half-angles in the x and y directions that define the extent of the
      | truncated-cone of vectors. Spread can be contain either 1,2 or 4 values.
      |
      | s or [s]
      |   A single value that will be used for both the x and y directions.
      | [x, y]
      |   Two values values that will be used for the x and y directions.
      | [xmin, xmax, ymin, ymax]
      |   For values that define the asymmetric exent in x and y directions.
      |   Example: [-0.1, 0.1, -0.5, 0.5]

    number : int
      The number of vectors to generate.

    Returns
    -------
    ndarray
      A numpy array of shape (number, 3) containing the generated unit vectors.
    """
    theta = _parse_spread_xy(spread)

    # Determine the extent of the circular cone than we need.
    theta_xmax = np.max(np.abs(theta[0:2]))
    theta_ymax = np.max(np.abs(theta[2:]))
    theta_max = np.arcsin(np.sqrt(np.sin(theta_xmax) ** 2 + np.sin(theta_ymax) ** 2))

    # Generate and filter rays until we have the requested number.
    output = np.empty((number, 3))
    n_filled = 0
    while n_filled < number:
        vectors = vector_dist_isotropic(theta_max, number)
        mask = np.ones(number, dtype=bool)

        mask &= vectors[:, 0] / np.sqrt(vectors[:, 0] ** 2 + vectors[:, 2] ** 2) > np.sin(theta[0])
        mask &= vectors[:, 0] / np.sqrt(vectors[:, 0] ** 2 + vectors[:, 2] ** 2) <= np.sin(theta[1])
        mask &= vectors[:, 1] / np.sqrt(vectors[:, 1] ** 2 + vectors[:, 2] ** 2) > np.sin(theta[2])
        mask &= vectors[:, 1] / np.sqrt(vectors[:, 1] ** 2 + vectors[:, 2] ** 2) <= np.sin(theta[3])

        n_new = np.sum(mask)
        n_need = number - n_filled
        if n_new > (number - n_filled):
            n_new = n_need

        output[n_filled:n_filled + n_new, :] = vectors[mask, :][:n_new, :]
        n_filled += n_new

    return output

def solid_angle_isotropic_xy(spread):
    """
    Calculate the solid angle for the vector_dist_isotropic_xy distribution.

    Units: [sr]
    """
    theta = _parse_spread_xy(spread)
    solid_angle = (
        np.arcsin(np.abs(np.sin(theta[0])*np.sin(theta[2])))
        + np.arcsin(np.abs(np.sin(theta[0])*np.sin(theta[3])))
        + np.arcsin(np.abs(np.sin(theta[1])*np.sin(theta[2])))
        + np.arcsin(np.abs(np.sin(theta[1])*np.sin(theta[3])))
        )
    return solid_angle

def vector_dist_flat(spread, number):
    """
    Return unit vectors from an flat (uniform planar) distribution that
    fall within an angular spread.

    The ray cone is aligned along the z-axis.

    Parameters
    ----------
    spread : float [radians]
      The half-angle of the emitted cone of vectors (axis to edge).

    number : int
      The number of vectors to generate.

    Returns
    -------
    ndarray
      A numpy array of shape (number, 3) containing the generated unit vectors.
    """
    theta = _parse_spread_single(spread)

    r = np.sqrt(np.random.uniform(0, np.tan(theta), number))
    angle1 = np.random.uniform(0, 2*np.pi, number)

    angle0 = np.arctan(r)

    output = np.empty((number, 3))
    output[:, 0] = np.cos(angle1) * np.sin(angle0)
    output[:, 1] = np.sin(angle1) * np.sin(angle0)
    output[:, 2] = np.cos(angle0)

    return output

def vector_dist_flat_xy(spread, number):
    """
    Return random unit vectors from an flat (uniform planar) distribution
    that fall within a given x and y angular spread.

    The truncated-cone of vectors is aligned along the z-axis.

    .. Note::
        This distribution is identical to that used by the SHADOW raytracing
        code for both the 'flat' and 'uniform' distributions (as of 2021-01).

    Parameters
    ----------
    spread : float or array [radians]
      | The half-angles in the x and y directions that define the extent of the
      | truncated-cone of vectors. Spread can be contain either 1,2 or 4 values.
      |
      | s or [s]
      |   A single value that will be used for both the x and y directions.
      | [x, y]
      |   Two values values that will be used for the x and y directions.
      | [xmin, xmax, ymin, ymax]
      |   For values that define the asymmetric exent in x and y directions.
      |   Example: [-0.1, 0.1, -0.5, 0.5]

    number : int
      The number of vectors to generate.

    Returns
    -------
    ndarray
      A numpy array of shape (number, 3) containing the generated unit vectors.
    """
    theta = _parse_spread_xy(spread)
    range = np.tan(theta)

    x = np.random.uniform(range[0], range[1], number)
    y = np.random.uniform(range[2], range[3], number)

    angle0 = np.arctan(np.sqrt(x ** 2 + y ** 2))
    angle1 = np.arctan2(y, x)

    output = np.empty((number, 3))
    output[:, 0] = np.cos(angle1) * np.sin(angle0)
    output[:, 1] = np.sin(angle1) * np.sin(angle0)
    output[:, 2] = np.cos(angle0)

    return output

def vector_dist_gaussian(spread, number):
    """
    Create distribution of vectors with a Gaussian distribution of angles
    around the z-axis.

    .. Note::
        The Gaussian distribution here is of the angles. This means that when
        projected onto an x-y plane the distribution will not be Gaussian
        (except approximately for small angles).

    Parameters
    ----------
    spread : float [radians]
      The half-with-at-half-max (hwhm) of the Gaussian angular distribution.

    Returns
    -------
    ndarray
      A numpy array of shape (number, 3) containing the generated unit vectors.
    """

    theta = _parse_spread_single(spread)

    # Convert the angluar hwhm to sigma.
    sigma = theta / (2 * np.sqrt(2 * np.log(2)))
    # convert the angular sigma into a linear-displacement-from-vertical
    sigma_z = 1 - np.cos(sigma)
    # create the half-normal distribution of vertical displacement.
    dist = abs(np.random.normal(0, sigma_z, number))
    z = 1.0 - dist

    phi = np.random.uniform(0, 2 * np.pi, number)

    output = np.empty((number, 3))
    output[:, 0] = np.sqrt(1 - z**2) * np.cos(phi)
    output[:, 1] = np.sqrt(1 - z**2) * np.sin(phi)
    output[:, 2] = z

    return output

def _to_ndarray(spread):
    """
    Convert input value to an numpy array.
    Scalars will be transformed to a one element array.
    """
    if np.isscalar(spread):
        out = np.array([spread])
    elif isinstance(spread, np.ndarray):
        out = spread
    else:
        out = np.array(spread)
    return out

def _parse_spread_single(spread):
    """
    Parse the number of input values in spread and return a standard array.
    Use only for distribution with a single spread value.
    """
    spread = _to_ndarray(spread)
    if len(spread) != 1:
        raise Exception('Spread must be a scalar or one element array.')

    return spread

def _parse_spread_xy(spread):
    """
    Parse the number of input values in spread and return a standard array.
    Use only for assymetric xy distributions.
    """
    spread = _to_ndarray(spread)
    if len(spread) == 1:
        out = [-spread[0], spread[0], -spread[0], spread[0]]
    elif len(spread) == 2:
        out = [-spread[0], spread[0], -spread[1], spread[1]]
    elif len(spread) == 4:
        out = [spread[0], spread[1], spread[2], spread[3]]
    else:
        raise Exception('Spread must have 1, 2 or 3 elements. See docstring.')

    return out
