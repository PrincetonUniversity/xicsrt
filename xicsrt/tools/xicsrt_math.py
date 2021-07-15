# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

A set of mathematical utilities and vector convenience functions for XICSRT.
"""

import numpy as np


def distance_point_to_line(origin, normal, point):
    o = origin
    n = normal
    p = point
    t = np.dot(p - o, n) / np.dot(n, n)
    d = np.linalg.norm(np.outer(t, n) + o - p, axis=1)
    return d


def toarray_1d(a):
    """
    Convert the input to a ndarray with at least 1 dimension.
    This is similar to the numpy function atleast_1d, but has less overhead
    and is jax compatible.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        a = a.reshape(1)
    return a


def vector_angle(a, b):
    """
    Find the angle between two vectors.
    """
    a_mod = np.linalg.norm(a)
    b_mod = np.linalg.norm(b)
    if a.ndim == 2 & b.ndim == 2:
        dot = np.einsum('ij,ik->i', a/a_mod, b/b_mod, optimize=True)
    elif a.ndim == 1 & b.ndim == 1:
        dot = np.dot(a/a_mod, b/b_mod)
    else:
        raise Exception('Input must have 1 or 2 dimensions.')
    angle = np.arccos(dot)
    return angle


def vector_rotate(a, b, theta):
    """
    Rotate vector a around vector b by an angle theta (radians)

    Programming Notes:
      u: parallel projection of a on b_hat.
      v: perpendicular projection of a on b_hat.
      w: a vector perpendicular to both a and b.
    """

    if a.ndim == 2:
        b_hat = b / np.linalg.norm(b)
        dot = np.einsum('ij,j->i', a, b_hat, optimize=True)
        u = np.einsum('i,j->ij', dot, b_hat, optimize=True)
        v = a - u
        w = np.cross(b_hat, v)
        c = u + v * np.cos(theta) + w * np.sin(theta)
    elif a.ndim == 1:
        b_hat = b / np.linalg.norm(b)
        u = b_hat * np.dot(a, b_hat)
        v = a - u
        w = np.cross(b_hat, v)
        c = u + v * np.cos(theta) + w * np.sin(theta)
    else:
        raise Exception('Input array must be 1d (vector) or 2d (array of vectors)')
    return c


def magnitude(vector):
    """
    Calculate magnitude of a vector or array of vectors.
    """
    if vector.ndim > 1:
        mag = np.linalg.norm(vector, axis=1)
    else:
        mag = np.linalg.norm(vector)

    return mag


def normalize(vector):
    """
    Normalize a vector or an array of vectors.
    If an array of vectors is given it should have the shape (N,M) where
    |  N: Number of vectors
    |  M: Vector length
    """

    if vector.ndim > 1:
        norm = np.linalg.norm(vector, axis=1)
        vector /= np.expand_dims(norm, 1)
    else:
        norm = np.linalg.norm(vector)
        vector /= norm

    return vector


def sinusoidal_spiral(phi, b, r0, theta0):
    r = r0 * (np.sin(theta0 + (b-1)*phi)/np.sin(theta0))**(1/(b-1))
    return r


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    matrix = np.array(
        [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)]
        ,[2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)]
        ,[2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return matrix


def bragg_angle(wavelength, crystal_spacing):
    """
    The Bragg angle calculation is used so often that it deserves its own
    function.

    .. Note::
      The crystal_spacing here is the true spacing, not the 2d spacing that
      is sometimes used in the literature.
    """
    bragg_angle = np.arcsin(wavelength / (2 * crystal_spacing))
    return bragg_angle


def cyl_from_car(point_car):
    """
    Convert from cartesian to cylindrical coordinates.
    """
    if not isinstance(point_car, np.ndarray):
        point_car = np.array(point_car)

    point_cyl = np.empty(point_car.shape)

    if point_car.ndim == 2:
        point_cyl[:, 0] = np.sqrt(np.sum(point_car[:, 0:2]**2, 1))
        point_cyl[:, 1] = np.arctan2(point_car[:, 1], point_car[:, 0])
        point_cyl[:, 2] = point_car[:, 2]
    else:
        point_cyl[0] = np.sqrt(np.sum(point_car[0:2]**2))
        point_cyl[1] = np.arctan2(point_car[1], point_car[0])
        point_cyl[2] = point_car[2]

    return point_cyl


def car_from_cyl(point_cyl):
    """
    Convert from cylindrical to cartesian coordinates.
    """
    if not isinstance(point_cyl, np.ndarray):
        point_cyl = np.array(point_cyl)

    point_car = np.empty(point_cyl.shape)

    if point_cyl.ndim == 2:
        point_car[:, 0] = point_cyl[:, 0]*np.cos(point_cyl[:, 1])
        point_car[:, 1] = point_cyl[:, 0]*np.sin(point_cyl[:, 1])
        point_car[:, 2] = point_cyl[:, 2]
    else:
        point_car[0] = point_cyl[0]*np.cos(point_cyl[1])
        point_car[1] = point_cyl[0]*np.sin(point_cyl[1])
        point_car[2] = point_cyl[2]

    return point_car


def tor_from_car(point_car, major_radius):
    """
    Convert from cartesian to toroidal coordinates.

    Arguments
    ---------
    point_car: array [meters]
      Cartesian coordinates [x,y,z]

    major_radius: float [meters]
      Torus Major Radius

    Returns
    -------
    point_tor: array [meters]
      Toroidal coordinates [r_min, theta_poloidal, theta_toroidal]
    """
    if not isinstance(point_car, np.ndarray):
        point_car = np.array(point_car)

    point_tor = np.empty(point_car.shape)

    if point_tor.ndim == 2:
        d = np.linalg.norm(point_car[:, 0:2], axis=1) - major_radius
        point_tor[:, 2] = np.arctan2(point_car[:, 1], point_car[:, 0])
        point_tor[:, 1] = np.arctan2(point_car[:, 2], d)
        point_tor[:, 0] = np.sqrt(np.power(point_car[:, 2], 2) + np.power(d, 2))
    else:
        d = np.linalg.norm(point_car[0:2]) - major_radius
        point_tor[2] = np.arctan2(point_car[1], point_car[0])
        point_tor[1] = np.arctan2(point_car[2], d)
        point_tor[0] = np.sqrt(np.power(point_car[2], 2) + np.power(d, 2))

    return point_tor


def car_from_tor(point_tor, major_radius):
    """
    Convert from toroidal to cartesian coordinates.

    Arguments
    ---------
    point_tor: array [meters]
      Toroidal coordinates [r_min, theta_poloidal, theta_toroidal]

    major_radius: float [meters]
      Torus Major Radius

    Returns
    -------
    point_car: array [meters]
      Cartesian coordinates [x,y,z]
    """

    if not isinstance(point_tor, np.ndarray):
        point_tor = np.array(point_tor)

    point_car = np.empty(point_tor.shape)

    if point_tor.ndim == 2:
        point_car[:, 0] = major_radius*np.cos(point_tor[:, 2])
        point_car[:, 1] = major_radius*np.sin(point_tor[:, 2])
        vector = point_car[:, 0:2]/np.linalg.norm(point_car[:, 0:2], axis=1)

        point_car[:, 0] = point_car[:, 0] + vector[:, 0]*point_tor[:, 0]*np.cos(point_tor[:, 1])
        point_car[:, 1] = point_car[:, 1] + vector[:, 1]*point_tor[:, 0]*np.cos(point_tor[:, 1])
        point_car[:, 2] = point_tor[:, 0]*np.sin(point_tor[:, 1])
    else:
        point_car[0] = major_radius*np.cos(point_tor[2])
        point_car[1] = major_radius*np.sin(point_tor[2])
        vector = point_car[0:2]/np.linalg.norm(point_car[0:2])

        point_car[0] = point_car[0] + vector[0]*point_tor[0]*np.cos(point_tor[1])
        point_car[1] = point_car[1] + vector[1]*point_tor[0]*np.cos(point_tor[1])
        point_car[2] = point_tor[0]*np.sin(point_tor[1])

    return point_car


def point_in_triangle_2d(pt, p0, p1, p2):
    """
    Determine if a point (or set of points) fall within a triangle as specified
    by three vertices.  Calculation is performed in two dimensions (2D).
    """
    pt = np.asarray(pt)
    if pt.ndim == 2:
        area = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
        a = 1 / (2 * area) * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * pt[:, 0] + (p0[0] - p2[0]) * pt[:, 1])
        b = 1 / (2 * area) * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * pt[:, 0] + (p1[0] - p0[0]) * pt[:, 1])
        c = 1 - a - b
    else:
        area = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
        a = 1 / (2 * area) * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * pt[0] + (p0[0] - p2[0]) * pt[1])
        b = 1 / (2 * area) * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * pt[0] + (p1[0] - p0[0]) * pt[1])
        c = 1 - a - b

    return (a >= 0) & (b >= 0) & (c >= 0)