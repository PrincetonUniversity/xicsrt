# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

A set of mathematical utilities and vector convenience functions for XICSRT.
"""

import numpy as np

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
    b_hat = b / np.linalg.norm(b)
    u = b_hat * np.dot(a, b_hat)
    v = a - u
    w = np.cross(b_hat, v)
    c = u + v * np.cos(theta) + w * np.sin(theta)
    return c

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
    The Bragg angle calculation is used so often that it deserves its own funct
    """
    bragg_angle = np.arcsin(wavelength / (2 * crystal_spacing))
    return bragg_angle

def cyl_from_car(point_car):
    #convert cartesian coordinates -> cylindirical coordinates
    radius  = np.sqrt(np.power(point_car[0],2) + np.power(point_car[1],2))
    azimuth = np.arctan2(point_car[1], point_car[0])
    height  = point_car[2]
    point_cyl = [radius, azimuth, height]
    return point_cyl

def tor_from_car(point_car, a):
    """
    point_car   = Cartesian Coordinates [x,y,z]
    a           = Torus Major Radius
    point_tor   = [Minor Radius, Poloidal Angle, Toroidal Angle]
    """
    rho  = np.sqrt(np.power(point_car[0],2) + np.power(point_car[1],2))
    tor  = np.arctan2(point_car[1], point_car[0]) + np.pi
    w    = rho - a
    pol  = np.arctan2(point_car[2], w)
    rad  = np.sqrt(np.power(point_car[2],2) + np.power(w,2))
    point_tor = np.asarray([rad, pol, tor])
    return point_tor
