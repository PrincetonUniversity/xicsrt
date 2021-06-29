# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

A set of mathematical function with jax acceleration. Many of these functions
are exact copies or slight modification of the functions in xicsrt_math. Other
function are specific to this module.

Programming Notes
-----------------

These module was developed to support some specific modeling work by N. Pablant
and is not used in any of the built-in xicsrt code. There is no plan to support
jax generally within xicsrt, so I am not really sure of the best way to handle
this module for the moment. Maybe move it into xicsrt_contrib?
"""

import jax.numpy as np

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
        dot = np.einsum('ij,ik->i', a/a_mod, b/b_mod)
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
        dot = np.einsum('ij,j->i', a, b_hat)
        u = np.einsum('i,j->ij', dot, b_hat)
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


def sinusoidal_spiral(phi, b, r0, theta0):
    r = r0 * (np.sin(theta0 + (b-1)*phi)/np.sin(theta0))**(1/(b-1))
    return r


def point_to_external(point_local, orientation, origin):
    return vector_to_external(point_local, orientation) + origin


def point_to_local(point_external, orientation, origin):
    return vector_to_local(point_external - origin, orientation)


def vector_to_external(vector, orientation):
    if vector.ndim == 2:
        vector = np.einsum('ij,ki->kj', orientation, vector)
    elif vector.ndim == 1:
        vector = np.einsum('ij,i->j', orientation, vector)
    else:
        raise Exception('vector.ndim must be 1 or 2')
    return vector


def vector_to_local(vector, orientation):
    if vector.ndim == 2:
        vector = np.einsum('ji,ki->kj', orientation, vector)
    elif vector.ndim == 1:
        vector = np.einsum('ji,i->j', orientation, vector)
    else:
        raise Exception('vector.ndim must be 1 or 2')
    return vector