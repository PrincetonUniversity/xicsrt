# -*- coding: utf-8 -*-

import jax.numpy as np

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
    b_hat = b / np.linalg.norm(b)
    u = b_hat * np.dot(a, b_hat)
    v = a - u
    w = np.cross(b_hat, v)
    c = u + v * np.cos(theta) + w * np.sin(theta)
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