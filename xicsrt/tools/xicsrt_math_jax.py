# -*- coding: utf-8 -*-
import jax.numpy as jnp

def vector_angle(a, b):
    """
    Find the angle between two vectors. Not vectorized.
    """
    angle = jnp.arccos(jnp.dot(a/jnp.linalg.norm(a), b/jnp.linalg.norm(b)))
    return angle

def vector_rotate(a, b, theta):
    """
    Rotate vector a around vector b by an angle theta (radians)

    Programming Notes:
      u: parallel projection of a on b_hat.
      v: perpendicular projection of a on b_hat.
      w: a vector perpendicular to both a and b.
    """
    b_hat = b / jnp.linalg.norm(b)
    u = b_hat * jnp.dot(a, b_hat)
    v = a - u
    w = jnp.cross(b_hat, v)
    c = u + v * jnp.cos(theta) + w * jnp.sin(theta)
    return c

def sinusoidal_spiral(phi, b, r0, theta0):
    r = r0 * (jnp.sin(theta0 + (b-1)*phi)/jnp.sin(theta0))**(1/(b-1))
    return r