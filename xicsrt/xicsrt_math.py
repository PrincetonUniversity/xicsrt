# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import wofz

def vector_angle(a, b):
    """
    Find the angle between two vectors. Not vectorized.
    """
    angle = np.arccos(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)))
    return angle


def vector_dist_uniform(theta, number):
    """
    Create a uniform distribution of vectors with an angular spread
    of theta. Here theta is the half of the total cone angle
    (axis to edge).
    """
    output = np.empty((number, 3))

    z = np.random.uniform(np.cos(theta), 1, number)
    phi = np.random.uniform(0, 2*np.pi, number)

    output[:, 0] = np.sqrt(1 - z**2) * np.cos(phi)
    output[:, 1] = np.sqrt(1 - z**2) * np.sin(phi)
    output[:, 2] = z

    return output

def vector_dist_gaussian(FWHM, number):
    """
    Create a gaussian distribution of vectors with an angular spread
    of FWHM. Here FWHM is half of the cone angle (axis to edge).
    """
    output = np.empty((number, 3))

    # Convert the angluar FWHM to sigma.
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    # convert the angular sigma into a linear-displacement-from-vertical
    sigma_z = 1 - np.cos(sigma)
    # create the half-normal distribution of vertical displacement.
    dist = abs(np.random.normal(0, sigma_z, 10))
    z = 1.0 - dist

    phi = np.random.uniform(0, 2 * np.pi, number)

    output[:, 0] = np.sqrt(1 - z ** 2) * np.cos(phi)
    output[:, 1] = np.sqrt(1 - z ** 2) * np.sin(phi)
    output[:, 2] = z

    return output

def bragg_angle(wavelength, crystal_spacing):
    """
    The Bragg angle calculation is used so often that it deserves its own funct
    """
    bragg_angle = np.arcsin(wavelength / (2 * crystal_spacing))
    return bragg_angle

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
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
def vector_rotate(a, b, theta):
    ## Rotate vector a around vector b by an angle theta (radians)
    #project a onto b, return parallel and perpendicular component vectors
    proj_para = b * np.dot(a, b) / np.dot(b, b)
    proj_perp = a - proj_para
    
    #define and normalize the unit vector w, perpendicular to a and b
    w  = np.cross(b, proj_perp)
    if (np.linalg.norm(w) != 0): 
        w /= np.linalg.norm(w)
    
    #return the final rotated vector c
    c = proj_para + (proj_perp * np.cos(theta)) + (
            np.linalg.norm(proj_perp) * w * np.sin(theta))
    return c

def cyl_from_car(x, y, z):
    #convert cartesian coordinates -> cylindirical coordinates
    radius  = np.sqrt(np.power(x,2) + np.power(y,2))
    azimuth = np.arctan2(y, x)
    height  = z
    
    return radius, azimuth, height

def tor_from_car(x, y, z, a):
    """
    X Y Z       = Cartesian Coordinates
    rho         = Cylindrical Radius
    a           = Tokamak Major Radius
    rad pol tor = Toroidal Radius, Poloidal Angle, Toroidal Angle
    """
    rho  = np.sqrt(np.power(x,2) + np.power(y,2))
    tor  = np.arctan2(y, x) + np.pi
    w    = rho - a
    pol  = np.arctan2(z, w)
    rad  = np.sqrt(np.power(z,2) + np.power(w,2))
    return rad, pol, tor
