
import numpy as np
from scipy.special import wofz

def voigt(x, y):
    """
    The Voigt function is also the real part of  w(z) = exp(-z^2) erfc(iz), 
    the complex probability function, which is also known as the Faddeeva 
    function. Scipy has implemented this function under the name wofz()
    """
    z = x + 1j*y
    I = wofz(z).real
    return I


def voigt_physical(intensity, location, sigma, gamma):
    """
    The voigt function in physical parameters.
    """
    u = (x-location)/np.sqrt(2)/sigma
    a = gamma/np.sqrt(2)/sigma
    y = voigt(a, u) / np.sqrt(2*np.pi)/sigma * intensity
    return y
