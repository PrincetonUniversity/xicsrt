"""
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:14:52 2019

@author: NKrvavica

This contains vectorized quadratic, cubic and quartic solvers

This code is adapted from https://github.com/NKrvavica/fqs commit d95d684
MIT License - Copyright (c) 2019 NKrvavica

Some modifications have been made to the Api, but the implementation is
identical to fqs.
"""

import math
import numpy as np

def multi_quadratic(a0, b0, c0):
    ''' Analytical solver for multiple quadratic equations
    (2nd order polynomial), based on `numpy` functions.

    Parameters
    ----------
    a0, b0, c0: array_like
        Input data are coefficients of the Quadratic polynomial::

            a0*x^2 + b0*x + c0 = 0

    Returns
    -------
    r1, r2: ndarray
        Output data is an array of two roots of given polynomials.
    '''
    ''' Reduce the quadratic equation to to form:
        x^2 + ax + b = 0'''
    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = np.sqrt(delta + 0j)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2


def multi_cubic(a0, b0, c0, d0, all_roots=True):
    ''' Analytical closed-form solver for multiple cubic equations
    (3rd order polynomial), based on `numpy` functions.

    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::

            a0*x^3 + b0*x^2 + c0*x + d0 = 0

    all_roots: bool, optional
        If set to `True` (default) all three roots are computed and returned.
        If set to `False` only one (real) root is computed and returned.

    Returns
    -------
    roots: ndarray
        Output data is an array of three roots of given polynomials of size
        (3, M) if `all_roots=True`, and an array of one root of size (M,)
        if `all_roots=False`.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13
    sqr3 = math.sqrt(3)

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    # Masks for different combinations of roots
    m1 = (f == 0) & (g == 0) & (h == 0)     # roots are real and equal
    m2 = (~m1) & (h <= 0)                   # roots are real and distinct
    m3 = (~m1) & (~m2)                      # one real root and two complex

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        root = np.zeros_like(x)
        positive = (x >= 0)
        negative = ~positive
        root[positive] = x[positive]**third
        root[negative] = -(-x[negative])**third
        return root

    def roots_all_real_equal(c):
        ''' Compute cubic roots if all roots are real and equal
        '''
        r1 = -cubic_root(c)
        if all_roots:
            return r1, r1, r1
        else:
            return r1

    def roots_all_real_distinct(a13, f, g, h):
        ''' Compute cubic roots if all roots are real and distinct
        '''
        j = np.sqrt(-f)
        k = np.arccos(-0.5*g / (j*j*j))
        m = np.cos(third*k)
        r1 = 2*j*m - a13
        if all_roots:
            n = sqr3 * np.sin(third*k)
            r2 = -j * (m + n) - a13
            r3 = -j * (m - n) - a13
            return r1, r2, r3
        else:
            return r1

    def roots_one_real(a13, g, h):
        ''' Compute cubic roots if one root is real and other two are complex
        '''
        sqrt_h = np.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        r1 = S_plus_U - a13
        if all_roots:
            S_minus_U = S - U
            r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
            r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
            return r1, r2, r3
        else:
            return r1

    # Compute roots
    if all_roots:
        roots = np.zeros((3, len(a))).astype(complex)
        roots[:, m1] = roots_all_real_equal(c[m1])
        roots[:, m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[:, m3] = roots_one_real(a13[m3], g[m3], h[m3])
    else:
        roots = np.zeros(len(a))  # .astype(complex)
        roots[m1] = roots_all_real_equal(c[m1])
        roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])

    return roots


def multi_quartic(a0, b0, c0, d0, e0):
    ''' Analytical closed-form solver for multiple quartic equations
    (4th order polynomial), based on `numpy` functions. Calls
    `multi_cubic` and `multi_quadratic`.

    Parameters
    ----------
    a0, b0, c0, d0, e0: array_like
        Input data are coefficients of the Quartic polynomial::

            a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0

    Returns
    -------
    r1, r2, r3, r4: ndarray
        Output data is an array of four roots of given polynomials.
    '''

    ''' Reduce the quartic equation to to form:
        x^4 ax^3 + bx^2 + cx + d = 0'''
    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = multi_cubic(1, p, r, p*r - 0.5*q*q, all_roots=False)

    # Additional variables
    s = np.sqrt(2*p + 2*z0.real + 0j)
    t = np.zeros_like(s)
    mask = (s == 0)
    t[mask] = z0[mask]*z0[mask] + r[mask]
    t[~mask] = -q[~mask] / s[~mask]

    # Compute roots by quadratic equations
    r0, r1 = multi_quadratic(1, s, z0 + t) - a0
    r2, r3 = multi_quadratic(1, -s, z0 - t) - a0

    return r0, r1, r2, r3


def eig_quadratic(p):
    '''Finds quadratic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 3x3 companion matrix
    '''
    a, b = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0])
    N = len(a)
    A = np.zeros((N, 2, 2))
    A[:, 1, 0] = 1
    A[:, :, 1] = - np.array([b, a]).T
    roots = np.linalg.eigvals(A)
    return roots


def eig_cubic(p):
    '''Finds cubic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 3x3 companion matrix'''
    a, b, c = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0], p[:, 3]/p[:, 0],)
    N = len(a)
    A = np.zeros((N, 3, 3))
    A[:, 1:, :2] = np.eye(2)
    A[:, :, 2] = - np.array([c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots


def eig_quartic(p):
    '''Finds quartic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 4x4 companion matrix
    '''
    a, b, c, d = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0],
                  p[:, 3]/p[:, 0], p[:, 4]/p[:, 0])
    N = len(a)
    A = np.zeros((N, 4, 4))
    A[:, 1:, :3] = np.eye(3)
    A[:, :, 3] = - np.array([d, c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots


def quadratic_roots(p, eigenvalue=False):
    '''
    A caller function for a fast quadratic root solver (2rd order polynomial).

    Parameters
    ----------
    p: array_like
        Input data are coefficients of the Cubic polynomial of the form:

            p[0]*x^2 + p[1]*x + p[2] = 0

        Stacked arrays of coefficient are allowed, which means that ``p`` may
        have size ``(3,)`` or ``(M, 3)``, where ``M>0`` is the
        number of polynomials. Note that the first axis should be used for
        stacking.

    Keywords
    --------
    eigenvalue: bool (False)
        Use an eigenvalue solver based on numpy.linalg.eigvals rather than the
        algebraic method.

    Returns
    -------
    roots: ndarray
        Output data is an array of two roots of given polynomials,
        of size ``(M, 2)``.

    Examples
    --------
    > roots = quadratic_roots([1, 3, 2])
    > roots
    array([[-2.+0.j, -1.+0.j]])

    > roots = quadratic_roots([3, 2, 1])
    > roots
    array([[-0.33333333-0.47140452j, -0.33333333+0.47140452j]])

    > roots = quadratic_roots([[1, 2, 3],
                               [3, 2, 1]])
    > roots
    array([[-2.        +0.j        , -1.        +0.j        ],
           [-0.33333333-0.47140452j, -0.33333333+0.47140452j]])
    '''
    # Convert input to array (if input is a list or tuple)
    p = np.asarray(p)
    if eigenvalue:
        return eig_quadratic(p)

    # If only one set of coefficients is given, add axis
    if p.ndim < 2:
        p = p[np.newaxis, :]

    # Check if four coefficients are given
    if p.shape[1] != 3:
        raise ValueError('Expected 2rd order polynomial with 3 '
                         'coefficients, got {:d}.'.format(p.shape[1]))

    roots = multi_quadratic(*p.T)
    return np.array(roots).T


def cubic_roots(p, eigenvalue=False):
    '''
    A caller function for a fast cubic root solver (3rd order polynomial).

    Equations are based on a closed-form analytical solutions by Cardano.

    Parameters
    ----------
    p: array_like
        Input data are coefficients of the Cubic polynomial of the form:

            p[0]*x^3 + p[1]*x^2 + p[2]*x + p[3] = 0

        Stacked arrays of coefficient are allowed, which means that ``p`` may
        have size ``(4,)`` or ``(M, 4)``, where ``M>0`` is the
        number of polynomials. Note that the first axis should be used for
        stacking.

    Keywords
    --------
    eigenvalue: bool (False)
        Use an eigenvalue solver based on numpy.linalg.eigvals rather than the
        algebraic method.

    Returns
    -------
    roots: ndarray
        Output data is an array of three roots of given polynomials,
        of size ``(M, 3)``.

    Examples
    --------
    > roots = cubic_roots([1, 7, -806, -1050])
    > roots
    array([[ 25.80760451+0.j, -31.51667909+0.j,  -1.29092543+0.j]])

    > roots = cubic_roots([1, 2, 3, 4])
    > roots
    array([[-1.65062919+0.j        , -0.1746854 +1.54686889j,
            -0.1746854 -1.54686889j]])

    > roots = cubic_roots([[1, 2, 3, 4],
                               [1, 7, -806, -1050]])
    > roots
    array([[ -1.65062919+0.j        ,  -0.1746854 +1.54686889j,
             -0.1746854 -1.54686889j],
           [ 25.80760451+0.j        , -31.51667909+0.j        ,
             -1.29092543+0.j        ]])
    '''
    # Convert input to array (if input is a list or tuple)
    p = np.asarray(p)
    if eigenvalue:
        return eig_cubic(p)

    # If only one set of coefficients is given, add axis
    if p.ndim < 2:
        p = p[np.newaxis, :]

    # Check if four coefficients are given
    if p.shape[1] != 4:
        raise ValueError('Expected 3rd order polynomial with 4 '
                         'coefficients, got {:d}.'.format(p.shape[1]))

    roots = multi_cubic(*p.T)
    return np.array(roots).T


def quartic_roots(p, eigenvalue=False):
    '''
    A caller function for a fast quartic root solver (4th order polynomial).

    Equations are based on a closed-form analytical solutions by Ferrari
    and Cardano.

    Parameters
    ----------
    p: array_like
        Input data are coefficients of the Quartic polynomial of the form:

            p[0]*x^4 + p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4] = 0

        Stacked arrays of coefficient are allowed, which means that ``p`` may
        have size ``(5,)`` or ``(M, 5)``, where ``M>0`` is the
        number of polynomials. Note that the first axis should be used for
        stacking.

    Keywords
    --------
    eigenvalue: bool (False)
        Use an eigenvalue solver based on numpy.linalg.eigvals rather than the
        algebraic method.

    Returns
    -------
    roots: ndarray
        Output data is an array of four roots of given polynomials,
        of size ``(M, 4)``.

    Examples
    --------
    > roots = quartic_roots([1, 7, -806, -1050, 38322])
    > roots
    array([[-30.76994812-0.j,  -7.60101564+0.j,   6.61999319+0.j,
             24.75097057-0.j]])

    > roots = quartic_roots([1, 2, 3, 4, 5])
    > roots
    array([[-1.28781548-0.85789676j, -1.28781548+0.85789676j,
             0.28781548+1.41609308j,  0.28781548-1.41609308j]])

    > roots = quartic_roots([[1, 2, 3, 4, 5],
                               [1, 7, -806, -1050, 38322]])
    > roots
    array([[ -1.28781548-0.85789676j,  -1.28781548+0.85789676j,
              0.28781548+1.41609308j,   0.28781548-1.41609308j],
           [-30.76994812-0.j        ,  -7.60101564+0.j        ,
              6.61999319+0.j        ,  24.75097057-0.j        ]])
    '''
    # Convert input to an array (if input is a list or tuple)
    p = np.asarray(p)
    if eigenvalue:
        return eig_quartic(p)

    # If only one set of coefficients is given, add axis
    if p.ndim < 2:
        p = p[np.newaxis, :]

    # Check if all five coefficients are given
    if p.shape[1] != 5:
        raise ValueError('Expected 4th order polynomial with 5 '
                         'coefficients, got {:d}.'.format(p.shape[1]))

    roots = multi_quartic(*p.T)
    return np.array(roots).T