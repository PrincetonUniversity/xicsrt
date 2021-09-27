"""
This File Contains Quartic Solvers, which are used in object oriented programming model
"""
import math
import numpy as np

# Quadratic solver
def multi_quadratic(a0, b0, c0):
    # Quadratic coefficient
    a, b = b0 / a0, c0 / a0

    # Some repeating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = np.sqrt(delta + 0j)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2

# Cubic solver
def multi_cubic(a0, b0, c0, d0, all_roots=True):
    # Cubic coefficients
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
        roots = np.zeros(len(a)).astype(complex)
        roots[m1] = roots_all_real_equal(c[m1])
        roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])

    return roots

# Quartic solver
def multi_quartic(a0, b0, c0, d0, e0):
    # Quartic coefficients
    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic equation
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = multi_cubic(1, p, r, p*r - 0.5*q*q, all_roots=False)

    # Additional variables
    s = np.sqrt(2*p + 2*z0.real)
    t = np.zeros_like(s)
    mask = (s == 0)
    t[mask] = z0[mask]*z0[mask] + r[mask]
    t[~mask] = -q[~mask] / s[~mask]

    # Compute roots by quadratic equations
    r0, r1 = multi_quadratic(1, s, z0 + t) - a0
    r2, r3 = multi_quadratic(1, -s, z0 - t) - a0

    return r0, r1, r2, r3

def eig_quartic_roots(p):    # Solves quartic equation
    """
    This quartic solver is Obtained from https://nkrvavica.github.io/post/on_computing_roots/
    """
    # Coefficients of quartic equation
    a, b, c, d = p[1]/p[0], p[2]/p[0], p[3]/p[0], p[4]/p[0]

    A = np.zeros((len(a), 4, 4))        # Construct of the companion matrix of given co-efficients
    A[:, 1:, :3] = np.eye(3)
    A[:, :, 3] = -np.array([d, c, b, a]).T


    return np.linalg.eigvals(A)        # Compute roots using eigenvalues

# Solves ELlipse Ray Intersection
def solve_ellipse(Center, minor, r1, dCap, dMod, rayOrig):
    dx = np.einsum('ij,ij->i', dCap, dMod)
    dy = np.einsum('ij,j->i', dCap, torusZaxis)
    
    orig = rayOrig - Center
    Ox = np.einsum('ij,ij->i', orig, dMod)
    Oy = np.einsum('ij,j->i', orig, torusZaxis)
    
    cos2 = (r1 / minor) ** 2
    
    a = dx ** 2 * cos2 + dy ** 2
    b = 2 * (dx * Ox * cos2 + dy * Oy)
    c = Ox ** 2 * cos2 + Oy ** 2 - minor ** 2
    
    return multi_quadratic(a, b, c)
