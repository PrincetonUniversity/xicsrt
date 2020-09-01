# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
"""

import numpy as np
import logging

from scipy import optimize

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from xicsrt.tools import xicsrt_math_jax as xmj
from xicsrt.tools import xicsrt_math as xm


def sinsp(phi, b, r0, theta0):
    """Equation for a Sinusoidal Spiral."""
    r = r0 * (jnp.sin(theta0 + (b-1)*phi)/jnp.sin(theta0))**(1/(b-1))
    return r


def spiral(phi, beta, inp, extra=False):
    """3D geometry for a Sinusoidal Spiral crystal."""
    b = inp['b']
    r0 = inp['r0']
    theta0 = inp['theta0']
    S = inp['S']

    r = xmj.sinusoidal_spiral(phi, b, r0, theta0)
    a = theta0 + b * phi
    t = theta0 + (b - 1) * phi
    C_norm = jnp.array([-1 * jnp.sin(a), jnp.cos(a), 0.0])

    C = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), 0.0])
    rho = r / (b * jnp.sin(t))
    O = C + rho * C_norm

    CS = S - C
    CS_dist = jnp.linalg.norm(CS)
    CS_hat = CS / CS_dist

    # When the source is at the origin, bragg will equal theta.
    bragg = jnp.pi / 2 - jnp.arccos(jnp.dot(CS_hat, C_norm))
    axis = jnp.array([0.0, 0.0, 1.0])
    CD_hat = xmj.vector_rotate(CS_hat, axis, -2 * (jnp.pi / 2 - bragg))
    CD_dist = rho * jnp.sin(bragg)
    D = C + CD_dist * CD_hat

    SD = D - S
    SD_hat = SD / jnp.linalg.norm(SD)

    CP_hat = -1 * jnp.cross(SD_hat, axis)
    aDSC = xmj.vector_angle(SD_hat, -1 * CS_hat)
    CP_dist = CS_dist * jnp.sin(aDSC)
    P = C + CP_hat * CP_dist

    CQ_hat = C_norm
    aQCP = xmj.vector_angle(CQ_hat, CP_hat)
    CQ_dist = CP_dist / jnp.cos(aQCP)
    Q = C + CQ_hat * CQ_dist

    XP_hat = xmj.vector_rotate(CP_hat, SD_hat, beta)
    X = P - XP_hat * CP_dist

    if extra:
        out = {}
        out['C'] = C
        out['O'] = O
        out['S'] = S
        out['D'] = D
        out['P'] = P
        out['Q'] = Q
        out['X'] = X
        return out
    else:
        return X


def get_source_origin(inp):
    inp_tmp = inp.copy()
    inp_tmp['S'] = np.array([0.0, 0.0, 0.0])
    out = spiral(inp['phiC'], 0.0, inp_tmp, extra=True)

    CO = out['O'] - out['C']
    CO_dist = np.linalg.norm(CO)
    CO_hat = CO/CO_dist

    aS = (np.pi/2 - inp['thetaC'])
    zaxis = np.array([0.0, 0.0, 1.0])

    S = out['C'] + inp['sC'] * xm.vector_rotate(CO_hat, zaxis, aS)

    return S


def spiral_SD(phi, inp):
    out = spiral(phi, 0.0, inp, extra=True)
    SD = out['D'] - out['S']
    SD_hat = SD / jnp.linalg.norm(SD)
    return SD_hat


def spiral_SD_x(phi, inp):
    return spiral_SD(phi, inp)[0]


grad_spiral_SD_x = jax.grad(spiral_SD_x)


grad_grad_spiral_SD_x = jax.grad(grad_spiral_SD_x)


def _root_b_func(b_in, inp_in):
    inp = inp_in.copy()
    output = []
    for b in b_in:
        inp['b'] = b
        output.append(grad_spiral_SD_x(inp['phiC'], inp))
    return output


def _root_b_jac(b_in, inp_in):
    inp = inp_in.copy()
    output = []
    for b in b_in:
        inp['b'] = b
        output.append(grad_grad_spiral_SD_x(inp['phiC'], inp))
    return output


def find_symmetry_b(inp):
    out = optimize.root(_root_b_func, inp['b'], jac=_root_b_jac, args=(inp))
    inp_out = inp.copy()
    inp_out['b'] = out.x[0]

    logging.info(f"b: {inp_out['b']}, grad: {grad_spiral_SD_x(inp_out['phiC'], inp_out)}")
    return inp_out

