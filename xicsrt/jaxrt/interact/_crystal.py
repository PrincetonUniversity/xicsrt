# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Bragg crystal interaction (jaxrt).

Port of :class:`xicsrt.optics._InteractCrystal.InteractCrystal` to
pure jax functions. Rays reflect specularly, with a reflection
probability given by the crystal rocking curve evaluated at the
difference between the incident angle and the Bragg angle for each
ray's wavelength.

Rocking curve types 'step', 'gaussian' and 'file' are supported. File
based rocking curves are loaded once during setup (the numpy engine
re-reads the file on every iteration; do not copy that) and evaluated
with linear interpolation, identical to the numpy engine.

This module was AI generated using Claude (Fable 5).
"""

import jax
import jax.numpy as jnp
import numpy as np

from xicsrt.jaxrt import _rays
from xicsrt.jaxrt.interact import _mirror
from xicsrt.tools import xicsrt_bragg


def setup(param):
    """
    Build static crystal parameters from an initialized optic param
    dict, precomputing the rocking curve table for 'file' types.
    """
    phys = {
        'crystal_spacing': float(param['crystal_spacing']),
        'reflectivity': float(param['reflectivity']),
        'check_bragg': bool(param['check_bragg']),
        'rocking_type': str.lower(param['rocking_type']),
    }

    rocking_type = phys['rocking_type']
    if 'step' in rocking_type or 'gauss' in rocking_type:
        phys['rocking_fwhm'] = float(param['rocking_fwhm'])
    elif 'file' in rocking_type:
        phys.update(_setup_rocking_file(param))
    else:
        raise Exception(f'Rocking curve type not understood: {rocking_type}')

    return phys


def _setup_rocking_file(param):
    """
    Load a rocking curve data file into interpolation tables.

    The mixing of sigma and pi polarization reflectivities is applied
    here (it is a linear combination, so mixing the tables before
    interpolation is identical to mixing after).
    """
    data = xicsrt_bragg.read(param['rocking_file'], param['rocking_filetype'])

    units = data['units']['dtheta_in']
    if units == 'urad':
        scale_theta = 1e-6
    elif units == 'arcset':
        scale_theta = np.pi / (180 * 3600)
    elif units == 'rad':
        scale_theta = 1.0
    else:
        raise Exception(f'Units in rocking curve data not understood: {units}')

    mix = param['rocking_mix']
    dtheta = np.asarray(data['value']['dtheta_in'], dtype=np.float64) * scale_theta
    reflect = (mix * np.asarray(data['value']['reflect_s'], dtype=np.float64)
               + (1 - mix) * np.asarray(data['value']['reflect_p'], dtype=np.float64))

    return {
        'rocking_dtheta': jnp.asarray(dtheta),
        'rocking_reflect': jnp.asarray(reflect),
    }


def interact(rays, xloc, norm, mask, phys, key):
    """
    Bragg reflection: stochastically accept rays based on the rocking
    curve, then reflect the accepted rays specularly.
    """
    mask = angle_check(rays, norm, mask, phys, key)

    rays = dict(rays)
    rays['origin'] = xloc
    rays['direction'] = _mirror.reflect(rays['direction'], norm, mask)
    rays['mask'] = mask
    return rays


def angle_calc(rays, norm, phys):
    """
    Bragg angle (from each ray's wavelength) and incident angle (from
    each ray's direction and the surface normal), in radians.
    """
    bragg_angle = jnp.arcsin(
        rays['wavelength'] / (2 * phys['crystal_spacing']))

    # The ray directions are unit vectors, so the magnitude division of
    # the numpy engine is not needed.
    dot = jnp.abs(_rays.dot(rays['direction'], -norm))
    incident_angle = (jnp.pi / 2) - jnp.arccos(dot)

    return bragg_angle, incident_angle


def angle_check(rays, norm, mask, phys, key):
    """
    Update the mask with the stochastic rocking curve acceptance.
    """
    if not phys['check_bragg']:
        return mask

    bragg_angle, incident_angle = angle_calc(rays, norm, phys)
    return mask & rocking_curve_filter(incident_angle, bragg_angle, phys, key)


def rocking_curve_filter(incident_angle, bragg_angle, phys, key):
    """
    Stochastic acceptance from the rocking curve: for each ray the
    reflection probability p is compared with an independent uniform
    random number. This is identical to the numpy engine and preserves
    exact photon statistics.
    """
    dtheta = incident_angle - bragg_angle
    rocking_type = phys['rocking_type']

    if 'step' in rocking_type:
        p = jnp.where(jnp.abs(dtheta) <= phys['rocking_fwhm'] / 2, 1.0, 0.0)
    elif 'gauss' in rocking_type:
        sigma = phys['rocking_fwhm'] / (2 * jnp.sqrt(2 * jnp.log(2)))
        p = jnp.exp(-dtheta**2 / (2 * sigma**2))
    elif 'file' in rocking_type:
        p = jnp.interp(
            dtheta, phys['rocking_dtheta'], phys['rocking_reflect'],
            left=0.0, right=0.0)
    else:
        raise Exception(f'Rocking curve type not understood: {rocking_type}')

    p = p * phys['reflectivity']

    test = jax.random.uniform(key, incident_angle.shape)
    return p >= test
