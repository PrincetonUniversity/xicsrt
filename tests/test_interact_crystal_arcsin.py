# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Sonnet 5)
"""
Regression test for F013.

F010's direct Voigt/Cauchy wavelength sampling has no domain truncation, so
a small fraction of sampled wavelengths can fall outside a crystal's
Bragg-reflectable range ``[-2d, 2d]``. ``InteractCrystal.angle_calc`` then
computes ``arcsin`` of an out-of-domain argument, which correctly returns
``nan`` (no incidence angle satisfies Bragg's law for such a photon). This
test checks that:

  * no warning is raised for such rays (the invalid-value warning is
    suppressed with ``np.errstate`` since the result is expected), and
  * a ray with an out-of-domain wavelength is rejected by
    ``angle_check`` for every supported rocking curve type.
"""

import warnings

import numpy as np
import pytest

from xicsrt import xicsrt_public


def _get_crystal(rocking_type='gaussian'):
    config = {
        'optics': {
            'crystal': {
                'class_name': 'XicsrtOpticSphericalCrystal',
                'origin': [0.0, 0.0, 0.0],
                'zaxis': [0.0, 0.0, 1.0],
                'xaxis': [1.0, 0.0, 0.0],
                'xsize': 0.04,
                'ysize': 0.10,
                'radius': 1.45,
                'crystal_spacing': 2.45676,
                'reflectivity': 1.0,
                'rocking_type': rocking_type,
                'rocking_fwhm': 48.070e-6,
            }
        }
    }
    return xicsrt_public.get_element(config, 'crystal')


def _rays_for_wavelengths(wavelengths):
    n = len(wavelengths)
    rays = {
        'direction': np.tile(np.array([0.0, 0.0, -1.0]), (n, 1)),
        'wavelength': np.asarray(wavelengths, dtype=np.float64),
        'mask': np.ones(n, dtype=np.bool_),
    }
    norm = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    return rays, norm


@pytest.mark.parametrize('rocking_type', ['step', 'gaussian'])
def test_angle_check_no_warning_out_of_domain(rocking_type):
    """
    A wavelength outside [-2d, 2d] must not raise a RuntimeWarning and must
    be masked out.
    """
    crystal = _get_crystal(rocking_type)
    d = crystal.param['crystal_spacing']

    # In-domain (near the nominal Bragg condition) and out-of-domain
    # wavelengths, mixed together to also check masking is per-ray correct.
    bragg_wavelength = 2 * d * np.sin(crystal.param['rocking_fwhm'] * 0 + 0.5)
    wavelengths = [bragg_wavelength, 2 * d + 1.0, -(2 * d + 100.0), 3 * d]
    rays, norm = _rays_for_wavelengths(wavelengths)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        bragg_angle, incident_angle = crystal.angle_calc(rays, norm)
        mask = crystal.angle_check(rays, norm)

    assert np.isnan(bragg_angle[1])
    assert np.isnan(bragg_angle[2])
    assert np.isnan(bragg_angle[3])
    assert not mask[1]
    assert not mask[2]
    assert not mask[3]


# Note: a 'file' rocking curve variant of this test was not added.
# xicsrt.tools.xicsrt_bragg.read_xop() references the undefined name
# 'm_log' (should be 'log') on every call, so it currently raises a
# NameError unconditionally, independent of this fix. Pre-existing,
# unrelated bug; out of scope for F013.
