# -*- coding: utf-8 -*-
"""
Geometry unit tests for the jaxrt engine.

This file includes AI generated code using Claude (Fable 5).

The jax shape intersect functions are compared with the numpy engine
shape classes on identical ray inputs. The deterministic geometry
should agree to near machine precision in float64.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')

import xicsrt.jaxrt  # noqa: F401  (enables jax x64 mode)
from xicsrt.jaxrt.shapes import _cylinder, _plane, _sphere, _torus
from xicsrt.objects._RayArray import RayArray
from xicsrt.xicsrt_public import get_element


def _make_rays(num, seed):
    """
    Build a bundle of random rays aimed toward the optic at z ~ 1.
    """
    rng = np.random.default_rng(seed)
    rays = RayArray()
    rays['origin'] = rng.uniform(-0.05, 0.05, (num, 3))
    direction = np.zeros((num, 3))
    direction[:, 0:2] = rng.uniform(-0.2, 0.2, (num, 2))
    direction[:, 2] = 1.0
    rays['direction'] = direction / np.linalg.norm(direction, axis=1)[:, None]
    rays['wavelength'] = np.full(num, 3.9492)
    rays['mask'] = np.ones(num, dtype=bool)
    return rays


def _jax_rays(rays):
    return {
        'origin': jax.numpy.asarray(rays['origin']),
        'direction': jax.numpy.asarray(rays['direction']),
        'wavelength': jax.numpy.asarray(rays['wavelength']),
        'weight': jax.numpy.ones(len(rays['mask'])),
        'mask': jax.numpy.asarray(rays['mask']),
    }


def _numpy_optic(config):
    config = dict(config)
    config.setdefault('origin', [0.0, 0.0, 1.0])
    config.setdefault('zaxis', list(np.array([0.0, 0.1, -1.0])
                                    / np.linalg.norm([0.0, 0.1, -1.0])))
    config.setdefault('xsize', 0.5)
    config.setdefault('ysize', 0.5)
    optic = get_element({'optics': {'optic': config}}, 'optic')
    return optic


def _compare_shape(config, jax_shape, num=200, seed=0):
    """
    Trace identical rays through the numpy shape class and the jax
    shape module and compare intersections, normals and masks.
    """
    optic = _numpy_optic(config)

    rays_np = _make_rays(num, seed)
    rays_jx = _jax_rays(rays_np)

    xloc_np, norm_np, mask_np = optic.intersect(rays_np.copy())

    param = dict(optic.param)
    param['orientation'] = optic.orientation
    geom = jax_shape.setup(param)
    xloc_jx, norm_jx, mask_jx = jax_shape.intersect(rays_jx, geom)

    xloc_jx = np.asarray(xloc_jx)
    norm_jx = np.asarray(norm_jx)
    mask_jx = np.asarray(mask_jx)

    assert np.array_equal(mask_np, mask_jx)
    assert np.sum(mask_np) > 0, 'test scenario produced no intersections'
    mm = mask_np
    np.testing.assert_allclose(xloc_jx[mm], xloc_np[mm], atol=1e-9)
    np.testing.assert_allclose(norm_jx[mm], norm_np[mm], atol=1e-9)


def test_plane():
    config = {'class_name': 'XicsrtOpticDetector'}
    _compare_shape(config, _plane)


@pytest.mark.parametrize('convex', [False, True])
def test_sphere(convex):
    config = {
        'class_name': 'XicsrtOpticSphericalMirror',
        'radius': 1.5,
        'convex': convex,
    }
    _compare_shape(config, _sphere)


def test_cylinder():
    config = {
        'class_name': 'XicsrtOpticCylindricalCrystal',
        'radius': 1.5,
        'crystal_spacing': 2.45676,
        'rocking_type': 'gaussian',
        'rocking_fwhm': 1e-4,
    }
    _compare_shape(config, _cylinder)


def test_torus():
    config = {
        'class_name': 'XicsrtOpticToroidalCrystal',
        'radius_major': 1.5,
        'radius_minor': 0.75,
        'crystal_spacing': 2.45676,
        'rocking_type': 'gaussian',
        'rocking_fwhm': 1e-4,
    }
    _compare_shape(config, _torus)
