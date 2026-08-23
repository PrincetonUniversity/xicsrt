# -*- coding: utf-8 -*-
"""
Engine-level tests for the jaxrt engine.

This file includes AI generated code using Claude (Fable 5).

These tests run small end-to-end scenarios through both the numpy and
jaxrt engines and check that:

  * output dictionary structure matches,
  * detected ray counts agree within Poisson error bars,
  * detector image centroids agree,
  * Poisson (capacity + mask) source statistics behave correctly,
  * analytic expectations (Bragg selection) hold.

The statistical comparisons use generous sigma bounds so the tests are
deterministic-in-practice while still catching real regressions.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')

import xicsrt
import xicsrt.jaxrt


def _base_config(**general):
    config = {
        'general': {
            'number_of_iter': 2,
            'print_results': False,
            'random_seed': 12345,
        },
        'sources': {
            'source': {
                'class_name': 'XicsrtSourceFocused',
                'intensity': 2e4,
                'wavelength': 3.9492,
                'temperature': 500.0,
                'mass_number': 40.0,
                'spread': np.radians(2.0),
                'target': [0.0, 0.0, 0.80374151],
            },
        },
        'optics': {
            'crystal': {
                'class_name': 'XicsrtOpticSphericalCrystal',
                'origin': [0.0, 0.0, 0.80374151],
                'zaxis': [0.0, 0.59497864, -0.80374151],
                'xsize': 0.2,
                'ysize': 0.2,
                'radius': 1.0,
                'crystal_spacing': 2.45676,
                'rocking_type': 'gaussian',
                'rocking_fwhm': 1e-4,
            },
            'detector': {
                'class_name': 'XicsrtOpticDetector',
                'origin': [0.0, 0.76871290, 0.56904832],
                'zaxis': [0.0, -0.95641806, 0.29200084],
                'xsize': 0.4,
                'ysize': 0.4,
            },
        },
    }
    config['general'].update(general)
    return config


def _centroid(image):
    xx, yy = np.indices(image.shape)
    return np.array([(xx * image).sum(), (yy * image).sum()]) / image.sum()


def test_output_structure():
    """
    The jaxrt results dict must have the same structure as the numpy
    engine results dict.
    """
    config = _base_config()
    results = xicsrt.jaxrt.raytrace(config)

    assert set(results) >= {'config', 'total', 'found', 'lost'}
    assert list(results['total']['meta']) == ['source', 'crystal', 'detector']
    assert 'num_out' in results['total']['meta']['detector']

    image = results['total']['image']['detector']
    assert image.shape == (100, 100)

    # Note: 'weight' is dropped by combine_raytrace (RayArray.zeros only
    # defines these four keys); the numpy engine behaves the same way.
    for key in ['origin', 'direction', 'wavelength', 'mask']:
        assert key in results['found']['history']['detector']

    num_found = results['total']['meta']['detector']['num_out']
    assert len(results['found']['history']['detector']['mask']) == num_found
    assert np.all(results['found']['history']['detector']['mask'])
    assert not np.any(results['lost']['history']['detector']['mask'])

    # Image counts and history must agree with the meta counts.
    assert image.sum() == num_found


def test_statistics_match_numpy():
    """
    Detected counts must agree with the numpy engine within Poisson
    error bars, and image centroids must coincide.
    """
    config = _base_config()
    res_np = xicsrt.raytrace(config)
    res_jx = xicsrt.jaxrt.raytrace(config)

    n_np = float(res_np['total']['meta']['detector']['num_out'])
    n_jx = float(res_jx['total']['meta']['detector']['num_out'])
    assert n_np > 100, 'test scenario should detect a reasonable number of rays'

    sigma = np.sqrt(n_np + n_jx)
    assert abs(n_np - n_jx) < 5 * sigma

    centroid_np = _centroid(res_np['total']['image']['detector'])
    centroid_jx = _centroid(res_jx['total']['image']['detector'])
    # Centroids in pixels; the images are 100x100.
    assert np.all(np.abs(centroid_np - centroid_jx) < 2.0)


def test_poisson_source():
    """
    With use_poisson enabled, generated ray counts must vary between
    iterations and stay within Poisson expectations of the mean.
    """
    config = _base_config(number_of_iter=4)
    config['sources']['source']['intensity'] = 1e4
    config['sources']['source']['use_poisson'] = True

    results = xicsrt.jaxrt.raytrace(config)

    num_source = results['total']['meta']['source']['num_out']
    mean = 4 * 1e4
    assert abs(num_source - mean) < 5 * np.sqrt(mean)
    # A fixed (non-poisson) count would give exactly the mean;
    # this is astronomically improbable for a true Poisson draw.
    assert num_source != mean


def test_bragg_selection():
    """
    With a narrow rocking curve, only wavelengths near the Bragg
    condition of the actual incident angles survive to the detector.
    """
    config = _base_config()
    results = xicsrt.jaxrt.raytrace(config)

    wavelength = results['found']['history']['detector']['wavelength']
    crystal_spacing = config['optics']['crystal']['crystal_spacing']

    # Reconstruct the incidence geometry from the crystal history.
    crystal = results['found']['history']['crystal']
    source = results['found']['history']['source']
    incident = crystal['origin'] - source['origin']
    incident /= np.linalg.norm(incident, axis=1)[:, None]
    outgoing = crystal['direction']

    # For a specular reflection the deflection angle between incident
    # and outgoing rays is twice the grazing incidence angle.
    deflection = np.arccos(np.sum(incident * outgoing, axis=1))
    theta = deflection / 2  # grazing incidence angle

    bragg_wavelength = 2 * crystal_spacing * np.sin(theta)
    # All detected rays satisfied the rocking curve, so the ray
    # wavelength must equal the Bragg wavelength for its own incidence
    # angle to within a few rocking widths.
    dtheta_equiv = np.abs(wavelength - bragg_wavelength) / (
        2 * crystal_spacing * np.cos(theta))
    assert np.all(dtheta_equiv < 5 * config['optics']['crystal']['rocking_fwhm'])


def test_label_field_present_and_inert():
    """
    F034: `new_rays` includes an always-present `label` field, which must
    pass through `xicsrt.jaxrt.raytrace` unchanged (all zeros, correct
    shape, integer dtype) since no jaxrt source populates it.
    """
    config = _base_config()
    results = xicsrt.jaxrt.raytrace(config)

    for name in ('source', 'crystal', 'detector'):
        for section in ('found', 'lost'):
            history = results[section]['history'][name]
            assert 'label' in history
            assert history['label'].shape == history['mask'].shape
            assert history['label'].dtype.kind in ('i', 'u')
            assert np.all(history['label'] == 0)


def test_multiple_runs_combine():
    """
    number_of_runs > 1 must combine correctly and use different seeds
    per run.
    """
    config = _base_config(number_of_runs=2, number_of_iter=1)
    results = xicsrt.jaxrt.raytrace(config)

    total = results['total']['meta']['source']['num_out']
    assert total == 2 * config['sources']['source']['intensity']

    # The two runs must not be identical: with different per-run seeds
    # the found ray directions should differ (the source is a point
    # source, so origins are all identical by construction).
    directions = results['found']['history']['source']['direction']
    num = len(directions) // 2
    assert not np.allclose(directions[:num], directions[num:2 * num])
