# -*- coding: utf-8 -*-
"""
Tests for the optional per-ray `label` field (F034).

Covers:
  * a source that sets `rays['label']` has that field preserved unchanged
    through a full `xicsrt.raytrace(config)` call, correctly split between
    the 'found' and 'lost' histories alongside `mask`.
  * `label` survives a round trip through `xicsrt_io.save_results` /
    `load_results` (HDF5 backend, i.e. `mirhdf5`).
"""

import numpy as np

import xicsrt
from xicsrt import xicsrt_io


# A minimal source that labels each ray by whether its index is even or
# odd, so the resulting label values can be checked against ray identity
# independent of anything the raytracer does internally.
_LABELED_SOURCE = '''
import numpy as np
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtSourceFocused import XicsrtSourceFocused

@dochelper
class XicsrtSourceLabeled(XicsrtSourceFocused):
    def generate_rays(self):
        rays = super().generate_rays()
        num = len(rays['mask'])
        rays['label'] = (np.arange(num) % 2).astype(int)
        return rays
'''


def _scenario_config(tmp_path, **general):
    (tmp_path / '_XicsrtSourceLabeled.py').write_text(_LABELED_SOURCE)

    config = {
        'general': {
            'number_of_iter': 2,
            'number_of_runs': 1,
            'random_seed': 42,
            'print_results': False,
            'save_config': False,
            'save_images': False,
            'save_results': False,
            'pathlist': [str(tmp_path)],
        },
        'sources': {
            'source': {
                'class_name': 'XicsrtSourceLabeled',
                'intensity': 2000,
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


def test_label_preserved_through_raytrace(tmp_path):
    """
    `label` must appear in both 'found' and 'lost' histories of every
    element, must only ever contain 0/1 (the values the source assigned),
    and must not be lost, resized, or corrupted by sorting/combining.
    """
    config = _scenario_config(tmp_path)
    output = xicsrt.raytrace(config)

    for section in ('found', 'lost'):
        for name in ('source', 'crystal', 'detector'):
            history = output[section]['history'][name]
            assert 'label' in history
            assert history['label'].dtype.kind in ('i', 'u')
            assert history['label'].shape == history['mask'].shape
            assert set(np.unique(history['label'])) <= {0, 1}

    num_found = len(output['found']['history']['detector']['mask'])
    num_lost = len(output['lost']['history']['detector']['mask'])
    assert num_found > 0
    assert num_lost > 0


def test_label_round_trip_hdf5(tmp_path):
    """
    A results dict containing `label` (int array) must survive a save/load
    round trip through the HDF5 backend with identical values and an
    integer dtype.
    """
    config = _scenario_config(tmp_path)
    output = xicsrt.raytrace(config)

    filename = tmp_path / 'results.hdf5'
    xicsrt_io.save_results(output, filename=str(filename))
    loaded = xicsrt_io.load_results(filename=str(filename))

    for section in ('found', 'lost'):
        for name in ('source', 'crystal', 'detector'):
            original = output[section]['history'][name]['label']
            restored = loaded[section]['history'][name]['label']
            np.testing.assert_array_equal(original, restored)
            assert restored.dtype.kind in ('i', 'u')
