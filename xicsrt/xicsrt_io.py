# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------

Deal with reading and writing input files for XICSRT.
"""

import numpy as np
import logging
from copy import deepcopy
import os
import pathlib

import json

from xicsrt import xicsrt_config
from xicsrt.tools import xicsrt_misc
from xicsrt.util import profiler

def load_config(filename):
    config = _dict_from_file(filename)
    return config


def save_config(config, filename=None):
    _make_output_path(config)
    if filename is None:
        filename = generate_filename(config, kind='config')
    _file_from_dict(config, filename)
    logging.info('Config saved to {}'.format(filename))


def save_results(output, filename=None):
    config = output['config']
    _make_output_path(config)
    if filename is None:
        filename = generate_filename(config, kind='results')
    _file_from_dict(output, filename)
    logging.info('History saved to {}'.format(filename))


def load_results(config=None, filename=None):
    if filename is None:
        filename = generate_filename(config, kind='results')
    output = _dict_from_file(filename)
    return output


def save_images(output, rotate=False):
    """
    Save images from the raytracing output.
    """
    from PIL import Image

    config = output['config']
    _make_output_path(config)

    for key_opt in output['config']['optics']:
        if key_opt in output['total']['image']:
            if output['total']['image'][key_opt] is not None:
                filename = generate_filename(output['config'], 'image', key_opt)

                image_temp = output['total']['image'][key_opt]
                if rotate:
                    image_temp = np.rot90(image_temp)

                generated_image = Image.fromarray(image_temp)
                generated_image.save(filename)

                logging.info('Saved image: {}'.format(filename))


def generate_filename(config, kind=None, name=None):
    config = xicsrt_config.get_config(config)
    prefix = config['general']['output_prefix']
    suffix = config['general']['output_suffix']
    run_suffix = config['general']['output_run_suffix']
    path = config['general']['output_path']

    if kind is None:
        ext = ''
    elif kind == 'image':
        ext = config['general']['image_ext']
    elif kind == 'results':
        ext = config['general']['results_ext']
    elif kind == 'config':
        ext = config['general']['config_ext']
    else:
        raise Exception(f'Data kind {kind} unknown.')

    if name is None:
        name = kind

    filename = '_'.join(filter(None, (prefix, name, suffix, run_suffix))) + ext
    filepath = os.path.join(path, filename)

    return filepath


def _dict_from_file(filename):
    ext = pathlib.Path(filename).suffix

    if ('pickle' in ext) or ('pkl' in ext):
        import pickle
        with open(filename, "rb") as ff:
            data = pickle.load(ff)

    elif ('json' in ext):
        import json
        with open(filename, "r") as ff:
            data = json.load(ff)
        data = xicsrt_misc._dict_to_numpy(data)

    elif ('hdf5' in ext) or ('h5' in ext):
        from xicsrt.util import mirhdf5
        data = mirhdf5.hdf5ToDict(filename)

    return data


def _file_from_dict(data, filename):
    ext = pathlib.Path(filename).suffix

    if ('pickle' in ext) or ('pkl' in ext):
        import pickle
        with open(filename, "wb") as ff:
            data = pickle.dump(data, ff)

    elif ('json' in ext):
        import json
        data = deepcopy(data)
        data = xicsrt_misc._dict_to_list(data)
        with open(filename, "w") as ff:
            data = json.dump(data, ff, indent=2)
        data = xicsrt_misc._dict_to_numpy(data)

    elif ('hdf5' in ext) or ('h5' in ext):
        from xicsrt.util import mirhdf5
        data = mirhdf5.dictToHdf5(data, filename)








def _make_output_path(config):
    if config['general'].get('make_directories', False):
        os.makedirs(config['general']['output_path'], exist_ok=True)
