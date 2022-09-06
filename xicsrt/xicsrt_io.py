# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
Handle reading and writing of files for XICSRT.
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

log = logging.getLogger(__name__)

def load_config(filename):
    config = _dict_from_file(filename)
    return config


def save_config(config, filename=None, path=None, mkdir=None, overwrite=None):
    if filename is None:
        filename = generate_filename(config, kind='config', path=path)
    elif path is not None:
        filename = os.path.join(path, filename)

    if mkdir is None:
        mkdir = config['general'].get('make_directories', False)
    _file_from_dict(config, filename, mkdir=mkdir, overwrite=overwrite)

    filename = pathlib.Path(filename).expanduser().resolve()
    log.info('Config saved to {}'.format(filename))


def save_results(output, filename=None, path=None, mkdir=None, overwrite=None):
    config = output['config']
    if filename is None:
        filename = generate_filename(config, kind='results', path=path)
    elif path is not None:
        filename = os.path.join(path, filename)

    if mkdir is None:
        mkdir = config['general'].get('make_directories', False)
    _file_from_dict(output, filename, mkdir=mkdir, overwrite=overwrite)

    filename = pathlib.Path(filename).expanduser().resolve()
    log.info('History saved to {}'.format(filename))


def load_results(filename=None, path=None, config=None):
    """
    Load an XICSRT results file.

    Keywords
      filename
        The name of the file to load.
        filename can be a full path; otherwise if path is given it will be
        prepended to the filename.
        If filename is not provided and a config is given, then a filename
        will be auto generated based on the config.
      path
        If provided the path will be prepended to the filename.
      config
        If provided (and filename is not), then the filename will be auto
        generated using the config settings.
    """
    print(path, filename)
    if filename is None:
        filename = generate_filename(config, kind='results', path=path)
    elif path is not None:
        filename = os.path.join(path, filename)

    print(filename)

    output = _dict_from_file(filename)
    return output


def save_images(output, rotate=True, path=None, mkdir=None):
    """
    Save images from the raytracing output.
    """
    from PIL import Image

    config = output['config']
    if mkdir is None:
        mkdir = config['general'].get('make_directories', False)

    for key_opt in output['config']['optics']:
        if key_opt in output['total']['image']:
            if output['total']['image'][key_opt] is not None:
                filename = generate_filename(output['config'], 'image', key_opt, path=path)
                if mkdir:
                    _make_path(filename)

                image_temp = output['total']['image'][key_opt]
                if rotate:
                    image_temp = np.rot90(image_temp)

                generated_image = Image.fromarray(image_temp)
                generated_image.save(filename)

                filename = pathlib.Path(filename).expanduser().resolve()
                log.info('Saved image: {}'.format(filename))


def generate_filename(config, kind=None, name=None, path=None):
    config = xicsrt_config.get_config(config)
    prefix = config['general']['output_prefix']
    suffix = config['general']['output_suffix']
    run_suffix = config['general']['output_run_suffix']
    if path is None:
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
    filename = pathlib.Path(filename).expanduser()
    ext = filename.suffix

    if ('pickle' in ext) or ('pkl' in ext):
        import pickle
        with open(filename, "rb") as ff:
            data = pickle.load(ff)

    elif ('json' in ext):
        import json
        with open(filename, "r") as ff:
            data = json.load(ff)
        data = xicsrt_misc._convert_to_numpy(data)

    elif ('hdf5' in ext) or ('h5' in ext):
        from xicsrt.util import mirhdf5
        data = mirhdf5.hdf5ToDict(filename)

    else:
        raise NotImplementedError(f'filetype: {ext} not currently supported.')

    return data


def _file_from_dict(data, filename, mkdir=False, overwrite=False):
    if mkdir:
        _make_path(filename)

    filename = pathlib.Path(filename).expanduser()
    if not overwrite and filename.exists():
        raise FileExistsError("File exists. Use overwrite=True to overwrite.")

    ext = filename.suffix

    if ('pickle' in ext) or ('pkl' in ext):
        import pickle
        with open(filename, "wb") as ff:
            pickle.dump(data, ff)

    elif ('json' in ext):
        import json
        data = deepcopy(data)
        data = xicsrt_misc._convert_from_numpy(data)
        with open(filename, "w") as ff:
            json.dump(data, ff, indent=2)

    elif ('hdf5' in ext) or ('h5' in ext):
        from xicsrt.util import mirhdf5
        mirhdf5.dictToHdf5(data, filename)

    else:
        raise NotImplementedError(f'filetype: {ext} not currently supported.')


def _make_output_path(config):
    if config['general'].get('make_directories', False):
        path = pathlib.Path(config['general']['output_path']).expanduser()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            log.info(f'Made directory: {path}')


def _make_path(filename):
    path = pathlib.Path(filename).expanduser()
    if path.suffix:
        path = path.parent
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        log.info(f'Made directory: {path}')


def path_exists(path):
    path = pathlib.Path(path).expanduser()
    return path.exists()
