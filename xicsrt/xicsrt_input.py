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

import json

from xicsrt.util import profiler

def load_config(filepath):
    with open(filepath, 'r') as ff:
        config = json.load(ff)
    config_to_numpy(config)
    return config

def save_config(config, filepath=None):

    if config['general'].get('make_directories', False):
        os.makedirs(config['general']['output_path'], exist_ok=True)

    if filepath is None:
        filepath = os.path.join(config['general']['output_path'], 'config.json')
    config_out = deepcopy(config)
    config_out = _dict_to_list(config_out)
    config_to_list(config_out)
    with open(filepath, 'w') as ff:
        json.dump(config_out, ff, indent=1)
    logging.info('Config saved to {}'.format(filepath))

def config_to_numpy(obj):
    _dict_to_numpy(obj)
    return obj

def config_to_list(obj):
    _dict_to_list(obj)
    return obj

def _dict_to_numpy(obj):
    for key in obj:
        if isinstance(obj[key], list):
            obj[key] = np.array(obj[key])
        elif isinstance(obj[key], dict):
            obj[key] = _dict_to_numpy(obj[key])
    return obj

def _dict_to_list(obj):
    for key in obj:
        if isinstance(obj[key], np.ndarray):
            obj[key] = obj[key].tolist()
        elif isinstance(obj[key], dict):
            obj[key] = _dict_to_list(obj[key])
    return obj
