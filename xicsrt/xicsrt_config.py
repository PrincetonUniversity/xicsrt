# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
"""


import numpy as np
import logging
import os
import copy
from xicsrt.objects._ConfigObject import ConfigObject
from xicsrt.tools import xicsrt_misc

from xicsrt import _version

log = logging.getLogger(__name__)

try:
    import xicsrt_contrib
except:
    log.debug('The xicrsrt_contrib package is not installed.')
else:
    log.debug('The xicrsrt_contrib package is installed.')


def default_config():
    """
    number_of_iter : int (1)
      Number of raytracing iterations to perform for each raytracing run.
      Iterations are performed in a single process and with a single
      initialization of the raytracing objects and the random seed.
      All iterations will be combined before saving images or output files.

    number_of_runs : int (1)
      Number of raytracing runs to perform. For each run, the specified
      number of iterations will be performed. Raytracing runs can be
      performed in separate processes enabling the use of multiprocessing.
      At the start of each run all raytracing objects will be created and
      initialized. Images and output files will be saved after each run.

    random_seed : int (None)
      A random seed to initialize the pseudo-random number generator. If
      random_seed is equal to None then a random seed will be provided by
      the python/numpy internals. When a integer seed is provided and
      multiple raytracing runs are performed, the random seed will be
      incremented by one for each successive run. This seed can be used to
      make raytracing runs reproducible on the level of individual rays;
      however, reproducibility is not guaranteed between different versions
      of XICSRT or different versions of Python. Random seed initialization
      performed using `np.random.seed()`.

    pathlist : list (list())
      A list of paths that contain user defined raytracing modules (sources,
      optics, filters, apertures, etc.). These paths will be searched for
      filenames that match the requested 'class_name' in the object config.
      User defined paths are searched before the builtin or contrib paths.

    pathlist_default : list
      A list of paths to the builtin and contributed raytracing objects.
      This option should not be changed by the user, but can be useful
      for inspection to see what directories are actually being searched.

    strict_config_check : bool (True)
      Use strict checking to ensure that the config dictionary only contains
      valid options for the given object. This helps avoid unexpected
      behavior as well and alerting to typos in the configuration. When set
      to `False`, unrecognized config options will be quietly ignored.

    output_path : str (None)
      Path in which to save images and results from the raytracing run. If
      note then the current working path will be used. Use the option
      `make_directories` to control directory creation.

    output_prefix : str ("xicsrt")
      Filenames for images and results are automatically generated. Use
      this option to add a prefix to the beginning of all filenames. An
      underscore will be automatically added after the prefix. For images
      the following format will be used: "prefix_optic_suffix_run.tif".
      For example: "xicsrt_detector_scan01_0000.tif".

    output_suffix : str (None)
      If present this string will be added to automatically generated
      filenames after the optic name and before the run_suffix. See the
      option `output_prefix` for more information.

    output_run_suffix : str (None)
      This option is used internally and should not be set by the user.

    image_ext : str ('.tif')
      Controls the file format of the saved images. Any format supported by
      the `pillow` package can be used; .tif images are recommended.

    results_ext : str ('.hdf5')
      Controls the file format for saving the results dictionary. Currently
      hdf5 (.hdf5, .h5), pickle (.pickle, .pkl) and json (.json) file
      formats are supported. The json format is not recommended as it may
      lead to very large file sizes!

    make_directories : bool (False)
      Controls whether the output path should be created if it does not
      already exist. If False an error will be raised if the output path
      does not exist.

    keep_meta : bool (True)
      Controls whether to calculate and keep metadata and statistics
      relating to the raytracing.

    keep_images : bool (True)
      Controls whether to generate and keep pixelated images of the ray
      intersections at each optic.  Control of image generation for
      individual optics can also be set within the object specific
      config sections.

    keep_history : bool (True)
      Controls whether to calculate and keep the raytracing history. Rays
      will be sorted into 'lost' and 'found' rays, where found rays are
      those that reach the final optic element. The 'found' ray history
      will be kept in full, the 'lost' ray history will be truncated (see
      option `history_max_lost`).

      The ray history provides a great deal of information about the
      raytracing and enables ray plotting (2d and 3d) and post processing.
      However, turning on the ray history also *greatly* increases memory
      usage since the rays must be duplicated and saved for every optical
      element. If only final intersection images are required, consider
      setting this option to `False` to improve raytracing performance.

    history_max_lost : int (10000)
      Number of 'lost' rays to retain in the raytrace history. Lost rays
      are those that are launched from the source but that do not reach the
      last optical element (typically a detector). For many x-ray raytracing
      applications the number of lost rays will be very large, and retention
      of all lost rays would quickly exhaust available system memory. To
      avoid memory issues, while still retaining some lost rays for
      diagnostic purposes, a randomized truncation of the lost rays is
      performed.

    save_config : bool (False)
      Option whether or not to save the config dictionary. Output format
      is currently limited to json format (hdf5 and pickle coming soon).

    save_images : bool (False)
      Controls saving of images. Images will be saved for every run, and a
      combined image will be saved at the conclusion of all runs. Control of
      output for individual optics can be set within the object specific
      config sections. Image format will be determined by the option
      `image_ext` (default .tif). Images will be saved to the `output_path`.

    save_results : bool (False)
      Controls saving of the raytracing results dictionary. The contents of
      the results dictionary are controlled by `keep_meta`, `keep_images`
      and `keep_history`. Results will only be saved after all runs are
      completed. Output format will be determined by the option
      `results_ext` (default .hdf5). OUtput will be saved to the
      `output_path`.

    print_results : bool (True)
      Control text output to terminal of raytracing summary and optics
      specific information. (Note: control of logging/debugging output is
      controlled through a separate option that is not yet implemented.)

    version: string
      The version number of xicsrt when this config was created.
      This option is set internally and should not be modified by the user.
    """
    config = dict()
    config['general'] = dict()
    config['general']['version'] = _version.__version__
    config['general']['number_of_iter'] = 1
    config['general']['number_of_runs'] = 1
    config['general']['random_seed'] = None
    config['general']['pathlist'] = []
    config['general']['pathlist_default'] = get_pathlist_default()
    config['general']['strict_config_check'] = True

    config['general']['output_path'] = None
    config['general']['output_prefix'] = 'xicsrt'
    config['general']['output_suffix'] = None
    config['general']['output_run_suffix'] = None
    config['general']['image_ext'] = '.tif'
    config['general']['results_ext'] = '.hdf5'
    config['general']['config_ext'] = '.json'
    config['general']['make_directories'] = False

    config['general']['keep_meta'] = True
    config['general']['keep_images'] = True
    config['general']['keep_history'] = True

    config['general']['history_max_lost'] = 10000

    config['general']['save_config'] = False
    config['general']['save_images'] = False
    config['general']['save_results'] = False

    config['general']['print_results'] = True

    config['sources'] = dict()
    config['optics'] = dict()
    config['filters'] = dict()
    config['scenario'] = dict()

    return config


def get_config(config_user=None):
    config = default_config()
    update_config(config, config_user, strict=False, update=True)

    return config


def refresh_config(config_new):
    """
    When a config file is loaded from a another system or from a different user
    it may contain default values that are not appropriate for the current
    environment. This function will overwrite these options with new default
    values where appropriate.
    """

    refresh = {}
    refresh['general'] = {}
    refresh['general']['pathlist_default'] = None

    config_new = copy.deepcopy(config_new)
    update_config(
        config_new,
        refresh,
        strict=False,
        update=True,
        ignore_none=False,
        )

    config = default_config()
    update_config(
        config,
        config_new,
        strict=False,
        update=True,
        ignore_none=True,
        )

    return config


def get_pathlist_default():
    """
    Return a list of the default sources and optics directories.
    These locations will be based on the location of this module.
    """

    pathlist = []
    pathlist = _add_pathlist_builtin(pathlist)
    pathlist = _add_pathlist_contrib(pathlist)
    return pathlist


def _add_pathlist_builtin(pathlist):
    # Add paths to built-in objects.
    path_module = os.path.dirname(os.path.abspath(__file__))
    pathlist.append(os.path.join(path_module, 'filters'))
    pathlist.append(os.path.join(path_module, 'sources'))
    pathlist.append(os.path.join(path_module, 'optics'))

    return pathlist


def _add_pathlist_contrib(pathlist):
    # Check if the xicsrt_contrib module was successfully imported.
    if not 'xicsrt_contrib' in globals():
        return pathlist

    # Add paths to the xicsrt_contrib objects.
    path_module = os.path.dirname(os.path.abspath(xicsrt_contrib.__file__))
    pathlist.append(os.path.join(path_module, 'filters'))
    pathlist.append(os.path.join(path_module, 'sources'))
    pathlist.append(os.path.join(path_module, 'optics'))

    return pathlist


def config_to_numpy(obj):
    xicsrt_misc._convert_to_numpy(obj)
    return obj


def config_from_numpy(obj):
    xicsrt_misc._convert_from_numpy(obj)
    return obj


def update_config(
        config,
        config_new,
        strict=None,
        update=None,
        ignore_none=None,
        ):
    """
    Overwrite any values in the given config dict with the values in the
    config_new dict.  This will be done recursively to allow nested
    dictionaries.

    keywords:
      strict (True)
        If True then an error will be raised if an option is found in
        the user dict that is not found in the default dict.

      update (False)
        If True any unmatched options that are found will be retained.
        When False they will simply be ignored. This option has no effect
        unless strict = False.

      ignore_none (False)
        If True any options found in config_new with a value of None will
        be ignored.

    """
    _update_config_dict(config, config_new, strict, update, ignore_none)


def _update_config_dict(
        config,
        config_new,
        strict=None,
        update=None,
        ignore_none=None,
        ):
    """
    Recursive worker function for `update_config`.
    """
    if strict is None:
        strict = True
    if update is None:
        update = False
    if ignore_none is None:
        ignore_none = False

    if config_new is None:
        return

    for key in config_new:
        if not key in config:
            if strict:
                raise Exception("User option not recognized: {}".format(key))
            if update:
                config[key] = config_new[key]
        else:
            if isinstance(config[key], dict):
                _update_config_dict(
                    config[key],
                    config_new[key],
                    strict=strict,
                    update=update,
                    ignore_none=ignore_none
                    )
            else:
                if ignore_none and config_new[key] is None:
                    pass
                else:
                    config[key] = config_new[key]
