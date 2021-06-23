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

from xicsrt import xicsrt_input
from xicsrt.objects._ConfigObject import ConfigObject

class XicsrtGeneralConfig(ConfigObject):

    def default_config(self):
        """
        number_of_iter: int (1)
          Number of raytracing iterations to perform for each raytracing run.
          Iterations are performed in a single process and with a single
          initialization of the raytracing objects and the random seed.
          All iterations will be combined before saving images or output files.

        number_of_runs: int (1)
          Number of raytracing runs to perform. For each run, the specified
          number of iterations will be performed. Raytracing runs can be
          performed in separate processes enabling the use of multiprocessing.
          At the start of each run all raytracing objects will be created and
          initialized. Images and output files will be saved after each run.

        random_seed: int (None)
          A random seed to initialize the pseudo-random number generator. If
          random_seed is equal to None then a random seed will be provided by
          the python/numpy internals. When a integer seed is provided and
          multiple raytracing runs are performed, the random seed will be
          incremented by one for each successive run. This seed can be used to
          make raytracing runs reproducible on the level of individual rays;
          however, reproducibility is not guaranteed between different versions
          of XICSRT or different versions of Python. Random seed initialization
          performed using `np.random.seed()`.

        pathlist_objects: list (list())
          A list of paths that contain user defined raytracing objects (sources,
          optics, or filters). These paths will be searched for filenames that
          match the requested 'class_name' in the object config. User defined
          paths are searched before the builtin or contrib paths.

        pathlist_default: list
          A list of paths to the builtin and contributed raytracing objects.
          This option should not be changed by the user, but can be useful
          for inspection to see what directories are actually being searched.

        strict_config_check: bool (True)
          Use strict checking to ensure that the config dictionary only contains
          valid options for the given object. This helps avoid unexpected
          behavior as well and alerting to typos in the configuration. When set
          to `False`, unrecognized config options will be quietly ignored.

        output_path: str (None)
          Path in which to save images and results from the raytracing run. If
          note then the current working path will be used. Use the option
          `make_directories` to control directory creation.

        output_prefix: str ("xicsrt")
          Filenames for images and results are automatically generated. Use
          this option to add a prefix to the beginning of all filenames. An
          underscore will be automatically added after the prefix. For images
          the following format will be used: "prefix_optic_suffix_run.tif".
          For example: "xicsrt_detector_scan01_0000.tif".

        output_suffix: str (None)
          If present this string will be added to automatically generated
          filenames after the optic name and before the run_suffix. See the
          option `output_prefix` for more information.

        output_run_suffix: str (None)
          This options is used internally and should not be set by the user.

        image_extension: str ('.tif')
          The image extension controls the file format of the same images.
          Any format supported by the `pillow` package can be used, however
          *.tif images are recommended.

        make_directories: bool (False)
          Controls whether the output path should be created if it does not
          already exist. If False an error will be raised if the output path
          does not exist.

        keep_meta: bool (True)
        keep_images: bool (True)
        keep_history: bool (True)
        history_max_lost: int (10000)

        save_config: bool (False)
          Option whether or not to save the config dictionary. Output format
          is currently limited to json format (hdf5 and pickle coming soon).

        save_images: bool (False)
          Option whether or not to save images to the output path. Images will
          be saved for every run. Image format will be determined by the option
          `image_extension`. Control of output for individual optics can be
          set within the object specific config section.

        save_meta: bool (False)
          Not currently implemented.

        save_history: bool (False)
          Not currently implemented)

        print_results: bool (True)
          Control text output to terminal of raytracing summary and optics
          specific information. (Note: control of logging/debugging output is
          controlled through a separate option that is not yet implemented.)

        """
        config = super().default_config()

        config['general'] = dict()
        config['general']['number_of_iter'] = 1
        config['general']['number_of_runs'] = 1
        config['general']['random_seed'] = None
        config['general']['pathlist_objects'] = []
        config['general']['pathlist_default'] = get_pathlist_default()
        config['general']['strict_config_check'] = True

        config['general']['output_path'] = None
        config['general']['output_prefix'] = 'xicsrt'
        config['general']['output_suffix'] = None
        config['general']['output_run_suffix'] = None
        config['general']['image_extension'] = '.tif'
        config['general']['make_directories'] = False

        config['general']['keep_meta'] = True
        config['general']['keep_images'] = True
        config['general']['keep_history'] = True

        config['general']['history_max_lost'] = 10000

        config['general']['save_config'] = False
        config['general']['save_images'] = False
        config['general']['save_meta'] = False
        config['general']['save_history'] = False

        config['general']['print_results'] = True

        config['sources'] = dict()
        config['optics'] = dict()
        config['filters'] = dict()
        config['scenario'] = dict()
    
        return config
        
def get_config(config_user=None):
    obj_config =  XicsrtGeneralConfig()
    obj_config.update_config(config_user, strict=False, update=True)
    config = obj_config.get_config()
    return config

def update_config(config, config_user):
    """
    Update a given config with the values from a second config.

    This is helpful when updating a default config with user values.
    """
    obj_config = XicsrtGeneralConfig()
    obj_config.update_config(config, strict=False, update=True)
    obj_config.update_config(config_user, strict=False, update=True)
    config_out = obj_config.get_config()
    return config_out

def config_to_numpy(config):
    # Temporarily just call the routine from xicsrt_input.
    # This should actually go the opposite way.
    config = xicsrt_input.config_to_numpy(config)
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
    try:
        import xicsrt_contrib
    except:
        logging.debug('The xicrsrt_contrib package is not installed.')
        return pathlist

    # Add paths to the xicsrt_contrib objects.
    path_module = os.path.dirname(os.path.abspath(xicsrt_contrib.__file__))
    pathlist.append(os.path.join(path_module, 'filters'))
    pathlist.append(os.path.join(path_module, 'sources'))
    pathlist.append(os.path.join(path_module, 'optics'))
    logging.debug('The xicrsrt_contrib objects are available.')

    return pathlist
