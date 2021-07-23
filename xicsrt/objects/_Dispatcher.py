# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
"""
import numpy as np
import logging

import importlib
import glob
import os

from copy import deepcopy
import importlib.util

from xicsrt.util import profiler

class Dispatcher():
    """
    A class to help find, initialize and then dispatch calls to
    raytracing objects.

    A dispatcher is used within XICSRT to find and instantiate objects based
    on their specification within the config dictionary. These objects are
    then tracked within the dispatcher, allowing methods to be called on all
    objects sequentially.
    """

    def __init__(self, config=None, section=None):
        self.log = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.section = section

        pathlist = []
        pathlist.extend(config['general'].get('pathlist', []))
        pathlist.extend(config['general'].get('pathlist_default', []))
        self.pathlist = pathlist
        
        self.objects = dict()
        self.meta = dict()
        self.image = dict()
        self.history = dict()

    def instantiate(self, names=None):
        if names is None:
            names = self.config[self.section].keys()
        elif isinstance(names, str):
            names = [names]

        strict = self.config['general']['strict_config_check']

        obj_info = self.find_xicsrt_objects(self.pathlist)
        # self.log.debug(obj_info)

        for key in names:
            obj = self._instantiate_single(
                obj_info
                ,self.config[self.section][key]
                ,strict=strict)
            self.objects[key] = obj

    def find_xicsrt_objects(self, pathlist):
        """
        Return a dictionary with all the XICSRT objects found in the given
        list of paths. Objects are identified by looking for python files
        that start with '_Xicsrt' prefix.

        Programming Notes
        -----------------
        If a given path does not exist glob will just return and empty list.
        For this reason no path existence checking is needed (unless we want
        to raise a user friendly error).
        """

        filepath_list = []
        name_list = []

        for pp in pathlist:
            filepath_list.extend(glob.glob(os.path.join(pp, '_Xicsrt*.py')))

        for ff in filepath_list:
            filename = os.path.basename(ff)
            objectname = os.path.splitext(filename)[0]
            objectname = objectname[1:]
            name_list.append(objectname)

        output = dict()
        for ii, ff in enumerate(name_list):
            output[ff] = {
                'filepath': filepath_list[ii],
                'name': name_list[ii]
            }

        return output

    def _instantiate_single(self, obj_info, config, strict=None):
        """
        Instantiate an object from a list of filenames and a class name.
        """

        if config['class_name'] in obj_info:
            info = obj_info[config['class_name']]
        else:
            raise Exception('Could not find {} in available objects.'.format(config['class_name']))

        spec = importlib.util.spec_from_file_location(info['name'], info['filepath'])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, info['name'])
        obj = cls(config, initialize=False, strict=strict)

        return obj

    def get_object(self, name):
        if not name in self.objects:
            self.instantiate(name)
        return self.objects[name]

    def check_config(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.check_config(*args, **kwargs)

    def check_param(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.check_param(*args, **kwargs)

    def get_config(self, *args, **kwargs):
        config = dict()
        for key, obj in self.objects.items():
            config[key] = obj.get_config(*args, **kwargs)
        return config

    def setup(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.setup(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.initialize(*args, **kwargs)

    def generate_rays(self, keep_meta=None, keep_history=None):
        """
        Generates rays from all sources.
        """
        if keep_meta is None: keep_meta = True
        if keep_history is None: keep_history = False
            
        if len(self.objects) == 0:
            raise Exception('No ray sources defined.')
        elif not len(self.objects) == 1:
            raise NotImplementedError('Multiple ray sources are not currently supported.')
                
        for key, obj in self.objects.items():
            rays = obj.generate_rays()

            if keep_meta:
                self.meta[key] = dict()
                self.meta[key]['num_out'] = np.sum(rays['mask'])
            
            if keep_history:
                self.history[key] = deepcopy(rays)
                
        return rays
            
    def trace(self, rays, keep_meta=None, keep_history=None, keep_images=None):
        """
        Perform raytracing for each object in sequence.
        """
        if keep_meta is None: keep_meta = True
        if keep_history is None: keep_history = False
        if keep_images is None: keep_images = False

        profiler.start('Dispatcher: raytrace')

        for key, obj in self.objects.items():
            
            profiler.start('Dispatcher: trace_global')
            rays = obj.trace_global(rays)
            profiler.stop('Dispatcher: trace_global')

            if keep_meta:
                self.meta[key] = dict()
                self.meta[key]['num_out'] = np.sum(rays['mask'])
            
            if keep_history:
                self.history[key] = deepcopy(rays)

            if keep_images:
                profiler.start('Dispatcher: collect')
                self.image[key] = obj.make_image(rays)
                profiler.stop('Dispatcher: collect')
                
        profiler.stop('Dispatcher: raytrace')
        
        return rays
    
    def apply_filters(self, filters):
        # Used by dispatcher objects to apply filters.
        # 'filters' is a filter dispatcher object that contains filter objects
        # 'self' should be a source or optics dispatcher object
        
        # read the filter list for each source and dispatch the matching filters
        for key in self.objects:
            if not 'filters' in self.objects[key].config:
                break
            if self.objects[key].config['filters'] is None:
                break
            for filter_name in filters.objects:
                if filter_name in self.objects[key].config['filters']:
                    self.objects[key].filter_objects.append(filters.objects[filter_name])

