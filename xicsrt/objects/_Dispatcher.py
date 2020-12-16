# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
"""
import numpy as np
import logging

from collections import OrderedDict
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

        # Todo: This needs to be renamed to something more sensible.
        self.config = config
        self.section = section

        pathlist = []
        pathlist.extend(config['general'].get('pathlist_objects', []))
        pathlist.extend(config['general'].get('pathlist_default', []))
        self.pathlist = pathlist
        
        self.objects = OrderedDict()
        self.meta = OrderedDict()
        self.image = OrderedDict()
        self.history = OrderedDict()

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
            self.meta[key] = {}

    def find_xicsrt_objects(self, pathlist):

        filepath_list = []
        name_list = []

        for pp in pathlist:
            filepath_list.extend(glob.glob(os.path.join(pp, '_Xicsrt*.py')))

        for ff in filepath_list:
            filename = os.path.basename(ff)
            objectname = os.path.splitext(filename)[0]
            objectname = objectname[1:]
            name_list.append(objectname)

        output = OrderedDict()
        for ii, ff in enumerate(name_list):
            output[ff] = {
                'filepath': filepath_list[ii]
                , 'name': name_list[ii]
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
        config = OrderedDict()
        for key, obj in self.objects.items():
            config[key] = obj.get_config(*args, **kwargs)
        return config

    def setup(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.setup(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.initialize(*args, **kwargs)

    def generate_rays(self, history=None):
        """
        Generates rays from all sources.
        """
        if history is None:
            history is False
            
        if len(self.objects) == 0:
            raise Exception('No ray sources defined.')
        elif not len(self.objects) == 1:
            raise NotImplementedError('Multiple ray sources are not currently supported.')
                
        for key, obj in self.objects.items():
            rays = obj.generate_rays()

            self.meta[key]['num_out'] = np.sum(rays['mask'])
            
            if history:
                self.history[key] = deepcopy(rays)
                
        return rays
            
    def trace(self, rays, images=None, history=None):
        if history is None:
            history is False
            
        if images is None:
            images is False
            
        profiler.start('Dispatcher: raytrace')

        for key, obj in self.objects.items():
            
            profiler.start('Dispatcher: trace_global')
            rays = obj.trace_global(rays)
            profiler.stop('Dispatcher: trace_global')

            self.meta[key]['num_out'] = np.sum(rays['mask'])
            
            if history:
                self.history[key] = deepcopy(rays)

            if images:
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
            for filter in filters.objects:
                if str(filters.objects[filter].name) in self.objects[key].config['filter_list']:
                    self.objects[key].filter_objects.append(filters.objects[filter])

