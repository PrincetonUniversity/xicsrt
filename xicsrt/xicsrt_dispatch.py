# -*- coding: utf-8 -*-
"""
Authors:
  | Novimir Antoniuk Pablant <npablant@pppl.gov>
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

class XicsrtDispatcher():

    def __init__(self, config=None, pathlist=None):
        self.log = logging.getLogger(self.__class__.__name__)
        
        self.config = config
        self.pathlist = pathlist
        
        self.objects = OrderedDict()
        self.meta = OrderedDict()
        self.image = OrderedDict()
        self.history = OrderedDict()
        
    def check_config(self, *args, **kwargs):
        for key, obj in self.objects.items():
            obj.check_config(*args, **kwargs)

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
            
    def raytrace(self, rays, images=None, history=None):
        if history is None:
            history is False
            
        if images is None:
            images is False
            
        profiler.start('Dispatcher: raytrace')

        for key, obj in self.objects.items():
            
            profiler.start('Dispatcher: light')
            rays = obj.light(rays)
            profiler.stop('Dispatcher: light')

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
            
    def instantiate_objects(self):
        obj_info = self.find_xicsrt_objects(self.pathlist)
        
        # self.log.debug(obj_info)

        for key in self.config:
            obj = self._instantiate_object_single(obj_info, self.config[key])
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
                'filepath':filepath_list[ii]
                ,'name':name_list[ii]
                }

        return output

    def _instantiate_object_single(self, obj_info, config):
        """
        Instantiate an object from a list of filenames and a class name.
        """
        
        if config['class_name'] in obj_info:
            info = obj_info[config['class_name']]
        else:
            raise Exception('Could not find {} in available objects.'.format(config['name']))
        
        spec = importlib.util.spec_from_file_location(info['name'], info['filepath'])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, info['name'])
        obj = cls(config, initialize=False)
        
        return obj

    def get_object(self, name):
        return self.objects[name]
