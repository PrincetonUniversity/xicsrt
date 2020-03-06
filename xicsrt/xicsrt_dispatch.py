# -*- coding: utf-8 -*-
"""
Authors:
  | Novimir Antoniuk Pablant <npablant@pppl.gov>
"""

from collections import OrderedDict
import importlib
import glob
import os

import importlib.util
        
class XicsrtDispatcher():

    def __init__(self):
        self.objects = OrderedDict()

    def instantiateObjects(self, config_dict, path_list):
        obj_info = self.findXicsrtObjects(path_list)
        print(obj_info)
        for key in config_dict:
            obj = self._instantiateObjectSingle(obj_info, config_dict[key])
            self.objects[key] = obj
    
    def findXicsrtObjects(self, path_list):

        filepath_list = []
        name_list = []

        for pp in path_list:
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

    def _instantiateObjectSingle(self, obj_info, config):
        
        if config['name'] in obj_info:
            info = obj_info[config['name']]
        else:
            raise Exception('Could not find {} in available objects.'.format(config['name']))
        
            
        spec = importlib.util.spec_from_file_location(info['name'], info['filepath'])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        obj = mod.XicsrtPlasmaGeneric(config)

        return obj
