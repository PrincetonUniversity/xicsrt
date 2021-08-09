# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>
"""
import numpy as np
import logging

import copy

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_string

from xicsrt.tools import xicsrt_misc
from xicsrt import xicsrt_config

@dochelper
class ConfigObject():
    """
    A base class for any objects with a configuration.
    """

    def __init__(self, config=None, strict=None, initialize=None):
        if initialize is None: initialize = True

        self.name = self.__class__.__name__
        self.log = logging.getLogger('xicsrt').getChild(self.name)

        self.config = self.default_config()
        self.update_config(config, strict=strict)
        self.check_config()

        self.param = copy.deepcopy(self.config)
        self.param = xicsrt_misc._convert_to_numpy(self.param, inplace=True)

        if initialize:
            self.setup()
            self.check_param()
            self.initialize()

    def default_config(self):
        """
        class_name
          Automatically generated.

        yo_mama
          Is a wonderful person!
        """

        config = dict()
        config['class_name'] = self.__class__.__name__
        config['yo_mama'] = 'Is a beautiful person and she loves you.'
        return config

    def get_config(self):
        return self.config

    def check_config(self):
        """
        Check the config before copying to the internal param. This is called
        during object instantiation (`__init__`) and therefore before `setup` is
        called.
        """
        pass

    def setup(self):
        """
        Perform any setup actions that are needed prior to initialization.
        """
        pass

    def check_param(self):
        """
        Check the internal parameters prior to initialization. This will be
        called after `setup` and before `initialize`.
        """
        pass

    def initialize(self):
        """
        Initialize the object.
        """
        pass

    def update_config(self, config_new, **kwargs):
        """
        Overwrite any config values in this object with the ones given. This
        will be done recursively for all nested dictionaries.
        """
        xicsrt_config.update_config(self.config, config_new, **kwargs)



