# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>
"""
import numpy as np
import logging

import copy
from collections import OrderedDict

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_string

# I might want to some of these functions into this object as methods.
from xicsrt import xicsrt_input

@dochelper
class ConfigObject():
    """
    A base class for any objects with a configuration.
    """

    def __init__(self, config=None, strict=None, initialize=None):
        if initialize is None: initialize = True

        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)

        self.config = self.default_config()
        self.update_config(config, strict)
        self.check_config()

        self.param = copy.deepcopy(self.config)
        self.param = xicsrt_input.config_to_numpy(self.param)

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

        config = OrderedDict()
        config['class_name'] = self.__class__.__name__
        config['yo_mama'] = 'Is wonderful!'
        return config

    def get_config(self):
        return self.config

    def check_config(self):
        pass

    def setup(self):
        """
        Perform any setup actions that are needed prior to initialization.
        """

        # Simplify all config strings.
        xicsrt_string.simplify_strings(self.param)

    def check_param(self):
        """
        Check the internal parameters prior to initialization.
        """
        pass

    def initialize(self):
        """
        Initialize the object.
        """
        pass

    def update_config(self, config, strict=None, update=None):
        config = copy.deepcopy(config)
        self._update_config_dict(self.config, config, strict, update)

    def _update_config_dict(self, config, user_config, strict=None, update=None):
        """
        Overwrite any values in the given options dict with the values in the
        user dict.  This will be done recursively to allow nested dictionaries.

        keywords:
          strict (True)
            If True then an error will be raised if an option is found in
            the user dict that is not found in the default dict.

          update (False)
            If True any unmatched options that are found will be retained.
            When False they will simply be ignored. This option has no effect
            unless strict = False.
        """
        if strict is None:
            strict = True
        if update is None:
            update = False

        if user_config is None:
            return

        for key in user_config:
            if not key in config:
                if strict:
                    raise Exception("User option not recognized: {}".format(key))
                if update:
                    config[key] = user_config[key]
            else:
                if isinstance(config[key], dict):
                    self._update_config_dict(config[key], user_config[key], strict=strict, update=update)
                else:
                    config[key] = user_config[key]


