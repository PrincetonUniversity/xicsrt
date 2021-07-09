# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A collections of routines to simplify interactive use of XICSRT.
"""
import numpy as np

from xicsrt import xicsrt_config
from xicsrt.objects._Dispatcher import Dispatcher

def get_element(config_user, name, section=None, initialize=True):
    """
    Retrieves an raytracing element (source, optic or filter) object.
    """
    config = xicsrt_config.get_config(config_user)
    if section is None:
        section = _find_element_section(config, name)

    disp = Dispatcher(config, section)
    disp.instantiate(name)
    obj = disp.get_object(name)
    if initialize:
        obj.setup()
        obj.check_param()
        obj.initialize()
    return obj

def _find_element_section(config, name):
    """
    Search config for the given element name and return the section.
    """
    section_list = ['optics', 'sources', 'filters']
    out_list = []
    for section in section_list:
        if name in config[section]:
            out_list.append(section)

    if len(out_list) == 0:
        raise Exception(f'Could not find element: {name} in any section.')
    if len(out_list) > 1:
        raise Warning(f'Element name: {name} was found in more than one section.'
                      f' Please provide an explicit section name.')

    return out_list[0]
