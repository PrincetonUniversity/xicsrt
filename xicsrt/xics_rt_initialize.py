# -*- coding: utf-8 -*-
"""
Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------

"""

from xicsrt.util import profiler

profiler.start('Import Time')
import numpy as np

from xicsrt.xics_rt_scenarios import setup_beam_scenario, setup_plasma_scenario
from xicsrt.xics_rt_scenarios import setup_crystal_test, setup_graphite_test, setup_source_test

profiler.stop('Import Time')


def initialize(config):

    profiler.start('Scenario Setup Time')

    ## Temporary
    config['source_input']['intensity'] = config['general_input']['number_of_rays']

    scenario = str.lower(config['general_input']['scenario'])

    ## Set up a plasma test scenario ----------------------------------------------
    if scenario == 'plasma':
        config = setup_plasma_scenario(config)

    ## Set up a beamline test scenario --------------------------------------------
    elif scenario == 'beam' or scenario == 'model':
        config = setup_beam_scenario(config)

    ## Set up a crystal test scenario ---------------------------------------------
    elif scenario == 'crystal':
        config = setup_crystal_test(config)

        config['graphite_input']['position']     = config['crystal_input']['position']
        config['graphite_input']['normal']       = config['crystal_input']['normal']
        config['graphite_input']['orientation']  = config['crystal_input']['orientation']

    ## Set up a graphite test scenario --------------------------------------------
    elif scenario == 'graphite':
        config = setup_graphite_test(config)

        config['crystal_input']['position']       = config['graphite_input']['position']
        config['crystal_input']['normal']         = config['graphite_input']['normal']
        config['crystal_input']['orientation']    = config['graphite_input']['orientation']

    ## Set up a source test scenario ----------------------------------------------
    elif scenario == 'source':
        config = setup_source_test(config)

    else:
        raise Exception('Scenario not defined: {}'.format(scenario))

    ## Backwards raytracing involves swapping the source and detector -------------
    if config['general_input']['backwards_raytrace']:
        swap_position   = config['source_input']['position']
        swap_orientation= config['source_input']['orientation']
        swap_normal     = config['source_input']['normal']

        config['source_input']['position']    = config['dector_input']['position']
        config['source_input']['orientation'] = config['dector_input']['orientation']
        config['source_input']['normal']      = config['dector_input']['normal']

        config['dector_input']['position']     = swap_position
        config['dector_input']['orientation']  = swap_orientation
        config['dector_input']['normal']       = swap_normal

    ## Simulate linear thermal expansion ------------------------------------------
    # This is calculated AFTER spectrometer geometry setup to simulate non-ideal conditions
    if config['general_input']['ideal_geometry'] is False:
        config['crystal_input']['spacing']  *= 1 + config['crystal_input']['therm_expand']  * (config['general_input']['xics_temp'] - 273)
        config['graphite_input']['spacing'] *= 1 + config['graphite_input']['therm_expand'] * (config['general_input']['xics_temp'] - 273)

    profiler.stop('Scenario Setup Time')

    return config


def initialize_multi(config_multi):
    for key in enumerate(config_multi):
        config_multi[key] = initialize(config_multi[key])

    return config_multi

