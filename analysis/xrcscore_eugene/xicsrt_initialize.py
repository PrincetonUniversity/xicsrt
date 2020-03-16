# -*- coding: utf-8 -*-
"""
Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
Takes the config dictionary created by xicsrt_iterstudy.py and runs it through
one of the scenarios in xicsrt_scenarios.py, returning a modified config
dictionary.
"""
from xicsrt_scenarios import setup_real_scenario
from xicsrt_scenarios import setup_throughput_scenario
from xicsrt_scenarios import setup_plasma_scenario
from xicsrt_scenarios import setup_beam_scenario
from xicsrt_scenarios import setup_manfred_scenario
from xicsrt_scenarios import setup_graphite_test
from xicsrt_scenarios import setup_crystal_test
from xicsrt_scenarios import setup_source_test


def initialize(config):
    scenario = str.upper(config['general_input']['scenario'])
    
    ## Set up a real scenario 
    if   scenario == 'REAL':
        config = setup_real_scenario(config)

    ## Set up a plasma test scenario 
    elif scenario == 'PLASMA':
        config = setup_plasma_scenario(config)
        
    ## Set up a throughput test scenario 
    elif scenario == 'THROUGHPUT':
        config = setup_throughput_scenario(config)

    ## Set up a beamline test scenario 
    elif scenario == 'BEAM' or scenario == 'MODEL':
        config = setup_beam_scenario(config)
        
    elif scenario == 'MANFRED':
        config = setup_manfred_scenario(config)

    ## Set up a crystal test scenario 
    elif scenario == 'CRYSTAL':
        config = setup_crystal_test(config)

    ## Set up a graphite test scenario 
    elif scenario == 'GRAPHITE':
        config = setup_graphite_test(config)

    ## Set up a source test scenario 
    elif scenario == 'SOURCE':
        config = setup_source_test(config)

    else:
        raise Exception('Scenario not defined: {}'.format(scenario))

    ## Backwards raytracing involves swapping the source and detector 
    if config['general_input']['backwards_raytrace']:
        swap_position   = config['source_input']['position']
        swap_orientation= config['source_input']['orientation']
        swap_normal     = config['source_input']['normal']

        config['source_input']['position']     = config['dector_input']['position']
        config['source_input']['orientation']  = config['dector_input']['orientation']
        config['source_input']['normal']       = config['dector_input']['normal']

        config['dector_input']['position']     = swap_position
        config['dector_input']['orientation']  = swap_orientation
        config['dector_input']['normal']       = swap_normal

    ## Simulate linear thermal expansion 
    # This is calculated AFTER spectrometer geometry setup to simulate non-ideal conditions
    if config['general_input']['ideal_geometry'] is False:
        config['crystal_input']['spacing']  *= 1 + config['crystal_input']['therm_expand']  * (config['general_input']['xics_temp'] - 273)
        config['graphite_input']['spacing'] *= 1 + config['graphite_input']['therm_expand'] * (config['general_input']['xics_temp'] - 273)

    return config


def initialize_multi(config_multi):
    for ii in range(len(config_multi)):
        config_multi[ii] = initialize(config_multi[ii])

    return config_multi

