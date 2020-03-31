# -*- coding: utf-8 -*-
"""
Authors:
  | Novimir Antoniuk Pablant <npablant@pppl.gov>

Purpose:
  A command line interface for the XICSRT raytracer.
"""

import numpy as np
import argparse

from xicsrt import xicsrt_raytrace
from xicsrt import xicsrt_input

def run():

    global parser
    global args

    parser = argparse.ArgumentParser(
"""
Perform an XICSRT raytrace.

The input to this command should be an XICSRT configuration dictionary
in json format.

Example:
python xicsrt.py config.json

""") 

    parser.add_argument(
        'config_file'
        ,type=str
        ,nargs='?'
        ,default='config.json'
        ,help='The path to the configuration file for this run.')

    parser.add_argument(
        '--suffix'
        ,type=str
        ,default=None
        ,help="A suffix to add to the output files.")

    args = parser.parse_args()
    config = xicsrt_input.load_config(args.config_file)

    if suffix:
        config['general']['output_suffix'] = args.suffix
        
    result = xicsrt_raytrace.raytrace_multi(config)

    
if __name__ == "__main__":
    run()
