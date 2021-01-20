# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Antoniuk Pablant <npablant@pppl.gov>

A command line interface for the XICSRT raytracer.
"""

import numpy as np
import argparse
import logging
import io

from xicsrt import xicsrt_raytrace
from xicsrt import xicsrt_input

def _get_parser():
    parser = argparse.ArgumentParser(
        "\n    xicsrt_command.py"
        ,description = """
description:
  Perform an XICSRT raytrace.

  The input to this command should be an XICSRT configuration dictionary
  in json format.


example:
  python xicsrt_command.py config.json

"""
        ,formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=79))

    parser.add_argument(
        'config_file'
        , type=str
        , nargs='?'
        , default='config.json'
        , help='The path to the configuration file for this run.')

    parser.add_argument(
        '--suffix'
        , type=str
        , default=None
        , help="A suffix to add to the output files.")

    parser.add_argument(
        '--numruns'
        , type=int
        , default=None
        , help="Number of runs.")

    parser.add_argument(
        '--numiter'
        , type=int
        , default=None
        , help="Number of iterations per run.")

    parser.add_argument(
        '--seed'
        , type=int
        , default=None
        , help="The random seed to use.")

    return parser

def run():
    """
    Parse command line arguments and run XICSRT.
    """
    global args

    parser = _get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    config = xicsrt_input.load_config(args.config_file)

    if args.suffix:
        config['general']['output_suffix'] = args.suffix
    if args.numruns:
        config['general']['number_of_runs'] = args.numruns
    if args.numiter:
        config['general']['number_of_iter'] = args.numiter
    if args.seed:
        config['general']['random_seed'] = args.seed
        
    result = xicsrt_raytrace.raytrace(config)

# Generate the module docstring from the helpstring.
parser = _get_parser()
with io.StringIO() as ff:
    parser.print_help(ff)
    help_string = ff.getvalue()
help_string = '| '+'\n| '.join(help_string.splitlines())
__doc__ += '\n'
__doc__ += help_string

if __name__ == "__main__":
    run()
