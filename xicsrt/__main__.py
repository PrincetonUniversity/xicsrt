# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Antoniuk Pablant <npablant@pppl.gov>

A command line interface for the XICSRT raytracer.
"""

import sys
import numpy as np
import argparse
import io

from xicsrt.util import mirlogging

from xicsrt import xicsrt_raytrace
from xicsrt import xicsrt_multiprocessing
from xicsrt import xicsrt_io

from xicsrt._version import __version__

m_log = mirlogging.getLogger(__name__)

def _get_parser():
    parser = argparse.ArgumentParser(
        "\n  xicsrt"
        ,description = f"""
xicsrt version {__version__}

description:
  Perform an XICSRT raytrace from the command line.

  The input to this command should be an XICSRT configuration dictionary
  in json format. (Pickle and hdf5 formats are also supported.)


example 1:
  xicsrt config.json
  
example 2: 
  python -m xicsrt config.json

"""
        ,formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, width=79))

    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        default='config.json',
        help='The path to the configuration file for this run.',
        )

    parser.add_argument(
        '--numruns',
        type=int,
        default=None,
        metavar='N',
        help="Number of runs.",
        )

    parser.add_argument(
        '--numiter',
        type=int,
        default=None,
        metavar='N',
        help="Number of iterations per run.",
        )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        metavar='N',
        help="The random seed to use.",
        )

    parser.add_argument(
        '--save',
        action='store_true',
        help="Save the results.",
        )

    parser.add_argument(
        '--images',
        action='store_true',
        help="Save intersection images.",
        )

    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        metavar='STR',
        help="A suffix to add to the output files.",
        )

    parser.add_argument(
        '--path',
        type=str,
        default=None,
        metavar='STR',
        help="Directory in which to store output.",
        )

    parser.add_argument(
        '--multiprocessing',
        '--mp',
        action='store_true',
        help="Use multiprocessing.",
        )

    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        metavar='N',
        help="Number of processes to use for muliprocessing.",
        )

    parser.add_argument(
        '--version',
        action='store_true',
        help="Show the version number.",
        )

    parser.add_argument(
        '--debug',
        action='store_true',
        help="Show debugging output in the log.",
        )

    return parser

def run():
    """
    Parse command line arguments and run XICSRT.
    """
    global args

    parser = _get_parser()
    args = parser.parse_args()

    if args.version:
        print(f'{__version__}')
        return

    if args.debug:
        log_level = mirlogging.DEBUG
    else:
        log_level = mirlogging.INFO
    mirlogging.defaultConfig(level=log_level, force=True)

    config = xicsrt_io.load_config(args.config_file)

    if args.suffix:
        config['general']['output_suffix'] = args.suffix
    if args.numruns:
        config['general']['number_of_runs'] = args.numruns
    if args.numiter:
        config['general']['number_of_iter'] = args.numiter
    if args.seed:
        config['general']['random_seed'] = args.seed
    if args.path:
        config['general']['output_path'] = args.path
    if args.save:
        config['general']['save_results'] = args.save
    if args.images:
        config['general']['save_images'] = args.images

    if args.multiprocessing:

        result = xicsrt_multiprocessing.raytrace(config, processes=args.processes)
    else:
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
