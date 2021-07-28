# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A set of utility routines to load bragg reflection data files from
several external applications including x0h, XOP and SHADOW.

Data will be returned in a standardized dictionary.
"""

import numpy as np
import os

import logging
log = logging.getLogger(__name__)

def read(filename, filetype=None):
    """
    The main routine to read Bragg reflection data files.

    This will switch between the various format specific routines
    based on the filetype parameter.

    Parameters
    ----------
    filename : str
      The file to read.

    Keyword Arguments
    -----------------
    filetype : str
      The source of the given file. If not provided, the filetype will be
      guessed. Currently supported types are: 'xop', 'x0h'.
    """

    # Try to guess the filetype if it was not provided.
    if filetype is None:
        filetype = _guess_filetype(filename)
        m_log.info(f'Rocking filetype guess: {filetype}')
    if filetype is None:
        raise Exception('Could not guess the filetype. Please use the filetype keyword.')

    if filetype == 'xop':
        out = read_xop(filename)
    elif filetype == 'x0h':
        raise NotImplementedError(f'A reader for filetype {filetype} not yet implemented.')
    else:
        raise Exception(f'Filetype {filetype} not recognized.')

    return out

def _guess_filetype(filename):
    pathname, basename = os.path.split(filename)
    rootname, extname = os.path.splitext(basename)
    if rootname == 'diff_pat':
        filetype = 'xop'
    else:
        filetype = None
    return filetype

def _read_xop__diff_pat_dat__header(filename):
    header = ''
    with open(filename) as ff:
        while True:
            line = ff.readline()
            if line[0] != '#':
                break
            header += line
    return header

def _read_xop__diff_pat_dat__data(filename):
    data = np.loadtxt(filename, dtype=np.float64)
    return data

def read_xop(filename):
    """
    Read a data file from XOP and return a standard rocking-curve dict.

    It is expected that the given file is diff_pat.dat file from XOP.
    """
    pathname, basename = os.path.split(filename)
    rootname, extname = os.path.splitext(basename)
    if not extname == '.dat':
        m_log.warning(f'File extion .dat expected for XOP data. Found {extname} instead.')

    m_log.info(f'Reading XOP data from{filename}')

    header = _read_xop__diff_pat_dat__header(filename)
    data = _read_xop__diff_pat_dat__data(filename)

    col_list = [
        ('dtheta_in',   'urad', 'Th-Th Bragg (in)')
        ,('dtheta_out', 'urad', 'Th-Th Bragg (out)')
        ,('phase_p',    'rad',  'Phase pi')
        ,('phase_s',    'rad',  'Phase sigma')
        ,('circular',   '',     'Circ. Polariz.')
        ,('reflect_p',  '',     'Reflectivity pi')
        ,('reflect_s',  '',     'Reflectivity sigma')
    ]

    out = {}
    out['value'] = {}
    out['units'] = {}
    out['label'] = {}

    for ii, info in enumerate(col_list):
        name, units, label = info
        out['value'][name] = data[:,ii]
        out['units'][name] = units
        out['label'][name] = label

    return out