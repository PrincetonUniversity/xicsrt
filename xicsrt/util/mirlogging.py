# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir pablant <npablant@pppl.gov>

A logging module for XICSRT.

For now, the purpose of this module is simply is set default logging options for
interactive use.
"""

from logging import *

def defaultConfig(level=None, long=False, force=False):
    fmt = '{levelname:>5.5s}:{name:>6.6s}: {message}'
    if long:
        fmt = fmt.replace('6.6s', '12.12s')
    if level is None:
        level = DEBUG
    basicConfig(format=fmt, style='{', force=force)
    getLogger('xicsrt').setLevel(level)

# Setup defaultConfig() when this module is imported, but do not force.
defaultConfig()