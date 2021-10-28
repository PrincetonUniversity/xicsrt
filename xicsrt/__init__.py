# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
"""

# Import top level user functions.
from xicsrt.xicsrt_raytrace import raytrace
from xicsrt._version import __version__, warn_version

try:
    from xicsrt.xicsrt_multiprocessing import raytrace as raytrace_mp
except:
    raise Warning('multiprocessing could not be loaded on your platform.')
