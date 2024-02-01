# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir pablant <npablant@pppl.gov>

Some utilities to help with version inspection.
"""

from xicsrt import _version

from packaging import version
from xicsrt.util import mirlogging
log = mirlogging.getLogger('xicsrt')

def warn_version(v_string):
    v_input = version.parse(v_string)
    v_current = version.parse(_version.__version__)

    if (v_input.major < v_current.major) or (v_input.minor < v_current.minor):
        log.warning('This config is for an older version of xicsrt. Some options may have changed.')
    elif (v_input.major > v_current.major) or (v_input.minor > v_current.minor):
        log.warning('This config is for a newer version of xicsrt. Please upgrade xicsrt (pip install --upgrade xicsrt).')
