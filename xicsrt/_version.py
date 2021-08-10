"""
Version Description:
major.minor.revision

major:
  - Major changes in codebase or functionality.

minor:
  - Changes to the public API
  - Old config files can no longer be loaded without conversion.
  - New output dictionaries may break existing external code.

revision:
  - New release that does not break any compatibility.
"""

__version__="0.8.0"

from packaging import version
from xicsrt.util import mirlogging
log = mirlogging.getLogger('xicsrt')

def warn_version(v_string):
    v_input = version.parse(v_string)
    v_current = version.parse(__version__)

    if (v_input.major < v_current.major) or (v_input.minor < v_current.minor):
        log.warning('This config is for an older version of xicsrt. Some options may have changed.')
    elif (v_input.major > v_current.major) or (v_input.minor > v_current.minor):
        log.warning('This config is for a newer version of xicsrt. Please upgrade xicsrt (pip install --upgrade xicsrt).')
