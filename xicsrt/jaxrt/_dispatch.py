# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Element dispatch for the jaxrt engine.

The jaxrt engine reuses the numpy element classes for config parsing,
validation (including strict_config_check) and static geometry setup.
This guarantees that a given config dict is interpreted identically by
both engines. After setup, each element's `param` dict is handed to
the pure-function jax modules that implement the actual physics.

The mapping from a numpy element class to jax functions is based on
the class's inheritance (MRO): the Shape* mixin selects the intersect
function and the Interact* mixin selects the interact function. This
means user-defined optics that only combine supported built-in mixins
work automatically.

This module was AI generated using Claude (Fable 5).
"""

from xicsrt.objects._Dispatcher import Dispatcher

from xicsrt.jaxrt import _bounds
from xicsrt.jaxrt import _images
from xicsrt.jaxrt.interact import _crystal
from xicsrt.jaxrt.interact import _mirror
from xicsrt.jaxrt.interact import _mosaic
from xicsrt.jaxrt.interact import _none
from xicsrt.jaxrt.shapes import _cylinder
from xicsrt.jaxrt.shapes import _plane
from xicsrt.jaxrt.shapes import _sphere
from xicsrt.jaxrt.shapes import _torus
from xicsrt.jaxrt.sources import _generic

# Numpy mixin class name -> jax shape module.
_SHAPE_MODULES = {
    'ShapePlane': _plane,
    'ShapeSphere': _sphere,
    'ShapeCylinder': _cylinder,
    'ShapeTorus': _torus,
}

# Numpy mixin class name -> jax interaction module.
# Order matters: the most derived interaction must be checked first,
# since e.g. InteractMosaicCrystal inherits from InteractCrystal.
_INTERACT_MODULES = [
    ('InteractMosaicCrystal', _mosaic),
    ('InteractCrystal', _crystal),
    ('InteractMirror', _mirror),
    ('InteractNone', _none),
]

# Numpy source class name -> emission cone aiming mode.
_SOURCE_AIM = {
    'XicsrtSourceGeneric': 'zaxis',
    'XicsrtSourceDirected': 'direction',
    'XicsrtSourceFocused': 'target',
}


def setup_sources(config):
    """
    Instantiate and initialize the sources through the numpy engine
    machinery, then build jax source parameter dicts.

    Returns a dict of {name: source_params} and the updated config.
    """
    dispatcher = Dispatcher(config, 'sources')
    dispatcher.instantiate()
    dispatcher.setup()
    dispatcher.check_param()
    dispatcher.initialize()
    config['sources'] = dispatcher.get_config()

    sources = {}
    for name, obj in dispatcher.objects.items():
        aim = _find_source_aim(obj)
        sources[name] = _generic.setup(obj.param, aim)
    return sources


def setup_optics(config):
    """
    Instantiate and initialize the optics through the numpy engine
    machinery, then build jax optic parameter dicts.

    Returns a dict of {name: optic_params} and the updated config.
    Each optic_params contains the shape/interact modules and their
    static parameter pytrees.
    """
    dispatcher = Dispatcher(config, 'optics')
    dispatcher.instantiate()
    dispatcher.setup()
    dispatcher.check_param()
    dispatcher.initialize()
    config['optics'] = dispatcher.get_config()

    optics = {}
    for name, obj in dispatcher.objects.items():
        shape_module = _find_shape_module(obj)
        interact_module = _find_interact_module(obj)

        if obj.param['trace_local']:
            raise NotImplementedError(
                "The 'trace_local' option is not supported by the jaxrt "
                f"engine (optic '{name}').")
        if obj.param['filters']:
            raise NotImplementedError(
                f"Filters are not supported by the jaxrt engine (optic '{name}').")

        # The numpy optic objects keep the orientation matrix as an
        # attribute rather than in param; copy it in for the jax setup.
        param = dict(obj.param)
        param['orientation'] = obj.orientation

        optics[name] = {
            'shape': shape_module,
            'interact': interact_module,
            'geom': shape_module.setup(param),
            'phys': interact_module.setup(param),
            'bounds': _bounds.setup(param),
            'image': _images.setup(param),
        }
    return optics


def _find_source_aim(obj):
    """
    Find the aiming mode for a source object from its class hierarchy.
    """
    for cls in type(obj).__mro__:
        if cls.__name__ in _SOURCE_AIM:
            return _SOURCE_AIM[cls.__name__]
    raise NotImplementedError(
        f"Source class '{type(obj).__name__}' is not supported by the "
        "jaxrt engine.")


def _find_shape_module(obj):
    """
    Find the jax shape module for an optic from its class hierarchy.
    """
    for cls in type(obj).__mro__:
        if cls.__name__ in _SHAPE_MODULES:
            return _SHAPE_MODULES[cls.__name__]
    raise NotImplementedError(
        f"No jaxrt shape available for optic class '{type(obj).__name__}'. "
        f"Supported shapes: {sorted(_SHAPE_MODULES)}.")


def _find_interact_module(obj):
    """
    Find the jax interaction module for an optic from its class
    hierarchy. The most derived supported interaction wins.
    """
    class_names = [cls.__name__ for cls in type(obj).__mro__]
    for name, module in _INTERACT_MODULES:
        if name in class_names:
            return module
    raise NotImplementedError(
        f"No jaxrt interaction available for optic class "
        f"'{type(obj).__name__}'. Supported interactions: "
        f"{[name for name, _ in _INTERACT_MODULES]}.")
