# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
Pure-function shape (intersection) modules for the jaxrt engine.

Each shape module provides:

setup(param) -> geom
    Host-side precomputation of static geometry from the `param` dict
    of the initialized numpy optic object.

intersect(rays, geom) -> (xloc, norm, mask)
    Pure jit-able intersection: returns intersection locations,
    surface normals and the updated ray mask.
"""
