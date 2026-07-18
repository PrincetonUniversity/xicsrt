# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

No interaction with the surface (jaxrt).

Port of :class:`xicsrt.optics._InteractNone.InteractNone`. Rays pass
through unchanged: the ray origin is moved to the intersection
location and the mask is updated. Used for detectors and apertures.

This module was AI generated using Claude (Fable 5).
"""


def setup(param):
    """
    No static parameters are needed for this interaction.
    """
    return {}


def interact(rays, xloc, norm, mask, phys, key):
    """
    Move rays to their intersection location without changing direction.

    Matching the numpy engine, the origin is set to the intersection
    location for all rays (lost rays end up with whatever value the
    intersection produced; they remain masked out).
    """
    rays = dict(rays)
    rays['origin'] = xloc
    rays['mask'] = mask
    return rays
