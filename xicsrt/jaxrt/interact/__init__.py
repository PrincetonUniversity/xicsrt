# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
Pure-function interaction modules for the jaxrt engine.

Each interaction module provides:

setup(param) -> phys
    Host-side precomputation of static physics parameters (e.g.
    rocking curve tables) from the `param` dict of the initialized
    numpy optic object.

interact(rays, xloc, norm, mask, phys, key) -> rays
    Pure jit-able interaction: returns the updated ray bundle.
    `key` is a jax random key (unused by deterministic interactions).
"""
