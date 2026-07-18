# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A JAX-accelerated raytracing engine for XICSRT.

This subpackage provides :func:`xicsrt.jaxrt.raytrace`, a drop-in
alternative to :func:`xicsrt.raytrace` that runs the raytracing
calculation through `JAX <https://jax.readthedocs.io>`_ for jit/vmap
acceleration on CPUs and GPUs.

Design principles (in priority order):

1. Readability first: the code should read like the physics.
2. Exact photon statistics at every element (the core XICSRT tenet).
3. Acceleration through jit compilation and vectorization.

The engine accepts the same configuration dictionaries as the numpy
engine and produces the same results dictionary structure. Random
number generation uses explicit `jax.random` key threading, which is
statistically equivalent to (but not bit-identical with) the numpy
engine.

Architecture notes
------------------
Unlike the numpy engine, which uses mutable mixin classes, the jaxrt
engine is built from pure functions. Element configuration and static
geometry setup reuse the numpy element classes (so config validation
and geometry interpretation are guaranteed identical), after which
the physics is dispatched to pure jax functions in the `shapes`,
`interact` and `sources` submodules.

`jax` is an optional dependency; install with ``pip install xicsrt[jax]``.
"""

import jax as _jax

# XICSRT requires float64 precision throughout.
# This must be set before any jax arrays are created.
_jax.config.update('jax_enable_x64', True)

from xicsrt.jaxrt._engine import raytrace
