
JAX Acceleration (xicsrt.jaxrt)
===============================

XICSRT includes an optional JAX-accelerated raytracing engine in the
:mod:`xicsrt.jaxrt` subpackage. The jaxrt engine uses
`JAX <https://jax.readthedocs.io>`_ jit compilation and vectorization
to accelerate raytracing on CPUs and GPUs, while preserving the core
XICSRT tenet that photon statistics are exactly correct at every
element.

To use the jaxrt engine, install the optional jax dependency and
replace the call :any:`xicsrt.raytrace()` with
:code:`xicsrt.jaxrt.raytrace()`:

.. code:: bash

    pip install xicsrt[jax]

.. code:: python

    import xicsrt.jaxrt
    results = xicsrt.jaxrt.raytrace(config)

The jaxrt engine accepts the same configuration dictionary as the
numpy engine and returns the same results dictionary structure.
Random number generation uses explicit `jax.random` key threading;
results are statistically equivalent to (but not bit-identical with)
the numpy engine, even when using the same `random_seed`.

Supported elements
------------------

The jaxrt engine currently supports a subset of the built-in elements:

- Sources: Generic, Directed, Focused.
- Shapes: Plane, Sphere, Cylinder, Torus.
- Interactions: None (Detector/Aperture), Mirror, Crystal, MosaicCrystal.
- Rocking curves: step, gaussian, file.

Plasma sources, mesh optics, filters and multiprocessing are not
currently supported; a clear error is raised if the configuration
requests an unsupported feature.

Architecture
------------

Unlike the numpy engine, which composes optics from mutable mixin
classes, the jaxrt engine is built from pure functions: each shape
provides an ``intersect`` function and each interaction provides an
``interact`` function, both operating on an immutable ray bundle (a
dict of jax arrays). Element configuration and static geometry setup
reuse the numpy element classes, so config validation and geometry
interpretation are guaranteed identical between the engines.

One function containing the ray generation and the full optic chain is
jit-compiled per scenario and reused for every iteration and run.

Poisson statistics
------------------

When ``use_poisson`` is enabled, the true ray count for each iteration
is drawn from a Poisson distribution exactly as in the numpy engine.
Because jit compilation requires fixed array shapes, the ray arrays are
allocated with a capacity of mean + 10 sigma and rays beyond the drawn
count are masked from birth. Every element sees exactly the drawn
number of statistically correct photons; if the draw ever exceeds the
capacity (probability ~1e-23) an error is raised rather than silently
truncating.

Cluster computing
-----------------

The jaxrt engine performs runs sequentially in a single process on a
single device; jit compilation and vectorization typically saturate one
CPU or GPU. For multi-node scaling use SLURM job arrays with different
random seeds and combine the saved results, exactly as with the numpy
engine (see :any:`multiple_processors`).
