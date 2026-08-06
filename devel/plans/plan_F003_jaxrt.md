# Plan: JAX-Accelerated Engine for XICSRT (F003)

Status: Approved 2026-07-17. Not yet started.
Tracking: devel/features_request.md (F003).

This document is self-contained: it includes the approved design, all user
decisions, and the codebase research needed to implement in a fresh session
without re-exploration.

## Goals and priorities (in order)

1. **Readability first** — code reads like the physics; understandable by
   doctorate-level physicists without JAX/CS expertise.
2. **Exact photon statistics at every element** — the core tenet of XICSRT,
   preserved without exception (see mandate in AGENTS.md).
3. JAX jit/vmap acceleration: float64, single device (CPU on dev laptop,
   one GPU on the Princeton Stellar cluster); multi-GPU deferred.
4. Same JSON config dicts as the numpy engine; numpy engine completely
   untouched. `jax` is an optional dependency (`pip install xicsrt[jax]`).

## User decisions (approved)

- Parallel subpackage `xicsrt/jaxrt/` (not a git branch, not a fork).
  Named `jaxrt`, not `jax`, to avoid shadowing `import jax`.
- Architecture: pure functions + config dicts (no mutable mixin classes).
- float64 everywhere: `jax_enable_x64 = True` set on `xicsrt.jaxrt` import.
- RNG: explicit `jax.random` key threading. Statistically equivalent to the
  numpy engine, NOT bit-identical, even with the same seed. Approved.
- Poisson ray counts (`use_poisson`): **capacity + mask**. Draw the true
  Poisson count `N ~ Poisson(intensity)`; arrays have fixed capacity
  = mean + 10 sigma; rays beyond N are masked from birth. Every element sees
  exactly N statistically correct photons. Overflow probability ~1e-23;
  raise an error rather than silently truncate. Exact statistics preserved.
- Parallelism phase 1: sequential runs, single process, single device.
  Rationale: jit+vmap already saturates one device; multiprocessing + JAX is
  counterproductive (recompilation per fork, GPU memory contention).
  Cross-node scaling on Stellar via SLURM job arrays with differing seeds
  plus existing hdf5 combining — no code needed. Per-run seed offsets are
  preserved (key splitting) so separate jobs combine exactly as today.
- Multi-GPU (phase 2+): shard runs across devices via pmap/sharding; touches
  only `_engine.py`, no physics code.
- Config compatibility: same config dict/JSON works on both engines;
  unsupported options raise clear errors (reuse strict_config_check).

## Subpackage structure

```
xicsrt/jaxrt/
    __init__.py          # exports raytrace(); enables jax x64 mode on import
    _engine.py           # raytrace(config): sequential run/iter loops, RNG key
                         #   threading, result combining (same output dict)
    _rays.py             # rays pytree: plain dict of jnp arrays (origin,
                         #   direction, wavelength, weight, mask) + helpers
    _dispatch.py         # class_name -> (setup, generate/trace) registry;
                         #   reuses xicsrt_config defaults + strict checking
    _images.py           # detector image accumulation via scatter-add
    sources/_generic.py  # Generic/Directed/Focused sources as pure functions
    shapes/              # _plane.py, _sphere.py, _cylinder.py, _torus.py
    interact/            # _mirror.py, _crystal.py, _mosaic.py
    tools/               # jnp spread/voigt sampling (reuse tools/
                         #   xicsrt_math_jax.py and the jax-ready Weideman
                         #   faddeeva kernel in tools/xicsrt_faddeeva.py)
```

- Each element module: a non-jit `setup(config)` (precompute geometry, load
  rocking-curve files once) + a pure jit-able `trace(rays, params, key)`.
  Optics composed as `intersect` + `interact` function pairs, mirroring the
  numpy Shape*/Interact* pairing as function composition in `_dispatch.py`.
- Entry point `xicsrt.jaxrt.raytrace(config)` returns the same
  hdf5-compatible results dict (meta/image/history as numpy arrays out).
- jit boundary: one jit'd function per iteration (source generate + full
  optic chain). Geometry enters as static/closure data — compiles once per
  scenario, reused across all runs/iterations.

## Key translation decisions

| Numpy pattern | JAX version |
|---|---|
| `m[m] &= cond[m]`, `arr[m] = ...` | `jnp.where(mask, ...)` via small shared helpers so each optic stays clean |
| Global `np.random` | Explicit `jax.random` keys, split per run/iter/element |
| Poisson ray count (dynamic N) | Capacity + mask (see decisions above) |
| Mosaic depth loop with `break` | Fixed `mosaic_depth` passes via `lax.fori_loop`; identical statistics, no early exit |
| Rejection sampler (`vector_dist_isotropic_xy`) | Direct inverse-CDF sampling where derivable (also more readable); else `lax.while_loop` |
| Per-ray image loop | `.at[].add()` scatter-add |
| `keep_history` deepcopies | Per-element ray snapshots returned from jit'd trace as stacked arrays |

## Phase 1 element scope

- Sources: Generic, Directed, Focused.
- Shapes: Plane, Sphere, Cylinder, Torus.
- Interactions: None (Detector/Aperture), Mirror, Crystal, MosaicCrystal.
- Rocking curves: step, gaussian, file (load file once in setup, `jnp.interp`
  in trace — numpy version re-reads every iteration, do not copy that).

Deferred phases: plasma sources (bundle loop, dynamic concatenation); mesh
optics (scipy cKDTree/Delaunay — needs JAX-compatible neighbor search);
filters; multi-GPU run-sharding; any `--mp` interplay.

## Validation and benchmarking

- `tests/jaxrt/` pytest suite:
  - Geometry unit tests vs numpy shape/interact functions on identical
    inputs (float64, near bit-level for deterministic parts).
  - Statistical tests: detector images and found-ray counts agree within
    Poisson error bars vs numpy engine (example_00-style scenarios).
  - Analytic checks (Bragg angle, focal geometry).
- Benchmark script comparing numpy vs jaxrt engines vs ray count
  (CPU now, GPU-ready).

## Housekeeping requirements

- AGENTS.md photon-statistics mandate (added 2026-07-17 — verify present).
- AI disclaimer headers on all new/modified files (per AGENTS.md).
- setup.py: `extras_require={'jax': ['jax']}` (keep existing 'test' extra).
- doc_source page + module docstrings explaining the pure-function
  architecture and key threading for future physicist contributors.

## Implementation order

1. Housekeeping (done: F003 entry, AGENTS.md directive, this plan file).
2. `_rays.py` + RNG helpers + Plane shape + None interact + Detector ->
   straight-line rays end to end through `_engine.py`.
3. Generic/Directed/Focused sources (Poisson capacity+mask, spread sampling).
4. Sphere/Cylinder/Torus; Mirror + Crystal interactions; image scatter-add.
5. MosaicCrystal; engine polish (history, meta, combine).
6. Tests + benchmark + docs.

## Codebase research notes (from 2026-07-17 exploration)

Reference points in the numpy engine relevant to the port:

- Core loop: `xicsrt/xicsrt_raytrace.py` — `raytrace` (runs loop, L55; seed
  offset per run L61-62), `raytrace_single` (global `np.random.seed` L111;
  iter loop L153-158), `_raytrace_iter` (hot path = `generate_rays()` +
  `optics.trace()`, L192-194), `_sort_raytrace` (found/lost split on last
  optic mask, L258-266; lost subsample via shuffle), `combine_raytrace`
  (sums meta/images, concatenates histories).
- Optic sequencing: `objects/_Dispatcher.py` `trace()` L166-196 — plain for
  loop over optics; `deepcopy(rays)` per optic when keep_history (L187).
  Elements found by file-glob `_Xicsrt*.py` (L63-113); jaxrt uses its own
  registry in `_dispatch.py` instead.
- RayArray: `objects/_RayArray.py` — dict subclass; origin/direction (N,3)
  f64, wavelength/weight (N,) f64, mask (N,) bool. Mutated in place by
  optics. Rays are never compacted during tracing (fixed N — good for jit).
- Per-optic pipeline: `optics/_TraceObject.py` `trace` L157-172:
  `intersect -> check_bounds (size L208-212 + aperture L230) -> interact`.
- Masking idioms to translate: `m[m] &= cond[m]` (`_ShapeSphere.py:85`,
  `_ShapeTorus.py:181`, `_InteractCrystal.py:128`); masked writes
  `distance[m] = ...` (`_ShapeSphere.py:82-98`, `_ShapePlane.py:43-48`);
  in-place reflection `D[m] -= 2*(D.n)n` (`_InteractMirror.py:38-39`).
- Shapes are all closed-form: plane (dot/div), sphere/cylinder (quadratic),
  torus (analytic quartic via `tools/xicsrt_quartic.py` `multi_quartic`,
  vectorized complex arithmetic — port to jnp).
- Crystal: `_InteractCrystal.py` — masked arcsin/arccos angle calc
  (L96-114); `rocking_curve_filter` L136-196: probability curve
  (step/gaussian/file via `np.interp`) then stochastic acceptance
  `p >= uniform(0,1,N)` (L189-194). File re-read every iter at L153 (fix in
  jaxrt setup).
- Mosaic: `_InteractMosaicCrystal.py:53-107` — loop to `mosaic_depth`
  (default 15) with early break; Gaussian crystallite normals via
  `xicsrt_spread.vector_dist_flat_gaussian`.
- Sources: `sources/_XicsrtSourceGeneric.py` — fields built independently
  (origin, direction, wavelength, weight, mask); Poisson count in
  `initialize` L218; wavelength samplers uniform/normal/voigt (voigt uses
  inverse-CDF + `np.interp`, `tools/xicsrt_voigt.py:166-177` — fully
  vectorizable); directions via `tools/xicsrt_spread.py` (rejection while
  loop at L198 in `vector_dist_isotropic_xy`).
- Image accumulation: `_TraceObject.py` `make_image` L234-293 — per-ray
  Python loop at L289-291, replace with scatter-add.
- Existing JAX footholds: `tools/xicsrt_math_jax.py` (jnp copies of
  xicsrt_math), `tools/xicsrt_faddeeva.py` (jax-ready Weideman Faddeeva
  kernel, see F001), `_ShapeMeshTorus.py:180` shape_jax hook.
- scipy in hot path: only voigt (`wofz`, replaceable by faddeeva kernel) and
  mesh optics (deferred). PIL only in io (not hot path).
- Profiler: `xicsrt/util/profiler.py` manual start/stop, off by default.

## Verification commands

    pip install pytest
    pytest tests/
    python examples/example_00/example_00.py
