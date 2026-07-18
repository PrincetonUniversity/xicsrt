# jaxrt Synchronization Guide

This file includes AI generated content using Claude (Fable 5).

Purpose: `xicsrt/jaxrt/` is a parallel JAX-based raytracing engine that mirrors
the physics of the numpy engine. This document tells a future AI coding session
(or human) exactly how to keep the two engines aligned when the numpy engine
changes, without re-deriving the architecture.

Background: full approved design in `devel/plan_jaxrt.md` (F003 in
`devel/features_request.md`). Read that for rationale; read *this* file for the
mechanical sync procedure.

## When to consult this file (trigger conditions)

Any change to the following requires checking whether jaxrt must be updated:

- `xicsrt/sources/` — any source class
- `xicsrt/optics/` — any Shape*, Interact*, TraceObject, or XicsrtOptic* class
- `xicsrt/tools/xicsrt_spread.py`, `xicsrt_voigt.py`, `xicsrt_voigt_multi.py`,
  `xicsrt_aperture.py`, `xicsrt_bragg.py`, `xicsrt_quartic.py`
- `xicsrt/xicsrt_raytrace.py`, `xicsrt/xicsrt_config.py` (config defaults,
  results dict structure)
- `xicsrt/objects/` — Dispatcher, ConfigObject, GeometryObject, RayArray

Changes elsewhere (io, visual, util, filters, plasma sources, mesh optics) do
not currently affect jaxrt.

## File correspondence map

| numpy engine | jaxrt mirror |
|---|---|
| `optics/_ShapePlane.py` | `jaxrt/shapes/_plane.py` |
| `optics/_ShapeSphere.py` | `jaxrt/shapes/_sphere.py` |
| `optics/_ShapeCylinder.py` | `jaxrt/shapes/_cylinder.py` |
| `optics/_ShapeTorus.py` | `jaxrt/shapes/_torus.py` |
| `optics/_InteractNone.py` (+ `_InteractObject.py`) | `jaxrt/interact/_none.py` |
| `optics/_InteractMirror.py` | `jaxrt/interact/_mirror.py` |
| `optics/_InteractCrystal.py` | `jaxrt/interact/_crystal.py` |
| `optics/_InteractMosaicCrystal.py` | `jaxrt/interact/_mosaic.py` |
| `sources/_XicsrtSourceGeneric/Directed/Focused.py` | `jaxrt/sources/_generic.py` (one module; the three classes differ only by the `aim` mode: 'zaxis' / 'direction' / 'target') |
| `optics/_TraceObject.py` `check_bounds`/`check_size`/`check_aperture` | `jaxrt/_bounds.py` |
| `optics/_TraceObject.py` `make_image` | `jaxrt/_images.py` |
| `tools/xicsrt_spread.py` samplers | `jaxrt/tools/_spread.py` |
| `tools/xicsrt_quartic.py` `multi_quartic` | `jaxrt/tools/_quartic.py` |
| source wavelength sampling (`_XicsrtSourceGeneric` + voigt tools) | `jaxrt/tools/_wavelength.py` |
| `xicsrt_raytrace.py` orchestration | `jaxrt/_engine.py` |
| `objects/_RayArray.py` | `jaxrt/_rays.py` (plain dict pytree, adds 'weight') |
| element discovery / mixin composition | `jaxrt/_dispatch.py` (explicit registries, MRO-based mapping) |

Each jaxrt physics module has the same two-function shape:
`setup(param) -> static pytree` (host-side, run once) and a pure jit-able
`intersect(rays, geom)` / `interact(rays, xloc, norm, mask, phys, key)` /
`generate(source, key)`.

## Live coupling points (numpy code jaxrt reuses at runtime)

jaxrt *imports and executes* the following numpy-engine code. Changes here
propagate to jaxrt automatically — which is convenient, but can also silently
break jaxrt. Check these call sites when refactoring:

- `objects/_Dispatcher.py` `Dispatcher` — used by `jaxrt/_dispatch.py` to
  instantiate/setup/initialize elements for config validation and geometry.
- `xicsrt_raytrace.py` `_sort_raytrace`, `combine_raytrace`, `check_config`,
  `print_raytrace` — used directly by `jaxrt/_engine.py`. If the results dict
  structure changes, jaxrt output changes with it (good), but `_engine._to_host`
  must still produce the matching per-iteration input structure.
- `tools/xicsrt_spread.py` `_parse_spread_single`, `_parse_spread_xy` — used in
  `jaxrt/tools/_spread.setup`.
- `tools/xicsrt_voigt.py` `voigt_cdf_tab` and `tools/xicsrt_voigt_multi.py`
  `multi_voigt_cdf_tab` — used in `jaxrt/tools/_wavelength.setup` to build CDF
  tables on the host.
- `tools/xicsrt_aperture.py` `_aperture_defaults` — used in `jaxrt/_bounds.py`.
- `tools/xicsrt_bragg.py` `read` — used in `jaxrt/interact/_crystal.setup`.
- `_dispatch.py` reads `obj.param` and `obj.orientation` from the initialized
  numpy element objects. Renaming param keys (e.g. 'center', 'torus_major',
  'root_idx', 'pixel_xsize') breaks the jaxrt `setup()` functions.

KEY RULE: because jaxrt reuses the numpy classes for config handling, a new
config option is *validated* automatically on both engines — but its *physics*
is NOT. When adding an option to a supported element, either port the behavior
into the jaxrt module or raise a clear `NotImplementedError` in the
corresponding jaxrt `setup()` when the option is set to a non-default value.
Never silently ignore an option.

## numpy -> JAX translation idioms

| numpy pattern | jaxrt pattern |
|---|---|
| `m[m] &= cond[m]` | `mask = mask & cond` |
| `arr[m] = value` | `arr = jnp.where(mask, value, arr)` (use `_rays.where_scalar` / `where_vector`) |
| `xloc[~m] = np.nan` (implicit via `np.full(nan)`) | `xloc = _rays.where_vector(mask, xloc, jnp.nan)` |
| global `np.random.*` | explicit `jax.random` keys; split per run/iter/element (`jax.random.split`) |
| loop with `break` (e.g. mosaic depth) | `jax.lax.fori_loop` with fixed trip count; exclude finished rays by mask |
| rejection `while` loop | `jax.lax.while_loop` (see `_spread._sample_isotropic_xy`) |
| per-ray Python loop for images | scatter-add `image.at[cx, cy].add(1.0, mode='drop')` |
| `np.einsum('ij,ij->i', a, b)` | `_rays.dot(a, b)` |
| branch on data (`if np.sum(m) > 0`) | not allowed under jit; compute unconditionally, combine with `jnp.where` |
| `np.sqrt` of possibly-negative masked values | clamp first: `jnp.sqrt(jnp.maximum(x, 0.0))` (unselected lanes must not produce nan) |

## Invariants — never violate

1. **Exact photon statistics** at every element (AGENTS.md core mandate). No
   approximation that alters statistics, ever, without explicit user approval.
2. **Poisson via capacity + mask**: N ~ Poisson(intensity) drawn fresh each
   iteration; arrays sized mean + 10 sigma; rays beyond N masked from birth;
   overflow raises `RuntimeError` (never truncate silently).
3. **Quartic root ordering**: `jaxrt/tools/_quartic.py` must return roots in
   the same order as `xicsrt_quartic.multi_quartic`; the torus selects its
   intersection by `root_idx`. If the numpy solver is ever replaced, both must
   be revalidated together.
4. **Rocking curve files load once in setup** (`jnp.interp` in trace). This is
   an intentional deviation — the numpy engine re-reads the file every
   iteration. Do not "fix" jaxrt back to match.
5. **Same config dict works on both engines**; unsupported options must raise
   clear errors, not warn-and-ignore.
6. **float64 everywhere** (`jax_enable_x64` set in `jaxrt/__init__.py`).
7. Fixed array shapes inside jit: rays are masked, never compacted.

## Adding a new element to jaxrt

1. Write the pure-function module in `jaxrt/shapes/`, `jaxrt/interact/` or
   `jaxrt/sources/` following the `setup` + trace-function pattern.
2. Register it in `jaxrt/_dispatch.py`:
   - shapes: `_SHAPE_MODULES` (keyed by numpy mixin class name)
   - interactions: `_INTERACT_MODULES` — this is an *ordered list*; the most
     derived class must come first (e.g. `InteractMosaicCrystal` before
     `InteractCrystal`), since matching walks the numpy class MRO.
   - sources: `_SOURCE_AIM` (class name -> aim mode) if it fits the generic
     source; otherwise extend `setup_sources`.
3. User optics composed only of supported mixins work automatically — the MRO
   lookup finds the built-in base classes.
4. Add tests (below).

## Verification procedure (run after any sync change)

    pytest tests/jaxrt/          # jaxrt parity + statistics tests
    pytest tests/                # full suite
    python examples/example_00/example_00.py

- Deterministic changes (geometry, transforms): add/extend a parity test in
  `tests/jaxrt/test_geometry.py` comparing against the numpy class on identical
  rays (atol 1e-9, float64).
- Stochastic changes (samplers, rocking curves, Poisson): add/extend a
  statistical comparison in `tests/jaxrt/test_engine.py` — detected counts
  within 5 sigma of the numpy engine, image centroids within ~2 pixels. If a
  single-seed result is marginal (>1.5 sigma), rerun across ~5 seeds and check
  the mean bias is near zero before concluding anything.
- Hot-path changes: run `testing/benchmark_jaxrt.py` and compare against the
  values in the divergence log below (or the F003 notes).

## Unsupported features (raise NotImplementedError; deferred by design)

- Plasma sources (bundle loop, dynamic concatenation)
- Mesh optics (scipy cKDTree/Delaunay)
- Filters (config section and per-element `filters` option)
- `trace_local` optic option
- Multiple sources per scenario (also unsupported in numpy engine)
- Multiprocessing / `--mp` (intentionally excluded; use SLURM job arrays)
- Multi-GPU sharding (phase 2+; would touch only `_engine.py`)

Known benign quirks:

- `combine_raytrace` drops the 'weight' ray field from combined histories
  (`RayArray.zeros` only defines origin/direction/mask/wavelength). Affects
  both engines identically.
- jaxrt is statistically equivalent but not bit-identical to numpy for the
  same `random_seed` (different RNG streams; approved design decision).

## Divergence log

Record here any numpy-engine change that was *intentionally not yet ported* to
jaxrt (date, numpy file/feature, reason, and what a port would require). Future
sessions must check this list before assuming the engines are in sync.

- (none as of 2026-07-17 — engines in sync for the phase 1 element scope)

## How to use this file (for the human)

When you change the numpy engine and want jaxrt updated, start an AI session
with a prompt like:

    I changed <numpy file(s)>: <one-line summary>. Read devel/jaxrt_sync.md
    and update the jaxrt subpackage to match, then run the verification
    procedure described there.

If you want to defer the port, instead ask the agent to add an entry to the
Divergence log above.
