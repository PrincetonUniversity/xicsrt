# F034 - Optional per-ray integer `label` field in the ray-data pipeline

Status: Pending

This file includes AI generated content using Claude (Sonnet 5).

## Goal

Add a general-purpose, lightweight `label` field (dtype `int`) to the ray
data dict, alongside the existing `origin`/`direction`/`wavelength`/`mask`
fields, for future per-ray metadata use. The first concrete consumer is
F035 (per-emission-line labeling for the W7-X XICS model), but this
feature adds no labeling scheme of its own.

## Design

- `label` is **opt-in and absent by default**, exactly like `weight`
  behaves today: a source sets `rays['label']` only if it has a reason to;
  no source is required to set it, and no downstream code requires it to
  be present.
- When present, `label` is a plain `int`-dtype `numpy.ndarray` of shape
  `(n_rays,)`, one integer per ray. No sentinel/reserved value is defined
  by this feature; whatever an individual source uses as its "default"
  value (e.g. 0) carries no meaning to the core pipeline.
- Verified during investigation (see PR/commit discussion) that the ray
  pipeline is already fully key-generic outside of `RayArray.zeros()`/
  `initialize()`:
  - `xicsrt/objects/_Dispatcher.py` (`trace`) only reads `rays['mask']`
    and deep-copies the whole dict for history.
  - `xicsrt/optics/_TraceObject.py` and all `Shape*`/`Interact*` classes
    only ever access `origin`/`direction`/`mask`/`wavelength` by name.
  - `xicsrt/xicsrt_raytrace.py`'s `_sort_raytrace` and `combine_raytrace`
    already iterate ray-dict keys generically (a deliberate fix,
    documented inline, to stop `weight` from being silently dropped).
  - `xicsrt/util/mirhdf5.py` (used by `xicsrt_io.py` for HDF5 save/load)
    branches only on Python type, never on key name.
  - `xicsrt/sources/_XicsrtPlasmaBundleSource.py::create_sources`
    concatenates per-bundle-source rays generically by key (documented
    inline for the same `weight`-preservation reason).

  So a new `label` key requires **no changes** in any of the above; it
  will flow through the full trace, sort, combine, and save/load pipeline
  unchanged, for both `found` and `lost` rays.

- `RayArray.zeros()` (`xicsrt/objects/_RayArray.py`) is a documentation/
  convenience method not called anywhere on the hot path (verified via
  repo-wide grep; only ever exercised directly in ad hoc tests/notebooks).
  It currently hardcodes only `origin`/`direction`/`mask`/`wavelength`,
  which is misleading since `weight` is also a standard ray field in
  practice (set by both `XicsrtSourceGeneric.generate_rays` and
  `XicsrtPlasmaGeneric.create_sources`). Add `weight` (`np.ones`, float64)
  and `label` (`np.zeros`, int) to `zeros()` so it remains an accurate,
  embedded-documentation reference of every standard ray field, its dtype,
  and its default value. `initialize()` is NOT changed: it stays
  responsible only for the two fields treated as structurally required
  (`mask`, `wavelength`), and must not auto-populate `label` (auto-filling
  there would make `label` appear on every `RayArray()` construction,
  defeating "absent by default").

- jaxrt (`xicsrt/jaxrt/_rays.py::new_rays`) uses a fixed-shape dict pytree
  (required for `jit`), so it cannot support a truly optional field the
  way the numpy engine does. Per user direction, add `label` to
  `new_rays()` unconditionally (`jnp.zeros(num, dtype=jnp.int64)`), always
  present but functionally inert until a jaxrt source or interact routine
  reads/writes it. This keeps the two engines' standard ray-field
  documentation in sync even though the numpy engine's field is opt-in.
  jaxrt has no plasma sources (see `devel/jaxrt_sync.md`), so nothing in
  jaxrt sets `label` to anything but zero as part of this feature.

## Scope

- `xicsrt/objects/_RayArray.py`:
  - Add AI-tool file header disclaimer (file is being modified).
  - `zeros(self, num)`: add `self['weight'] = np.ones(num, dtype=np.float64)`
    and `self['label'] = np.zeros(num, dtype=int)`.
  - Class/module docstring: document `label` as an optional, opt-in
    per-ray integer field, with no default pipeline semantics.
- `xicsrt/jaxrt/_rays.py`:
  - Add AI-tool file header disclaimer if not already present.
  - `new_rays(num)`: add `'label': jnp.zeros(num, dtype=jnp.int64)`.
  - Update the module docstring's ray-bundle field list to include
    `label`, noting it is currently unused/inert.
- Tests (new, in `tests/`):
  - A test that builds a small scenario, has a source (or a thin
    monkeypatch/subclass) set `rays['label']` to known values, runs it
    through `Dispatcher.trace`, `_sort_raytrace`/`combine_raytrace` (i.e.
    a full `xicsrt.raytrace(config)` call), and asserts `label` values are
    preserved and correctly split between `found`/`lost` histories
    alongside `mask`.
  - A test that round-trips a results dict containing `label` through
    `xicsrt_io.save_results`/load (or directly through `mirhdf5`) and
    confirms `label` survives with the same values and dtype.
  - A jaxrt test (guarded by `pytest.importorskip('jax')`, matching
    existing jaxrt test conventions) confirming `new_rays` includes
    `label` and that it passes through `xicsrt.jaxrt.raytrace` unchanged
    (all zeros, correct shape).
- `devel/jaxrt_sync.md`: add a divergence-log entry describing the two
  engines' differing opt-in-vs-always-present conventions for `label`,
  noting there is no functional divergence since the field is inert on
  both sides as of this feature.
- `doc_source/userguide/development_projects.rst`: append one sentence to
  the existing "Make sure that RayDict is used everywhere" admonition
  noting `label` as a standard-but-optional field (matching `weight`).

## Out of scope

- Any actual labeling scheme or population of `rays['label']` with
  meaningful values — that is F035.
- Any new config option to "turn on" label tracking. The opt-in mechanism
  is simply "a source sets the key or it doesn't"; F035's `multi_voigt`
  branch is the gating condition there, not a new user-facing config flag.
- Changes to `RayArray.initialize()`.
- Any change to jaxrt beyond the inert `new_rays()` field.

## Verification

- `pytest tests/` (full suite) passes, including the new tests above.
- `python examples/example_00/example_00.py` still runs end-to-end
  (sanity check that `RayArray.zeros()` changes don't affect any code path
  that happens to call it).
- Manual check that a results HDF5 file containing a `label` array opens
  correctly with existing `xicsrt_io` load helpers.
