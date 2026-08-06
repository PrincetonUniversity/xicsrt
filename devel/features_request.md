# XICSRT Feature Requests

## F027 - Integrate FluxLookupTable into a renamed XicsrtPlasmaDesc
Started: 2026-08-06
Status: Pending

Follow-on to F026: wires the dependency-free `FluxLookupTable` (built by
F026) into `xicsrt_contrib`'s VMEC/DESC plasma source so that
`XicsrtPlasmaW7x` (and other subclasses) can use the fast lookup table for
`flx_from_car`/`bundle_generate` instead of always calling DESC's
`map_coordinates` directly, while transparently falling back to a real DESC
equilibrium for any operation the table cannot serve.

Planned changes:
- Rename `xicsrt_contrib.sources.XicsrtPlasmaVmec` ->
  `XicsrtPlasmaDesc` (file and class), and `initialize_vmec` ->
  `initialize_desc`. Rename `XicsrtPlasmaVmecDatafile` ->
  `XicsrtPlasmaDescDatafile`, also fixing its pre-existing broken import
  (`xicsrt.sources._XicsrtPlasmaVmec` -> `xicsrt_contrib.sources._XicsrtPlasmaDesc`).
- `wout_file` (key name unchanged) now accepts a VMEC `.nc`, a saved DESC
  `.h5` equilibrium, or a saved `FluxLookupTable` `.h5` file. The format is
  identified from file content (magic bytes + top-level HDF5 dataset
  names), not the extension.
- `self.eq` becomes a lazy-loading property: in lookup-table mode,
  `flx_from_car` uses `FluxLookupTable.car_to_flx` directly; any other
  access to `self.eq` (e.g. `car_from_flx`, or `XicsrtPlasmaW7x`'s direct
  `eq.compute`/`eq.M_grid`/`eq.iota` use) transparently loads the real DESC
  equilibrium on first access, using the source-equilibrium path recorded
  in the table's own metadata (or a same-named file next to the table).
  `bundle_generate`'s outside-LCFS validity check uses the table's own
  `is_valid` output in table mode instead of the DESC round-trip check.
- `cyl_from_car`/`car_from_cyl` switched to `xicsrt.tools.xicsrt_math`
  (pure geometry), since they never actually needed `self.eq`.
- New standalone CLI `xicsrt_contrib/xicsrt_contrib/tools/generate_desc_lookup.py`,
  extracted from Part A of the "2026-07-28 ... Part 11" logbook notebook,
  to build a `FluxLookupTable` from a DESC equilibrium file from the
  command line.
- Updated `xicsrt_analysis/w7x_npablant` (`XicsrtPlasmaW7x`,
  `xicsrt_w7x_npablant.calibrate_emissivity_scale`) to the renamed class
  and method.

---

## F026 - Persistent full-torus DESC flux-coordinate lookup table
Started: 2026-08-05
Status: Done (2026-08-06)

Part 10's ("Path C/E2, simplified") interpolated Cartesian -> flux-coordinate
table covered only a single small box around one poloidal cross-section and
was rebuilt from scratch every notebook run. This extends it to a table
covering a full half field period (exploiting stellarator symmetry to fold
the other half and NFP periodicity to cover the whole torus), with save/load
to disk so the table build is a one-time-per-equilibrium cost, and a
dependency-free query module usable without `desc`/`jax` installed. Intended
eventually to replace the direct `eq.map_coordinates` calls in
`xicsrt_contrib`'s `XicsrtPlasmaVmec` (tracked as separate future work, not
part of this feature).

Delivered: new logbook notebook `2026-07-28 - Desc Coordinate Transform
Performance Minimal Example - Part 11 - Full torus persistent lookup
table.ipynb` (desc-dependent; split into a fully independent "Part A: build
and save" and "Part B: load, validate, and profile" so Part B can be re-run
on its own, in a fresh kernel, from just the saved table file), plus a new
dependency-free `xicsrt_contrib/xicsrt_contrib/tools/flux_lookup_table.py`
module (`FluxLookupTable` dataclass: `car_to_flx`, `save`/`load` via plain
h5py, float32/lzf-compressed storage, plus a module-level `load()`
convenience wrapper) with no `desc`/`jax` import.

Verified: notebook executed end-to-end via `nbconvert` against the real
W7-X equilibrium (`wout_desc_solved.h5`, NFP=5): 0 errors, table build
(120 x 161 x 160 = 3.09M nodes, `CloughTocher2DInterpolator` per-plane
resample) took ~19 s one-time, saved table is 12.1 MB. Validation against
`eq.map_coordinates` used query points with the cylindrical toroidal angle
spanning `phi` in `[-2*pi, 3*pi]` (5 field periods each direction, both the
direct and mirrored half of each period): interior (`rho<=1`) rms `rho`
error 3.9e-5 (`rho<0.9`) / 7.1e-5 (`0.9<=rho<=1.0`), and `zeta` (the analytic
toroidal-angle passthrough) matched to <1e-9 rad in all zones, confirming
the fold/mirror/periodicity logic is correct, not just the interpolation. A
second validation pass at a random subsample of the table's own grid nodes
(isolating the resampling error from the `RegularGridInterpolator`
between-node interpolation error) showed rms `rho` error of 2.0e-6
(`rho<0.9`) / 5.7e-6 (`0.9<=rho<=1.0`), confirming most of the total error
comes from between-node interpolation, not resampling. Lookup measured
~580-610x faster than warm `map_coordinates` over the same full-torus
points. Loaded and queried the saved table successfully in a separate
Python environment with `desc`/`jax` not installed, confirming the query
module has no such dependency.

## F025 - Resumable W7-X training-set generation via a pending-results manifest
Started: 2026-08-04
Status: Done (2026-08-04)

`run_training_set` (Part 3d notebook) and `xicsrt_train_task.py` (SLURM
production driver) both selected configuration files to process by list
position (`num_analyze`, or a fixed `task_id * images_per_task` shard),
relying on `run_one_sample`'s per-file skip-if-exists check to avoid
redoing completed samples. This meant `num_analyze=100` processed the
first 100 config files (some possibly already done), not "the next 100
undone samples" - making it hard to resume a partially completed training
set (e.g. after a SLURM timeout) without re-scanning from the start.
Entirely within `xicsrt_analysis`; no changes to core xicsrt.

Changes:
- `xicsrt_results_util.py`: new `list_pending_config_files`,
  `write_pending_manifest`, `read_manifest`. `run_training_set` gained
  `use_manifest` (default True) and `manifest_path`; when `use_manifest`
  is True it (re)generates a pending manifest and applies `num_analyze`
  to that list. `overwrite` keeps its original meaning (replace an
  existing result once a sample is run) and no longer affects which
  configs are selected.
- New `production/xicsrt_manifest_task.py`: manual, one-time CLI wrapping
  `write_pending_manifest`, intended to be run once before submitting a
  SLURM job array so that every array task shards over one fixed,
  consistent pending list instead of each task recomputing it
  independently while others are still writing results.
- `xicsrt_train_task.py`: new `--use-manifest`/`--no-use-manifest`
  (default: use manifest) and `--manifest-path`; requires the manifest to
  already exist when enabled (fails fast with a pointer to
  `xicsrt_manifest_task.py` otherwise).
- `slurm_train.batch`: fails fast before launching Python if
  `MANIFEST_PATH` does not exist, rather than generating or waiting for
  it. Known limitation, not guarded against: resubmitting a job array (or
  rerunning `xicsrt_manifest_task.py`) while a previous array over the
  same config/output path is still running.
- Part 3d notebook markdown updated to describe the new pending-manifest
  behavior; no code cell changes needed.
- New pytest suite `xicsrt_analysis/w7x_npablant/tests/` (previously none
  existed for this package).

Verified: new pytest suite (10 tests) passes; full core `xicsrt` suite
(110 tests) unaffected; manual sandbox run of `xicsrt_manifest_task.py`
and `xicsrt_train_task.py` against fake config/result directories
confirmed correct manifest generation, shard selection, and the fail-fast
missing-manifest error.

## F024 - DESC flux-coordinate extrapolation outside the LCFS (Part 9 notebook)
Started: 2026-08-04
Status: Pending

Follow-through on the note in section 13 of the "Part 8 - Interpolated mapping"
notebook: the interpolation table clamps every exterior node to `rho = 1.0`,
which biases any interpolation cell straddling the LCFS. Part 9 replaces the
clamp with real extrapolation into a `rho <= RHO_MAX = 1.1` shell and quantifies
how far that can be trusted. Plan: `devel/plan_desc_interpolation_and_extrapolation.md`
(externally generated, with a measured-corrections addendum). Entirely a logbook
notebook; no changes to core xicsrt.

Decisive constraint, measured: DESC 0.17.2 `Equilibrium.map_coordinates` silently
hard-clamps to `rho = 1` outside the LCFS - no NaN, no warning, error exactly
`rho_asked - 1`. There is therefore no ground truth outside the LCFS, only a
choice of convention, and exterior values can only come from the forward map
(`eq.compute`, which accepts `rho > 1` cleanly and is essentially free).

Variants: `A/E0`, `A/E1`, `C/E0`, `C/E1`, `C/E2`, where `E0` is the Part 8 clamp,
`E1` is normal offset to the nearest LCFS foot point (`rho = 1 + d/L`,
`L = e_rho . nhat`), and `E2` is the DESC Zernike continuation (nominal).
`E3` (radial offset from the magnetic axis) was dropped.

Delivered: 16-cell notebook mirroring Part 8's section numbering, written to the
logbook (the previous Part 9 file, which held only the T1-T5 diagnostics used to
write the plan, was backed up to `.bak_pre_part9`; Part 9t left untouched).
Executed end-to-end via `nbconvert` against the real equilibrium: 0 errors, 58 s
of in-notebook work, all 7 plots rendered, both build-time assertions pass.

Verified results:
- Interior boundary band `0.98 <= rho <= 1.0`: rms `rho` error improves 10.9x
  (C/E0 -> C/E2) and 19.5x (A/E0 -> A/E1), while `rho < 0.9` is unchanged. This
  confirms Part 8's suspicion that the clamp, not table resolution, set the
  error floor near the boundary. Section 14 shows the clamped table plateauing
  under refinement while the extrapolated one keeps converging.
- Convention spread scales as `eps^1.82` in `rho` but `eps^0.84` in `theta`. At
  `rho = 1.1`: `rho` spread 13.4 mm, BELOW the 16.5 mm spectral ambiguity floor;
  `theta` spread 72.1 mm, 4.4x ABOVE it. Cause is the ~40 deg mean obliquity of
  `e_rho` to the flux surface, so DESC's continuation slides points tangentially
  while E1 does not. Corroborated independently by the interior `rho*dtheta`
  score, where C/E2 beats C/E1 by 9.1x.
- Net: the plan's headline claim holds for `rho` and fails for `theta`, so E2 is
  nominal on C1-continuity and `theta` grounds rather than "the conventions
  agree". Lookups run ~1400x faster than warm `map_coordinates`.

## F023 - Align W7-X SLURM production driver with the Part 3c/3d flat-file procedure
Started: 2026-08-04
Status: Done (2026-08-04)

`xicsrt_train_task.py` (F010 Phase 2) built each training image's config
in-memory per SLURM task and called `raytrace_mp`, saving a full `.tif`
image and full raw `.hdf5` per image. This diverged from the config-file
and flat-output procedure developed in the "Part 2b"/"Part 3c" notebooks
(saved JSON configs consumed one-by-one, `simplify_results`/`flatten_dict`
reduction to config + found-ray intersections only). Entirely within
`xicsrt_analysis`; no changes to core xicsrt.

Changes:
- New `xicsrt_analysis/w7x_npablant/xicsrt_results_util.py`: shared
  `simplify_results`, `flatten_dict`, `run_one_sample`, `run_training_set`,
  `list_config_files`, factored out of the Part 3c notebook (xarray/netCDF
  conversion deliberately not included; F020's combiner reads `.hdf5`
  directly).
- `xicsrt_train_task.py` rewritten: consumes a contiguous shard of
  pre-saved configuration files (`task_id * images_per_task` block)
  instead of generating configs per-seed, and runs a `multiprocessing.Pool`
  of `run_one_sample` calls (one single-process raytrace per config file)
  instead of `raytrace_mp` over one shared config. Per-image profile
  randomization/`calibrate_emissivity_scale` moved upstream to
  config-generation time; this driver only runs whatever each file
  specifies.
- New `xicsrt_config_task.py`: reference CLI mirroring Part 2b's
  `build_training_set_configs`. Not intended for SLURM (config generation
  is fast); kept for scripted/version-controlled use, e.g. generating and
  inspecting configs locally before uploading to the cluster.
- New notebook "Part 3d - Generating xicsrt results" (uses
  `xicsrt_results_util` instead of embedding the functions); Part 3c is
  kept as history.
- `slurm_train.batch` updated to pass `--config-path` through to
  `xicsrt_train_task.py`.

Verified: `xicsrt_config_task.py` reproduces the existing 1000-sample
config set bit-for-bit (mount-path prefix aside) for the same seed;
`xicsrt_train_task.py` run locally on a small shard produces the same
flat `.hdf5` output as `xicsrt_results_util.run_training_set`, including
the skip-if-exists and per-sample error-log behavior.

## F022 - Normalize W7-X emissivity_scale to give a consistent source ray count
Started: 2026-08-03
Status: Pending verification review

For the W7-X ML training set, per-sample generated ray counts vary by ~700x
(31-22533 detected, per F021's investigation), dominated by two multiplicative
factors upstream of the Poisson draw: `generate_random_emissivity` normalizes
the profile SHAPE to peak 1.0 but its volume integral varies ~157x with the
randomized zero-crossing radius, and `XicsrtPlasmaW7x.bundle_generate`'s
Te-dependent `I_tot/I_w` full-spectrum correction varies ~26x. This blocks
targeting a consistent ~1e4 rays/image and using a fixed-shape (rather than
ragged) array when combining samples for ML training.

Plan: `devel/plan_ray_count_calibration.md`. Entirely within `xicsrt_analysis`;
no changes to core xicsrt. Adds `XicsrtPlasmaW7x.shape_integral()`, which
computes the DESC-flux-surface-volume-weighted average of
`profile_emissivity(rho) * (I_tot/I_w)(Te(rho))`, normalized by the total
volume V(rho=1) so the result is dimensionless and independent of the
absolute VMEC volume, and `xicsrt_w7x_npablant.calibrate_emissivity_scale
(config)`, which sets `emissivity_scale` so this average equals a fixed
reference value (`TARGET_MEAN_EMISSIVITY`). `time_resolution` is then
hand-tuned once per campaign to hit the desired image ray count.

Known limitation, deliberate (per user direction): `shape_integral` averages
over the ENTIRE DESC flux-surface volume, while `XicsrtPlasmaGeneric`'s bundle
Monte-Carlo only samples the local diagnostic box (a small fraction of the
flux-surface volume at each rho). This is a geometric approximation, not
exact; different samples' generated ray counts are NOT expected to be equal
up to pure Poisson statistics, only substantially more consistent than
without calibration. Verified over 50 random profile seeds
(`verify_ray_calibration.py`): relative std of generated ray count drops from
0.99 to 0.10 and max/min ratio from 39.7 to 1.8 when a per-sample calibrated
`emissivity_scale` is used instead of one fixed value applied to all samples.
A whole-torus DESC `V_r(r)`-based integral without this box/torus-volume
normalization was tried first and rejected: it overcounts the emitting
volume the raytrace can actually see by ~200x, since the raytrace box
(`ysize=1.7 m` etc.) is a small sightline volume compared to the full torus
(circumference ~34 m).

Wired into both existing pipeline entry points: the Part 2b notebook's
`get_w7x_ml_config` and `xicsrt_train_task.update_config_for_image` both call
`calibrate_emissivity_scale` after `update_config_with_profiles`, so the
notebook and SLURM production paths produce identical calibrated configs for
the same seed.

Out of scope: detected/imaged ray count (Bragg efficiency varies with
Ti/velocity, a separate smaller effect), and the xarray/padding output-format
question (F020).


## F021 - Randomized spline profiles are seed-correlated
Started: 2026-08-03
Status: Done (2026-08-03)

Implemented as planned in `devel/plan_profile_seed_decorrelation.md`:
`xicsrt_spline.profile_seeds(seed, names)` derives one independent
`SeedSequence` per profile via `SeedSequence(parent.entropy, spawn_key=(ii,))`,
and `generate_random_profiles` (xicsrt_analysis) now passes one child seed to
each generator. The public signature `generate_random_profiles(seed)` is
unchanged, so `update_config_with_profiles`, `xicsrt_train_task` and all
notebooks inherit the fix without call-site edits.

Verified: max cross-profile |r| over 500 seeds drops from 1.000 to 0.103
(null 3-sigma is 0.134). Both rank-1 blocks are gone and the x_knots are no
longer shared. Two regression tests added to `tests/test_spline.py`
(110 pass); the correlation test was confirmed to fail at |r| = 1.000 when
the old shared-seed behavior is restored.

Also documented, per plan: a "Seeding" section in the `xicsrt_spline` module
docstring explaining why shared seeds correlate, and a note in
`generate_random_temp` that Ti and Te are drawn fully independently. Measured
over 2000 seeds this gives Ti > Te in 25.2% of samples and Ti > 5*Te in 3.8%
(the plan file's 24%/3.3% were from a smaller sample). A physics-constrained
alternative enforcing Te >= Ti for ECRH is described but deliberately not
imposed.

Not done, as scoped out: the 1000 configs and 199 samples were not
regenerated, and no new notebooks were added (Parts 2b/3c inherit the fix
unchanged).

`xicsrt_w7x_npablant.generate_random_profiles` passes the identical `seed` to
all five spline generators. Every generator opens with
`sample_interior_x(n_interior=3, ...)`, consuming the same three uniforms, then
draws its amplitudes from the same stream position. With `n_knots=5` everywhere
the streams are byte-aligned, so ~25 nominally independent ML labels collapse
onto ~3 random numbers per sample.

Measured over 600 seeds (correlation matrix of the scalar labels):

```
                  emiss_x2        Ti0        Te0  vperp_ion  vperp_ele    vpar_pk
      emiss_x2       1.000      0.020      0.020      0.020     -0.049     -0.049
           Ti0       0.020      1.000      1.000      1.000     -0.010     -0.010
           Te0       0.020      1.000      1.000      1.000     -0.010     -0.010
     vperp_ion       0.020      1.000      1.000      1.000     -0.010     -0.010
     vperp_ele      -0.049     -0.010     -0.010     -0.010      1.000      1.000
       vpar_pk      -0.049     -0.010     -0.010     -0.010      1.000      1.000
```

Two exact rank-1 blocks:
- Block A (r = 1.000): Ti_core, Te_core, v_perp,ion-root
- Block B (r = 1.000): v_perp,electron-root, v_par,peak
- x_knots identical across all five profiles, 500/500 seeds.

Proof of mechanism: the shared uniform variate is recoverable. For seeds
1234-1243, `(Ti0-200)/4800` and `(v_perp,ion+20e3)/20e3` agree to all printed
digits, as do `v_perp,elec/20e3` and `(v_par,peak+20e3)/40e3`. Equivalently
`Ti = 200 + (Te-200)*4800/9800` exactly, to machine precision.

As a training set this means Ti carries zero information independent of Te, and
the 199 existing samples explore a 3-D slice of the intended parameter space.

`xicsrt_spline` itself is NOT buggy: same seed -> same output is the documented
contract. The defect is in how the caller derives the five seeds.

Fix: add `profile_seeds(seed, names)` to `xicsrt/tools/xicsrt_spline.py`,
deriving one independent `SeedSequence` per profile via
`SeedSequence(parent.entropy, spawn_key=(ii,))`, and call it from
`generate_random_profiles`. See `devel/plan_profile_seed_decorrelation.md`.

Per user direction: Ti and Te are drawn FULLY INDEPENDENTLY (option a). No
backwards compatibility; the 199 existing samples and 1000 configs are
knowingly invalidated and are NOT regenerated as part of this feature.

Found while investigating why per-sample ray counts span 31 to 22533 in the
W7-X ML training set. The ray-count spread itself has a different root cause
(peak-vs-volume emissivity normalization, and the Te-dependent I_tot/I_w
restoration of the full Ar16+ spectrum) and is deliberately NOT tracked here,
at user request, to keep this change focused.


## F020 - xarray/netCDF results output format
Started: 2026-07-31
Status: Done (2026-08-04, still notebook-only, not yet in xicsrt itself)

Resolved (notebook "Part 4a - Combined ML training-set file"): combined
training-set files now read the per-sample `.hdf5` files directly (not the
sibling `.nc`), store found rays as a padded `(sample, ray, axis)` float32
array (NaN beyond `sample_count`) rather than a CF ragged array, and give
every kept config entry a `sample` dimension unconditionally (no varying/
constant split). Padding is only cheap because F022 narrowed the per-sample
ray-count spread from ~400x to ~1.8x; the varying/constant split was
dropped because repeating all ~121 config entries per sample costs only
~2.7% of file size at the 10k-100k sample scale. Verified end-to-end on the
100-sample set: 0 mismatches vs. the source `.hdf5` files. `None`/empty-
dict/list config entries are still dropped (no netCDF representation) and
recorded in `ds.attrs['dropped_keys']`.

Prototyped in notebook "Part 10 - Xarray results" converting the Part 9 flat
results dictionary into an `xarray.Dataset` and saving it as netCDF, as a
step towards an ML-friendly results format.

Key findings:
- netCDF attributes cannot hold `None`, `bool`, or `{}` values (raises
  `TypeError` on `to_netcdf`), so flat config entries must be stored as
  xarray data variables, not `attrs`. As 0-D/1-D data variables, `bool`,
  `int`, `float`, `str`, and `ndarray` all round trip exactly.
- `None` and empty-`dict` leaf values have no netCDF representation and are
  dropped on write. The resulting `.nc` is therefore a derived, lossy view
  of the sibling `.hdf5` file (`mirhdf5` stores `None`/`{}` exactly); the
  `.hdf5` remains authoritative and the dropped keys are always recoverable
  from it.
- `zarr` and `h5netcdf` are not installed in the environment used for this
  prototype; only the `netCDF4` backend was exercised.
- `config['general']['random_seed']` defaults to `None`, so an `.hdf5`/`.nc`
  pair saved from a `None`-seed run cannot be regenerated bit-for-bit by
  re-running `xicsrt.raytrace` on the same config; only loading the saved
  files is exact. Setting an explicit integer seed removes this caveat (and
  incidentally removes `random_seed` from the dropped-key list).

Round-trip verified end-to-end on a real W7-X raytrace (107 flat keys, 27277
found rays): 0 unexpected missing keys, 0 mismatched values after loading
the `.nc` back, aside from the expected `None`/`{}` drops.

If this is promoted to a real `xicsrt_io` output format, it should reuse
`flat_to_dataset`/`dataset_to_flat` from the Part 10 notebook as a starting
point, and decide whether to record a `dropped_keys` manifest in `ds.attrs`
(deliberately omitted in the prototype).


## F019 - `XicsrtPlasmaToroidal` evaluates profiles at the wrong radius
Started: 2026-07-31
Status: Done 2026-07-31

Implementation: `flx_from_car` now puts `rho = r/a` in column 0,
`rho_from_car` returns it directly, and `car_from_flx` multiplies by `a`
without the sqrt. Verified with the case below: `rho_from_car` returns
exactly 0.5 and `car_from_flx(flx_from_car(x))` is the identity.

Found incidentally while unifying flux-coordinate conventions for F014.
`XicsrtPlasmaToroidal.flx_from_car` puts `r**2/minor_radius` in column 0,
then `rho_from_car` takes the square root, giving `rho = r/sqrt(a)` instead
of `r/a`. Two consequences: every profile is evaluated at the wrong radius,
and the LCFS boundary is wrong because `r/sqrt(a)` does not reach 1.0 at the
edge. Separately, `car_from_flx` multiplies by `a` after the sqrt is undone,
so `car_from_flx(flx_from_car(x))` is not the identity.

Verified numerically with `major_radius=5.0`, `minor_radius=0.5` and a point
at true minor radius `r=0.25` (so rho should be 0.5):
`rho_from_car` returns 0.354, and the round trip recovers `r_min = 0.177`
instead of 0.25. Both defects vanish when `minor_radius = 1.0`, which is
presumably why this survived.

Affects `XicsrtPlasmaToroidal` and `XicsrtPlasmaToroidalDatafile`. Prior
results from toroidal plasma configs used the wrong radius. Fix: column 0 of
`flx_from_car` becomes `rho = r/a`, `rho_from_car` returns it directly, and
`car_from_flx` multiplies by `a` without the sqrt. The user chose to fix this
as part of the F014 convention unification rather than defer it.


## F018 - eV/keV convention cleanup (remaining items)
Started: 2026-07-31
Status: Done 2026-07-31 (in-scope items; I-4 and I-5 below remain deferred)

Implementation: I-1 (spline generators now emit eV / m/s), I-2
(`temperature`/`temperature_e` init `np.ones` -> `np.zeros`) and I-3
(`temperature_e_scale` added to `XicsrtPlasmaToroidal`) are done. The eV
convention is now pinned on both engines by `tests/test_doppler_sigma.py`
(anchor: 1000 eV, argon, 3.9492 A -> sigma 6.474e-4 A). I-4 and I-5 are
in other repos / dead code and remain open.

A full audit of temperature units across xicsrt, xicsrt_contrib and
xicsrt_analysis confirmed that eV is the canonical unit and is forced by the
physics: all four copies of the Doppler-sigma formula divide by
`electron volt-joule relationship`, so an input in keV would give a sigma
too small by sqrt(1000). The audit found five issues; I-1, I-2 and I-3 are
being fixed under F014. The remainder are logged here:

- I-4: `xicsrt_contrib/.../_XicsrtPlasmaImas.py:431` converts IMAS
  `t_i_average` (eV per the IMAS data dictionary) with `* 1.e-3`, producing
  keV. Currently dead in xicsrt_contrib because `get_temperature` reads
  `temperature_profile` instead, but the same bug at
  `xicsrt_iter/xrcscore_npablant/objects/_XicsrtPlasmaImas.py:150` IS live.
- I-5: `_XicsrtPlasmaBundleSource.py:238` forwards `temperature_e` to per-
  bundle sources, but `XicsrtSourceGeneric` does not define that option, so
  it is silently dropped. By design (`strict=False`), but undocumented, and
  it means electron temperature is unreachable from a per-bundle source.
- `xicsrt_iter` has already diverged to
  `get_velocity(self, rho, veloc_interp)` and will not match the F014 hook
  signature. Out of scope there, noted here.

Also noted: `XicsrtPlasmaGeneric`'s docstring override degrades several
inherited entries to "No documentation yet", including the sigma formula
that establishes the eV convention in `XicsrtSourceGeneric`.


## F017 - Flux-surface-average to local flow conversion
Started: 2026-07-31
Status: Pending (future work, not started)

The F016 velocity implementation treats perpendicular and parallel velocity
as flux-surface functions. This is physically WRONG. A correct treatment must
account for flow incompressibility and Pfirsch-Schlueter flows, i.e. the
local flow varies over a flux surface even when the flux-surface-average
quantities are fixed.

Converting an FSA flow to a local flow requires substantial work and is
deliberately deferred. The F016 implementation carries an explicit code
comment stating the assumption and its inadequacy.


## F016 - Port the W7-X velocity profile from stelltools to DESC
Started: 2026-07-31
Status: Pending (plan approved; see devel/plan_lalston_integration.md Phase 5)

`XicsrtPlasmaW7xSimple.get_velocity` is dead code behind `if False:`; it
depends on LIBSTELL/STELLOPT wrappers in `stelltools`, which is not
importable in the current environment. `enable_velocity` and
`enable_flux_compression` are therefore inert options today, and the first
training set was to be generated with velocity disabled.

Reimplement using DESC, which the plasma source already uses for all
coordinate transforms. Confirmed equivalences: `e^rho` for
`gradrho_car_from_flx`, `B` for `b_car_from_flx`, `<|grad(rho)|>` and
`<|B|>` for the flux-surface averages, all with `basis='xyz'`. Note
stelltools flux coordinates use `s` while DESC uses `rho = sqrt(s)`.

Two mandatory performance measures (both measured, see the plan): seed
`data={'iota': eq.iota(rho)}` into `eq.compute` to avoid an internal override
grid that OOM-kills the process at the production bundle_count of 1e4, and
feed fixed-shape padded arrays to avoid jax retracing.

Depends on the F014 `get_velocity(point_flx)` signature change. Carries the
F017 caveat comment.


## F015 - W-line emissivity normalization for the W7-X Ar16+ model
Started: 2026-07-31
Status: Done 2026-07-31

Implementation: `XicsrtPlasmaW7x.bundle_generate` evaluates the Ar16+ line
model once per iteration, caches it for `get_line_parameters`, and scales
`bundle_input['emissivity']` by `I_tot/I_w` before the Poisson draw.

One addition beyond the plan, approved by the user during implementation: a
`te_min` config option (default 200 eV) below which the emissivity is set to
zero. The measured `I_tot/I_w` diverges at low Te (2.4e4 at 100 eV, 2.5e38 at
11 eV) because the model's satellite emission is unphysical where there is no
Ar16+ charge-state population; without the cutoff, edge bundles blow the ray
budget. Verified in situ: 1,123,000 w-line rays counted against a 1,123,803
target (-0.76 sigma).

For the W7-X Ar16+ model, the `emissivity` config option should mean the
emissivity of the 'w' line only, not of the entire Ar16+ spectrum. Because
the ratio of total to w-line intensity is Te-dependent, the number of
photons drawn from the multi-Voigt model must be scaled per bundle by
`I_tot / I_w`, applied to `bundle_input['emissivity']` BEFORE the Poisson
draw in `create_sources`.

Photon statistics remain exact: this scales the true expected photon number,
it is not a reweighting. Verified at Te = 1.0 and 4.0 keV (w-fractions 0.3633
and 0.6807), recovering 50,107 and 49,959 w-line rays against a 50,000
target (+0.48 and -0.18 sigma).

Originates from lalston commit `3ba693d`, which used
`size * (1 + I_w/I_notw)`. That factor is different (up to ~5x, and the two
cross over with Te) and the user confirmed `I_tot/I_w` is the intent. That
commit also returned more wavelengths than `size`. It does not port directly
because F010 removed `random_wavelength_ar16_voigt` and moved the Ar16+
physics to the per-bundle `get_line_parameters` hook.

Per user direction: implement on the shared `XicsrtPlasmaW7x` base class,
with no config option and no dedicated test.


## F014 - Randomized spline plasma profiles for the W7-X ML training set
Started: 2026-07-31
Status: Done 2026-07-31 (all phases; see devel/plan_lalston_integration.md)

Implementation notes: all 8 plan phases done. The stale
`suli/suli2026_alston/xicsrt_contrib` clone was deleted; the `devel_lalston`
branches are still deliberately kept. The two typo fixes from lalston commit
`3ba693d` had no surviving target (F010 had already rewritten those comments
correctly in `_XicsrtPlasmaW7xSimple.get_line_parameters`), so nothing was
ported. `make_profile_dict` returns numpy arrays, not lists: `xicsrt_io`
already converts arrays on save/load, matching the convention for other
array-valued config options. Profiles are shared between the notebook and
SLURM paths via `xicsrt_w7x_npablant.update_config_with_profiles`.
Verification: 108 tests pass (55 baseline + new spline/doppler suites),
`example_00` runs, and a config-save -> load -> raytrace round trip works.

Integrate the SULI 2026 (L. Alston) work into the optimized branch so that
randomized spline plasma profiles can drive the W7-X ML training-set
generation. `xicsrt/tools/xicsrt_spline.py` is currently orphaned (zero
importers, no `profile_*` config option anywhere) and
`xicsrt_train_task.py:83 update_config_for_image` is an empty stub; these are
the two ends of a bridge that was never built.

Scope:

- Fix `xicsrt_spline.py`: temperature ranges in eV with the randomized peak
  PRESERVED (normalizing it would destroy the primary ML label), explicit
  velocity ranges in m/s, remove the undeclared plotly import, fix the
  hardcoded 5-element mask arrays, add `spline_from_profile`.
- Change `get_velocity(rho)` to `get_velocity(point_flx)` across
  `XicsrtPlasmaGeneric`, `XicsrtPlasmaVmec` and `XicsrtPlasmaToroidal`, so
  that DESC receives fixed-shape input (measured 1.8 s vs 0.11 s per
  retrace). The scalar hooks keep `rho`. Also fixes I-2 (`np.ones` ->
  `np.zeros` for the bundle temperature arrays) and I-3 (missing
  `temperature_e_scale`), and adds the first regression test for the eV
  convention.
- New `XicsrtPlasmaW7x` base class (replacing dead `W7xPlasma`) with
  `XicsrtPlasmaW7xSimple` and a new `XicsrtPlasmaW7xProfile` beneath it. No
  polynomial fallback in the spline class.
- New logbook notebooks Part 2 and Part 3; wire `update_config_for_image`.

The `devel_lalston` branches are deliberately NOT deleted; that is a
remaining step to be done in a later session. The colleague's
`xicsrt_w7x_lalston` module does not exist in any accessible repo and is
dropped in favor of `xicsrt_w7x_npablant`.


## F013 - `arcsin` RuntimeWarning in InteractCrystal from untruncated Voigt tails
Started: 2026-07-30
Status: Done (2026-07-30)

Reported: after F010 (direct Voigt/Cauchy sampling, no domain truncation),
running the W7-X Ar16+ notebook produces
`RuntimeWarning: invalid value encountered in arcsin` from
`InteractCrystal.angle_calc` (`bragg_angle[m] = np.arcsin(W[m] / (2 *
crystal_spacing))`).

Root cause: before F010, wavelengths were drawn from a truncated inverse-CDF
table (`multi_voigt_cdf_tab`, cutoff domain a few `1e-3` Å around the line
centers), so `wavelength / (2d)` never left `[-1, 1]`. F010's direct
`center + Normal(0, sigma) + Cauchy(0, gamma)` sampling has genuine heavy
Cauchy tails with no cutoff, so a tiny fraction of sampled wavelengths land
outside `[-2d, 2d]` (confirmed: samples up to ~200 Å at 5000 bundles). For
such a wavelength no incidence angle satisfies Bragg's law, so `arcsin`
correctly returns `nan`. That `nan` propagates into a `nan` reflection
probability in `rocking_curve_filter` for all three rocking curve types
(step/gaussian/file), and `nan >= test` is `False` in numpy, so the ray is
correctly rejected. Statistics and physics are unaffected; this is a noisy
but harmless warning, not a correctness bug.

Fix: wrapped the `arcsin` call in `np.errstate(invalid='ignore')` in
`xicsrt/optics/_InteractCrystal.py::angle_calc`, with a comment explaining
why `nan` is expected and safely handled downstream. No change to
`rocking_curve_filter` or the wavelength sampling itself (clamping/filtering
the wavelength would be a physically incorrect approximation).
`jaxrt/interact/_crystal.py` has the identical `jnp.arcsin` expression but
`jax.numpy.arcsin` returns `nan` silently (no warning), so no jaxrt change
was needed; documented in `devel/jaxrt_sync.md`'s convergence log.

New test: `tests/test_interact_crystal_arcsin.py` asserts `angle_calc` /
`angle_check` do not warn for an out-of-domain wavelength and that such a
ray is masked out, for the 'step' and 'gaussian' rocking curve types. A
'file' rocking curve variant was not added: `xicsrt_bragg.read_xop`
references the undefined name `m_log` (should be `log`) unconditionally,
so it currently raises `NameError` on every call regardless of this fix.
Pre-existing, unrelated bug; not filed as its own feature, noted here only.


## F012 - Randomize found-ray order in the history (`shuffle_history`)
Started: 2026-07-30
Status: Done (2026-07-30)

Plasma sources emit rays in contiguous per-bundle blocks
(`XicsrtPlasmaGeneric.create_sources`: `bundle_index = np.repeat(np.arange(len(counts)), counts)`),
and this block order survives unshuffled through `_sort_raytrace` and
`combine_raytrace` into the returned `found` history. A user taking a naive
subset (e.g. `history['detector']['origin'][:1000]`) would get rays from only
a handful of bundles instead of a statistically representative sample of the
plasma.

Implementation: new `general.shuffle_history` option (default `True`).
`_sort_raytrace` permutes `w_found` (the found-ray index array) with a
dedicated `rng_shuffle` generator, independent from the existing `rng` used
for lost-ray subsampling, so enabling/disabling the shuffle cannot change
which lost rays are retained and the shuffle cannot perturb the global
`np.random` stream that generates the rays. The `lost` rays are already
unordered (`rng.choice(..., replace=False)`) and are not reshuffled.
`raytrace_single` derives `rng_shuffle` via `np.random.SeedSequence(seed,
spawn_key=(1,))`, keeping `rng_lost`'s stream byte-identical to before this
change. `xicsrt/jaxrt/_engine.py` mirrors the same generator setup since it
imports `_sort_raytrace` directly.

Shuffling is done at the found-ray-selection stage, not in the source: a
benchmark of shuffling inside `XicsrtPlasmaGeneric.create_sources` (or the
`bundle_index` array feeding it) cost 13-31% of total raytrace time for the
'voigt' wavelength distribution, and >100% for 'multi_voigt' (random gather
into the per-bundle line table destroys locality). Shuffling only the found
rays in `_sort_raytrace` costs approximately 0.005% of raytrace time at a
typical x-ray efficiency (found << traced), rising to ~2% only when a large
fraction of traced rays are found. Cross-iteration/run shuffling in
`combine_raytrace` was also rejected: each iteration is already an unbiased
sample of the same config, so within-iteration shuffling alone makes any
prefix of the found history a fair sample.

New tests: `tests/test_history_shuffle.py` (unit-level `_sort_raytrace`
behavior on synthetic bundle-blocked data, plus an end-to-end check with a
real point-bundle plasma source). `testing/compare_raytrace_regression.py`
sets `shuffle_history=False` in its baseline scenario (for the Tier A exact
comparison against pre-F012 source trees) and `_call_sort` now explicitly
disables shuffling for Tier B, which tests lost-ray subsampling and is
orthogonal to this feature.

No bundle id is stored in the ray arrays, so shuffling is currently the only
way to lose track of bundle membership; recovering it (if ever needed) would
require a separate per-ray bundle-index feature, deliberately left out of
this change.


## F011 - XicsrtPlasmaCubic ignores the `velocity` option (no Doppler shift)
Started: 2026-07-30
Status: Pending (pre-existing bug, found while verifying F010 addendum)

`XicsrtPlasmaGeneric.create_sources` applies a Doppler shift from
`bundle_input['velocity']`, but `XicsrtPlasmaCubic.bundle_generate` only
fills `temperature` and `emissivity`; it never copies `self.param['velocity']`
into `bundle_input['velocity']`. The array therefore stays at its zero
initialization and the configured `velocity` is silently ignored.

Reproduce (monochrome line, temperature 0, so the shift is unambiguous):

    velocity = [0, 0, 3e6] -> mean wavelength shift 0.0 (expected ~ -0.0395 A)

`XicsrtPlasmaToroidal` does this correctly (`bundle_input['velocity'][m] = ...`),
as does the new `XicsrtPlasmaBundleSource`, so the bug is specific to
`XicsrtPlasmaCubic`. Verified present on 6a94671 (pre-dates the F010
addendum work). Fix is a one-line addition to `bundle_generate`, but it
changes results for any existing `XicsrtPlasmaCubic` config that sets
`velocity`, so it is left for explicit approval rather than folded into an
unrelated change.


## F010 - W7-X ML training-set acceleration (numpy path)
Started: 2026-07-30
Status: Phase 1 implemented 2026-07-30 (verification notes below); Phases 2-3
pending.
Plan: devel/plan_w7x_training_accel.md

Addendum (2026-07-30, Done): Phase 1b's vectorization of
`XicsrtPlasmaGeneric.create_sources` removed the ability to model each
bundle with an arbitrary, user-selectable ray source (the old code
instantiated a fresh `XicsrtSourceFocused` per bundle). This capability was
reintroduced as a new example class, `XicsrtPlasmaBundleSource`
(`xicsrt/sources/_XicsrtPlasmaBundleSource.py`), which loops over bundles
and dispatches a ray source chosen by the config option
`bundle_source_class` (resolved through the dispatcher's plugin search
paths, so a user's own source class works). `XicsrtPlasmaGeneric` remains
the fast, production default; the new class is documented as a worked
example, not a performance-equivalent replacement.

Enabling this required a small framework change: elements previously had
no way to see the plugin search paths used to find them (`general.pathlist`
+ `general.pathlist_default`), since only their own element-level config is
passed down. `Dispatcher._instantiate_single` now sets
`obj.param['pathlist']` on every element it constructs, and
`ConfigObject.__init__` seeds a builtin-only default so directly
constructed elements (e.g. via `xicsrt_public.get_element`) still work. The
class lookup half of `_instantiate_single` was factored out into
module-level `xicsrt.objects._Dispatcher.find_xicsrt_class`, reused by the
new plasma class. `pathlist` is a `param`-only key (never added to
`default_config`) so it is never written into a saved config file, which
would leak machine-specific absolute paths.

Phase 1 implementation notes (2026-07-30):
- 1a: voigt_random / multi_voigt_random rewritten as exact direct sampling
  (Normal + Cauchy; mixture by intensity weights). New batched sampler
  multi_voigt_random_batched (per-bundle line tables, one call per iteration).
  ~108x faster than the per-bundle CDF-table build at Ar16+ scale, and more
  exact (no tail truncation at `cutoff`, no interpolation error). Deleted
  xicsrt_voigt_multi_jax.py, xicsrt_faddeeva_jax.py, tests/test_voigt_multi_jax.py,
  _USE_JAX_VOIGT_MULTI. Removed multi_gridsize/multi_cutoff config options.
  New statistical tests: tests/test_voigt_direct.py (KS vs analytic pdf; KS vs
  the retained CDF tables with tolerances above the tables' own ~0.2-0.6%
  truncation bias). jaxrt _wavelength.py mirrored to direct sampling.
- 1b: XicsrtPlasmaGeneric.create_sources vectorized: per-bundle Poisson draws,
  bundle_index = repeat(arange, counts), vectorized origins/directions/
  wavelengths/Doppler. XicsrtSourceFocused no longer instantiated per bundle.
  vector_dist_isotropic and solid_angle_isotropic accept array spread.
  Verified 5-sigma statistical equivalence vs baseline (testing/
  compare_f010_plasma.py: generated 0.55 sigma, detected 0.66 sigma,
  centroids <0.05 px over 5 seeds).
- 1c: ar16_voigt and the xics_jax import removed from public xicsrt.
  New hook XicsrtPlasmaGeneric.get_line_parameters (default: broadcast static
  line_* config); XicsrtPlasmaW7xSimple overrides it with a vmapped+jit'd
  xics_jax._compute_line_params over all bundle (Ti, Te) pairs (one call per
  iteration). W7X config now uses wavelength_dist='multi_voigt'. Also removes
  the F009 xics_jax-import exposure from the public repo (F009 itself still
  open for the analysis repo).
- 1d: XicsrtPlasmaVmec caches the loaded DESC equilibrium across iterations;
  map_coordinates now always called with fixed bundle_count-shaped arrays
  (masked rows padded) to avoid jax recompilation on masked-count changes.
- Measured: W7X model (10k bundles, 3 iter, single process, M1) wall
  138.8s -> 27.4s (5.1x). Direct sampling changes per-seed results
  (statistically identical, not bit-identical).
- Version bump 0.8.13 -> 0.9.0 (config options removed; behavior change).

Goal: generate 10,000 W7-X training images (~1e6 detected counts each) on the
Princeton Stellar cluster within a ~4096-core x 24-48 h envelope. Requires
roughly 2x end-to-end speedup of the numpy engine plasma path relative to the
Stellar run_08 baseline (12 workers x 8 threads, 21 core-h per 1e6-count image).

Scope (Phase 1, implement):
1a. Direct Voigt sampling: replace CDF-table sampling in voigt_random /
    multi_voigt_random with exact Normal+Cauchy mixture sampling; delete
    xicsrt_voigt_multi_jax.py / xicsrt_faddeeva_jax.py (obsoletes F004 sampler).
1b. Vectorized create_sources in XicsrtPlasmaGeneric (remove per-bundle
    XicsrtSourceFocused loop); vectorize xicsrt_spread over array spread.
1c. Relocate ar16_voigt / xics_jax out of public xicsrt into w7x_npablant via
    a per-bundle line-parameter hook (also resolves F009 exposure).
1d. DESC equilibrium caching + fixed-shape map_coordinates (xicsrt_contrib).
Phase 2 (implement): SLURM job-array production template.
Phase 3 (document only): hybrid numpy-generation -> jaxrt GPU optics design
    note in devel/plan_hybrid_gpu.md.

Constraints: exact photon statistics (no reweighting; per-image independent
sampling), readability first, no backwards-compat shims, jaxrt sync per
devel/jaxrt_sync.md, minor version bump on completion.


## F009 - `xics_jax` import consumes global RNG stream on first use in a process
Started: 2026-07-28
Status: Pending

`import xics_jax` at module scope in `sources/_XicsrtSourceGeneric.py:30` draws
from the global `np.random` stream as a side effect of the import itself:

    np.random.seed(12345) -> stream position 624
    import xics_jax       -> stream position 10

The plugin dispatcher imports element modules lazily on the first
`instantiate()`, so that draw lands *between* `np.random.seed()` and ray
generation on the first raytrace in a process, but not on later ones. Two
identical `raytrace()` calls in one process therefore give different results
(measured: nfound = 27, then 29).

Consequences:
- `random_seed` does not currently guarantee reproducibility, contrary to its
  documentation in `xicsrt_config.default_config`.
- Any A/B comparison must import `xics_jax` up front or run each case in a
  fresh process, or the confounder swamps the signal.

Suggested fix: save and restore the global `np.random` state around the
`xics_jax` import, or move the import so it cannot occur after seeding.


## F008 - Raytrace not reproducible on the first run of a process (seed 0)
Started: 2026-07-28
Status: Pending

Discovered incidentally while validating F007; NOT caused by F007 (reproduced
on clean HEAD with no modifications).

Symptom: with `random_seed=0`, running the same config three times in a single
process gives detector counts 41456, 41661, 41661. Run 1 differs from runs 2
and 3; runs 2+ are stable. Seeds 1-7 are stable from the first run.

Implication: the first raytrace in a fresh process does not see the configured
seed state, so a single-shot script is not reproducible against a repeated one.
This silently undermines any "set random_seed for reproducibility" workflow and
would also affect multiprocessing workers, which each run exactly one first run.

Suspected cause: seed application order in `xicsrt_raytrace` / config setup,
where `0` may be being treated as falsy somewhere, or the global `np.random`
stream is touched before seeding. Not yet diagnosed.

Next step: bisect where the global RNG is first consumed relative to seeding.


## F007 - Acceleration of einsum and vector operations (numpy engine)
Started: 2026-07-28
Status: Implemented, pending review
Baseline commit: 7390575

Goal: reduce single-core cost of the numpy raytracing hot path.

IMPORTANT framing correction. This entry was opened on the premise that
`np.einsum` is not parallelized over CPUs and that switching to BLAS calls
would recover multicore scaling. That premise is wrong and was disproved by
measurement:

- Every hot array in XICSRT has shape (N,3). Such operations are
  memory-bandwidth-bound, not FLOP-bound.
- `a @ M` for a (2e6,3) @ (3,3) takes 16.5 ms on 1 thread and 16.4 ms on 10
  threads, i.e. zero scaling, while a 2000^3 gemm in the same process scales
  59.8 -> 31.3 ms. The BLAS alternative is therefore just as serial as einsum.
- This independently corroborates the F006 finding that einsum does not use
  threaded BLAS, and extends it: the BLAS replacement does not either.

Cores must therefore continue to come from multiprocessing over runs (F006).
What this entry actually buys is single-core efficiency.

Key corollary: `np.einsum('ij,ij->i', a, b)` is the FASTEST available row-wise
dot product (8.0 ms vs 15.9 ms for `(a*b).sum(axis=1)` at N=2e6). Those call
sites are already optimal. This is a targeted change, not an einsum purge;
"de-einsumming" the row-wise dots would be a ~2x regression.

Measured hot spots (1e7 rays, spherical crystal, keep_history=False; c_einsum
itself is only ~7% of runtime):

| operation                                   | current  | replacement | gain |
|---------------------------------------------|----------|-------------|------|
| `np.linalg.norm(a, axis=1)`                  | 14.1 ms  | 7.5 ms      | 1.9x |
| `einsum('ij,ki->kj', M, a)`                  | 28.8 ms  | 16.0 ms     | 1.8x |
| `einsum('ij,ijk->ik')` + (N,3,3) assembly    | 58.3 ms  | 19.8 ms     | 2.9x |

Planned work (hot path only, by user decision):
1. `tools/xicsrt_math.py` `magnitude`/`normalize`: `np.linalg.norm(v, axis=1)`
   -> `np.sqrt(np.einsum('ij,ij->i', v, v))`, with a comment.
2. `objects/_GeometryObject.py` `vector_to_external`/`vector_to_local`:
   einsum -> `vector @ orientation` / `vector @ orientation.T`. Preserve the
   existing `copy=` handling and the `vector[:]` in-place writeback.
3. `sources/_XicsrtSourceGeneric.py` `random_direction`: drop the (N,3,3)
   rotation-matrix buffer in favour of the component-sum form already used by
   the jax engine. Single code path retained.
4. `make_normal` in `_XicsrtSourceGeneric.py` and `_XicsrtSourceDirected.py`:
   normalize the constant axis once instead of N times. Both files need the
   edit; the dispatcher loads element modules by file path, so patching the
   base class alone does not cover the override.

Out of scope: all `einsum('ij,ij->i')` row-wise dots (already optimal); mesh,
torus, cylinder, mosaic, xicsrt_spread and filter call sites; the
`location_from_distance` masked-gather rewrite (71 -> 33 ms via `where=`,
real but a masking-semantics change rather than an einsum one -- candidate for
its own entry).

Measured result (all four changes applied).

Per-function, N=2e6, base -> new:

| function            | base     | new      | gain |
|---------------------|----------|----------|------|
| `xm.magnitude`      | 14.0 ms  | 7.7 ms   | 1.8x |
| `xm.normalize`      | 20.5 ms  | 14.4 ms  | 1.4x |
| `make_normal`       | 29.8 ms  | 9.1 ms   | 3.3x |
| `random_direction`  | 219.8 ms | 155.7 ms | 1.4x |

End-to-end (1e7 rays, spherical crystal, keep_history=False), separate
processes, best of 3, run against a clean git worktree for the baseline:

| source           | base    | new     | gain  |
|------------------|---------|---------|-------|
| SourceDirected   | 5.26 s  | 4.90 s  | 1.07x |
| SourceFocused    | 2.80 s  | 2.61 s  | 1.07x |
| SourceGeneric    | 4.71 s  | 4.29 s  | 1.10x |

MEASUREMENT CAVEAT, recorded so it is not repeated. An earlier figure of 1.24x
for this same change set was wrong. It came from monkeypatching the baseline
and the optimized version into a *single* process and timing them one after
the other; the second measurement benefits from warmed allocator and page
cache state. Measuring each variant in its own process, with the baseline
taken from a clean `git worktree`, gives ~1.07-1.10x. Always benchmark engine
changes in separate processes.

So the honest gain is ~7-10% end-to-end, not the ~24% first estimated. The
per-function speedups are real and reproducible; they are simply a smaller
share of total runtime than the microbenchmarks suggested, because the hot
path is spread across many masked-gather and reduction operations that this
change does not touch (see `location_from_distance`, `check_bounds`,
`make_image` in the profile).

Correctness: detector images are bit-identical to the baseline across 28
cases (4 source configurations x 7 seeds), compared via a position-weighted
image checksum. `pytest tests/` 36 passed. Photon statistics are untouched:
no sampling, masking, or RNG-consumption changes; only last-bit float
reassociation. No config, API, or output-dict change, so no version bump.

Also fixed here (pre-existing, unrelated to the optimization):
`examples/example_01/example_01.py` referenced the stale class name
`XicsrtOpticCrystalSpherical`, which no longer exists, so the example raised
"Could not find ... in available objects" on the baseline commit. Renamed to
`XicsrtOpticSphericalCrystal` to match `optics/_XicsrtOpticSphericalCrystal.py`.
The companion `example_01.ipynb` already used the correct name, which is why
the drift went unnoticed. All three examples now run, and example_01 is usable
as a regression check again.


## F006 - Raytrace memory and multiprocessing instrumentation
Started: 2026-07-28
Status: Implemented (2026-07-28), pending user verification

Full approved plan: devel/plan_raytrace_memory.md
Baseline commit: e89a3bd

Implementation summary (2026-07-28):
- Regression harness `testing/compare_raytrace_regression.py` (temporary, not
  for master): Tier A bit-exact on found history / images / meta, Tier B
  statistical on lost rays, over 4 scenario variants (history, nohistory,
  multi-run, multiprocessing), each at `number_of_iter=3`. Self-tested by
  injecting real source mutations: a 1 ULP ray perturbation and a global-RNG
  stream shift both correctly FAIL the harness.
- Instrumentation (all verified bit-identical): real `mp: pool_join` /
  `mp: result_transfer` / `mp: combine` timers, confirming defect 3 (the old
  `mp: gathering` measured 12 us against 1.27 s of real cost);
  `profiler.getResults`/`profiler.merge` plus a pool initializer so worker
  timings surface in the parent under `spawn` as well as `fork`; per-iteration
  found/lost/history-bytes/peak-RSS logging.
- Fix 4 (`_sort_raytrace`): dedicated `np.random.Generator` + `choice`,
  measured 22.2x faster (0.0787 s -> 0.0035 s at 4.4e6 rays). This shifts the
  global RNG stream for `num_iter > 1`, contrary to the plan's Finding 6; see
  Finding 8 in the plan for the correction and a four-part proof that the
  change is purely a stream shift (compensation test is bit-exact; detected
  counts and per-ray distributions statistically identical, all p > 0.6).
  User-approved 2026-07-28.
- Fix 5 (`make_image`): `np.bincount` scatter-add, bit-identical, 75x faster
  than the per-ray loop and 13x faster than `np.add.at`.
- Fix 6 (`raytrace_single`): releases the dispatcher history each iteration.
  Bit-identical; measured 1629 MB -> 1175 MB peak RSS (-27.9%) at 2.44e6
  rays/iter, matching one full history copy (453 MB predicted) to 0.2%.
- Fix 7 (`combine_raytrace`): allocates from the *input* ray keys with
  `np.empty`, which fixes the dropped `weight` array; frees inputs as they are
  consumed behind a new opt-in `consume_input` flag (default False, so the
  documented public usage does not have its inputs destroyed). Verified
  `weight` is now present and correct in both engines and survives the hdf5
  round-trip.
- jaxrt: fixes 4 and 7 propagate automatically via the direct import in
  `jaxrt/_engine.py`; the "known benign quirk" entry in devel/jaxrt_sync.md is
  resolved and deleted. No divergence introduced.
- Verification: pytest tests/ (36 passed), pytest tests/jaxrt/ (10 passed),
  example_00 and example_02, and both `python -m xicsrt` and `--mp` CLI paths.

Pre-existing defects found during implementation, NOT fixed (out of scope,
reported only):
- The `xics_jax` import perturbs the global RNG stream, so `random_seed` does
  not guarantee reproducibility. Tracked as F009; see also Finding 9 in the
  plan.
- `examples/example_01/example_01.py` is broken on baseline: it requests
  `XicsrtOpticCrystalSpherical`, but the class is `XicsrtOpticSphericalCrystal`.

Goal: enable ~1e9 generated / ~1e6 detected ray runs on the Princeton Stellar
cluster (768 GB, 96 cores), scaling up from a working 5.295e7 / 1.272e4 run.
Originally motivated by a suspicion that `combine_raytrace` was the
bottleneck; that hypothesis was disproved by measurement (see below). The
remaining work stands on its own merits: one correctness bug, one real memory
win, and two speedups.

Findings (all measured, details and tables in the plan):
- `combine_raytrace` is not the bottleneck. ~195 MB combined history at 1e6
  detected rays, and with `keep_history=False` its history block is skipped
  entirely by the guard at `xicsrt_raytrace.py:359` (same for
  `_sort_raytrace` at line 254).
- Memory is not the constraint. Measured 195 B/ray of history (65 B/ray x 3
  elements) + ~110 B/ray temporaries + a ~2x transient => ~26 GB at 12
  workers x 4.4e6 rays/iter with history on; ~30x headroom at 768 GB.
- `np.einsum` does NOT use threaded BLAS: measured zero speedup from 1 to 8
  threads on `'ij,ij->i'` and `'ij,ijk->ik'` (the entire hot path), while a
  2000^3 gemm scaled 62 -> 33.5 ms. A 12-runs x 8-threads layout therefore
  uses ~12 of 96 cores. Retained by user decision; documented, not changed.
- `_sort_raytrace` shuffles millions of indices to keep ~833 lost rays:
  0.0734 s vs 0.0035 s for `Generator.choice(replace=False)` (21x), with
  uniformity verified (chi2=9961.9, dof=9970, p=0.52).
- `np.random.shuffle` perturbs the *global* RNG stream in an N-dependent way;
  a dedicated `np.random.Generator` avoids this (verified), keeping found
  rays bit-identical while only lost-ray selection changes.

Pre-existing defects found:
- `combine_raytrace` silently drops the `weight` array from combined
  histories (`RayArray.zeros` omits it; the copy loop iterates output keys).
  This is the "known benign quirk" at `devel/jaxrt_sync.md:164-166`; fixing
  it resolves that entry for both engines.
- Worker profiling is invisible: `profiler_results` is a per-process global
  that is never returned, so `profiler.report()` shows only parent timings.
- `mp: gathering` (`xicsrt_multiprocessing.py:58`) times `.get()` on
  already-completed results and measures nothing; the real cost is inside the
  untimed `pool.join()`.
- `raytrace_single` holds ~2x history (previous iteration stays live while
  the next allocates).

Planned work: build a two-tier regression harness first (Tier A bit-identical
for found rays / images / meta over `number_of_iter=3`; Tier B statistical
for lost rays), then add multiprocessing + worker-profiler instrumentation,
then four fixes: `_sort_raytrace` sampling, `make_image` vectorization,
history release in `raytrace_single`, and the `combine_raytrace` rewrite.

Out of scope: per-element history compaction (~3x further ceiling gain, needs
its own feature entry); worker/thread layout changes.

Note: fix 7 changes the output dict structure (histories gain `weight`),
warranting a minor version bump at release time.

---

## F005 - Allow XicsrtPlasmaVmec to load either VMEC or saved DESC equilibria
Started: 2026-07-26
Status: Done (2026-07-26)

Implementation: `XicsrtPlasmaVmec.initialize_vmec` (in
`xicsrt_contrib/xicsrt_contrib/sources/_XicsrtPlasmaVmec.py`) now picks the
loader from the `wout_file` extension: `.nc` -> `VMECIO.load`, `.h5` ->
`desc.io.load`; any other extension raises `ValueError`. No other methods
needed changes since they only call `self.eq.map_coordinates(...)`, which is
identical for both equilibrium types. Verified with a standalone script
loading both `wout.nc` and `wout_desc_solved.h5` and checking
flux/Cartesian round-trip error (~1e-6 m or better for both); also ran the
full `pytest tests/` suite (36 passed) and `examples/example_00/example_00.py`
end-to-end. Updated the SULI Part 5 logbook notebook to point
`wout_file` at `/u/npablant/data/w7x/vmec/w7x_ref_172/wout_desc_solved.h5`.

Request: update the W7-X SULI Part 5 logbook notebook to use a pre-solved DESC
equilibrium file (`wout_desc_solved.h5`, produced by a separate converter
notebook via `VMECIO.load` + `solve_continuation_automatic` + `eq.save`)
instead of the original VMEC `wout.nc`. `desc.io.load` on a native DESC `.h5`
skips the VMEC spectral re-fit and is much faster to load than `VMECIO.load`
on a `wout.nc` (see the "Desc Coordinate Transform Performance Minimal
Example" notebooks). `XicsrtPlasmaVmec.initialize_vmec` currently only
supports `VMECIO.load` (`.nc`), so both VMEC and DESC equilibrium inputs
need to be valid.

Decision (user-approved plan): keep the `wout_file` config key name and the
`XicsrtPlasmaVmec` class name unchanged; detect the equilibrium format from
the file extension (`.nc` -> `VMECIO.load`, `.h5` -> `desc.io.load`) inside
`initialize_vmec`. No jaxrt sync needed (plasma sources are out of scope per
`devel/jaxrt_sync.md`). Scope limited to the canonical
`xicsrt_contrib/xicsrt_contrib/sources/_XicsrtPlasmaVmec.py` (the separate
`suli/suli2026_alston/xicsrt_contrib` clone is left untouched).

## F004 - Exploratory JAX-accelerated tools_jax for the numpy OO engine
Started: 2026-07-21
Status: Complete (2026-07-25). No clear advantage for CPU-based computation.
Update 2026-07-30 (F010, 1a): the opt-in JAX CDF sampler was made obsolete by
direct Voigt sampling (exact Normal+Cauchy mixture, no CDF tables at all).
`xicsrt_voigt_multi_jax.py`, `xicsrt_faddeeva_jax.py`,
`tests/test_voigt_multi_jax.py`, and the `_USE_JAX_VOIGT_MULTI` toggle were
deleted.
Manual benchmarking under realistic `raytrace_multiprocessing` usage (the
actual way production W7-X jobs are run) showed no net speedup from the
opt-in JAX path once multiprocessing already saturates the CPU; see "Final
disposition: manual multiprocessing benchmark" below. The single-process-only
benchmark from the 2026-07-21 session (below) is superseded by this finding.
The opt-in code (`_USE_JAX_VOIGT_MULTI`, default `False`) is left in place,
disabled by default, in case a future GPU target changes the conclusion.

Final disposition: manual multiprocessing benchmark (2026-07-25):
- The 2026-07-21 sessions below only benchmarked single-process
  (`raytrace`/`raytrace_single`) runs, where the JAX path showed a real
  ~1.3-2x speedup on the 184-line Ar16+ workload. That is not how production
  jobs are actually run.
- Manual benchmarking (M1 MacBook, CPU only) comparing plain numpy under
  `raytrace_multiprocessing` (10 runs, all cores) against the JAX opt-in path
  run with an equivalent multiprocess/iteration split found the two
  statistically indistinguishable (~5m33s vs ~5m37s for matched configs,
  efficiency/ray counts consistent within statistics). Once multiprocessing
  already parallelizes across all CPU cores, the single-process JAX
  speedup is not additive and provides no net benefit.
- Follow-up tuning of the gridsize-bucketing strategy (fixed gridsize 1024,
  fixed 4096, and a smaller-bucket/lower-max-gridsize variant) was also
  tried and gave no further improvement over the power-of-two/8192-max
  scheme already implemented; performance returned to baseline in each case.
- Conclusion: on CPU, for this workload and at this problem scale, there is
  no configuration of the JAX opt-in path that outperforms plain numpy once
  multiprocessing is used as it is in practice. The code is retained
  (disabled by default) rather than removed, since a GPU target (not tested
  here) remains a plausible future path per F003's Princeton Stellar A100
  reference; any future revisit should start from a GPU benchmark rather
  than further CPU tuning.

Second follow-up: revisited at production scale (2026-07-21):
- Prompted by: "I would like to start a new feature to try to accelerate
  plasma bundle generation with tools/xicsrt_voigt_multi... For this feature
  I want to accelerate the Object Oriented python+numpy code; not move fully
  to the jaxrt code... I want to be able to turn on and off jax acceleration
  (hard-coded changes such as commented in and out code)."
- The original abandonment (below) benchmarked a toy ~15-line spectrum.
  Re-benchmarking at the true production scale (184-line Ar16+ table, the
  actual `ar16_voigt` line count) gave the opposite conclusion: JIT dispatch
  overhead is no longer dominant once the per-call Faddeeva work is large
  enough, and a JAX path is faster than plain numpy for this workload
  (measured ~2x on `generate_wavelength` wall-clock, ~1.3x on full
  `raytrace`, in a real W7-X `ar16_voigt` run with `bundle_count` reduced to
  ~1000 for fast iteration).
- Two retracing hazards from the original attempt were both fixed
  differently this time:
  - `gridsize` (auto-computed per bundle from local sigma/gamma, so it
    varies continuously bundle-to-bundle): now rounded up to the next
    power-of-two "bucket" (floor 128) before being passed as a
    `static_argnames` argument to `jax.jit`, bounding the number of distinct
    compiled shapes across a whole run to a handful instead of ~1 per bundle.
  - `size` (the Poisson-derived ray count per bundle, also varying
    bundle-to-bundle): this is no longer a jit argument at all. The final
    uniform draw + inverse-CDF `numpy.interp` sampling step happens in plain
    numpy on the host (as it always did in the numpy engine), so `size`
    never touches the jit trace signature. This also means results stay
    reproducible via `numpy.random.seed` (no RNG deviation from the numpy
    engine, unlike the mirror-directory design considered in the original
    session).
- Implementation: two new modules, `xicsrt/tools/xicsrt_faddeeva_jax.py` and
  `xicsrt/tools/xicsrt_voigt_multi_jax.py`, API-compatible drop-ins for
  `xicsrt_faddeeva.py`/`xicsrt_voigt_multi.py` (not a `tools_jax/`
  subpackage this time; they live directly in `xicsrt/tools/` since `jax`
  via `xics_jax` is already a mandatory import of
  `_XicsrtSourceGeneric.py`). A single hard-coded module-level flag,
  `_USE_JAX_VOIGT_MULTI` in `_XicsrtSourceGeneric.py` (default `False`),
  switches `random_wavelength_multi_voigt` and `random_wavelength_ar16_voigt`
  between the two backends; there is no config option and no runtime
  detection/fallback, per the "readability first, hard-coded toggle" request.
- Also fixed in the same session (pre-existing, unrelated bug):
  `random_wavelength_ar16_voigt` was reading `self.param.get("gridsize")` /
  `self.param.get("cutoff", 1e-4)` instead of the actual config keys
  `self.param['multi_gridsize']` / `self.param['multi_cutoff']`, silently
  ignoring those two config options for `ar16_voigt` (they happened to
  default correctly by luck: `None`/`1e-4`). Now reads the correct keys,
  matching `random_wavelength_multi_voigt`.
- Tests: `tests/test_voigt_multi_jax.py` (`pytest.importorskip('jax')`),
  covering Faddeeva/Voigt numeric agreement with the plain-numpy kernel,
  CDF/PDF agreement at a matching gridsize, gridsize-bucketing correctness
  (including that two different requested gridsizes landing in the same
  bucket give bit-identical CDF tables), sampling histogram-vs-PDF
  statistics, and `numpy.random.seed` reproducibility. Full suite (36 tests)
  and `examples/example_00/example_00.py` pass with the (default) flag left
  `False`.
- Caveat (documented in the new module's docstring): this is a workload-
  dependent opt-in, not a strict upgrade. It is only faster for large line
  lists (validated at 184 lines); for small hand-specified `multi_voigt`
  line lists (the original F004 toy-benchmark regime) plain numpy is still
  expected to be faster, matching the original conclusion below.

Original session, root cause and benchmarks (2026-07-21 follow-up session):
- Root cause of the hang: the (uncommitted) edit wiring
  `xicsrt/sources/_XicsrtSourceGeneric.py` to import `xicsrt.tools_jax`
  instead of `xicsrt.tools` exposed the plain numpy engine's per-bundle
  Python loop (`XicsrtPlasmaGeneric.create_sources`, one
  `multi_voigt_random`/`ar16_voigt` call per bundle) to `jax.jit`. Each
  bundle has its own temperature-dependent `gridsize` (auto-computed in
  `_prepare_bounds` when `gridsize=None`) and its own Poisson-derived
  `size`; both are part of the `jax.jit(static_argnames=('N','size'))`
  trace/shape signature, so XLA recompiled on nearly every bundle
  (measured ~150-900ms/compile in isolation). For a realistic plasma
  (`bundle_count ~ 1e4`, see `xicsrt_w7x_npablant.get_config()`) this
  compounds to tens of minutes to hours. `tests/test_voigt_jax.py` never
  hit this because it always called with fixed/repeated shapes (jit cache
  hit), which is why "pytest passes but the notebook hangs".
- Benchmark (CPU, ~15-line Ar16+ spectrum, `bundle_count=1e4`, ~8-30 rays/
  bundle, matching the real w7x_npablant production config):
  - Plain numpy (`xicsrt.tools.xicsrt_voigt_multi`, pre-jax behavior):
    ~5.3s for all 10,000 bundles.
  - Per-bundle jax with `gridsize`/`size` rounded to fixed power-of-two
    "buckets" (fixes the retracing/hang): ~7.5s -- correctness fixed, but
    still *slower* than plain numpy.
  - Fully batched `jax.vmap` over all 10,000 bundles in a single jit call
    (fixed common grid, padded per-bundle ray capacity): ~12.6s warm
    (~20s incl. compile) -- also slower than plain numpy.
  - Conclusion: for this workload (many small independent per-bundle
    draws, ~15 lines, single-digit-to-tens of rays/bundle) on CPU, JAX
    dispatch/compile overhead is not recovered by the small amount of
    vectorizable math, regardless of whether the shape/retracing bug is
    fixed. There is currently no `tools_jax`-based approach that beats
    plain numpy for `XicsrtPlasmaGeneric.create_sources` at this problem
    scale on CPU.
  - Open question for a future session: whether a GPU target (e.g.
    Princeton Stellar A100s, per F003) and/or a much larger line list/
    grid/ray-count per bundle shifts this crossover point in JAX's favor.
    Not evaluated in this session.
- Disposition: `xicsrt/tools_jax/` and `tests/test_voigt_jax.py` were
  deleted (not merely left unwired) since the benchmarks show no viable
  path to a speedup for the actual production use case as currently
  architected; reintroducing this approach should start from a fresh
  GPU/large-scale benchmark rather than re-adding the CPU-oriented
  mirror-directory code as-is.

Original session notes (2026-07-21, historical; describes code that has
since been deleted -- see "Disposition" above):

Goal: Let the ordinary object-oriented numpy+scipy code opt in to targeted
JAX acceleration of `tools/xicsrt_voigt_multi.py` (and
`xicsrt_voigt.py`/`xicsrt_faddeeva.py`) for exploratory use, without touching
the numpy engine or the separate `xicsrt.jaxrt` raytracing engine. Follows
up on F001 (~94% of multi_voigt runtime is in the Faddeeva/wofz kernel) and
is unrelated to F003/jaxrt (not tracked in `devel/jaxrt_sync.md`).

Implementation (2026-07-21):
- Chose the "mirror directory" option (over adding an `xp=numpy` array-
  namespace parameter to the existing `xicsrt/tools/*.py` files) to keep
  zero blast radius on the numpy engine, at the cost of a duplicated (but
  small) copy of the Weideman/voigt math.
- New subpackage `xicsrt/tools_jax/` (not added to `setup.py`/
  `extras_require`; requires `jax`/`jaxlib` installed manually; importing it
  without jax raises a plain `ModuleNotFoundError`):
  - `xicsrt_faddeeva.py`: jax/jit port of `wofz_weideman`/`voigt_profile`
    (`_weideman_coeffs` setup reused unchanged from `xicsrt.tools`, plain
    numpy, `lru_cache`d).
  - `xicsrt_voigt_multi.py` / `xicsrt_voigt.py`: drop-in API-compatible
    ports of `multi_voigt`/`multi_voigt_cdf_tab`/`multi_voigt_random` and
    `voigt`/`voigt_cdf_tab`/`voigt_random`; return plain `numpy.ndarray`.
- Jit boundary: data-dependent grid/domain sizing (min/max over line
  widths, `warnings.warn`, validity checks) stays host-side Python/numpy,
  duplicated from the numpy engine (cannot be traced under `jax.jit`); the
  grid evaluation, cumsum/normalize, and (for the `_random` functions) the
  random draw + inverse-CDF interpolation are fused into one `jax.jit` call.
- RNG deviation (deliberate, user-approved): the `_random` functions use an
  internal `jax.random.PRNGKey` auto-seeded from OS entropy (`secrets.
  randbits`) each call, not the global `numpy.random` state, so they are
  NOT reproducible via `numpy.random.seed` unlike the numpy-engine versions.
  Documented in each function's docstring and covered by a dedicated test.
- Tests: `tests/test_voigt_jax.py` (`pytest.importorskip('jax')`), checking
  numeric agreement with `xicsrt.tools` for deterministic functions and
  histogram-vs-PDF statistics for the random samplers. Full suite (30
  tests) and `examples/example_00/example_00.py` verified passing.

Key prompt from user: "I would like to use jax to accelerate the code in
xicsrt_voigt_multi.multi_voigt_random when running a 'normal' object
oriented (OO) python+numpy code ... I just want a way to temporarily test
out some targeted jax accelerations." Later confirmed: mirror-directory
approach (not xp-injection), jax.random for sampling (auto-seeded from OS
entropy, not a `key` parameter), jit fused across CDF-build + sampling,
`xicsrt/tools_jax/` as an importable subpackage, port `xicsrt_voigt.py` too,
no benchmark script, no `setup.py` changes.

---

## F003 - JAX-accelerated raytracing engine (xicsrt.jaxrt)
Started: 2026-07-17
Status: In Progress (2026-07-17: phase 1 implemented, pending user verification)

Goal: A parallel JAX-based engine in a new subpackage `xicsrt/jaxrt/`, enabling
jit/vmap acceleration on CPU now and GPU (Princeton Stellar cluster, A100s)
later. The numpy engine remains completely untouched; `jax` is an optional
dependency. Priorities, in order: (1) readability for doctorate-level physics
researchers without CS background, (2) exact photon statistics at every
element (core XICSRT tenet, no exceptions), (3) acceleration.

Full approved plan: devel/plan_jaxrt.md

Key decisions (approved 2026-07-17):
- Parallel subpackage `xicsrt/jaxrt/`, pure functions + config dicts (no
  mutable mixin classes); same JSON configs as the numpy engine.
- float64 everywhere (jax x64 mode enabled on import).
- Explicit jax.random key threading; statistically equivalent to the numpy
  engine, not bit-identical (approved).
- Poisson ray counts via capacity + mask: exact Poisson draw N, fixed
  capacity = mean + 10 sigma, rays beyond N masked from birth; error on
  overflow (~1e-23 probability). Photon statistics exactly preserved.
- Phase 1 scope: Generic/Directed/Focused sources; Plane/Sphere/Cylinder/
  Torus shapes; None/Mirror/Crystal/MosaicCrystal interactions; rocking
  curves step/gaussian/file. Sequential runs, single process, single device.
- Deferred: plasma sources, mesh optics, filters, multi-GPU pmap/sharding,
  any multiprocessing interplay (intentionally excluded).

---

## F002 - Kent / FB8 directional distribution for angular spread
Started: 2026-07-17
Status: Pending

Goal: Implement the Kent (FB5) and/or FB8 family of directional distributions
as a proper angular-spread option in `xicsrt/tools/xicsrt_spread.py`, replacing
the small-angle Gaussian approximation for anisotropic emission. This should
include at minimum a sampler (analogous to existing `vector_dist_*` functions)
and a corresponding `solid_angle` calculation.

The FB8 family is the mathematically correct set of distributions for
anisotropic/elliptical angular emission on the unit sphere:
FB8 (8-param) ⊃ FB6 ⊃ FB5 / Kent (5-param) ⊃ von Mises–Fisher.
The Kent distribution is the natural spherical analogue of the bivariate normal.

Reference implementation (`fb8` v1.2.2, MIT, Tianlu Yuan):
- https://pypi.org/project/fb8/
- https://github.com/tianluyuan/sphere

Findings / notes (from code review, 2026-07-17):
- Only sampling and pdf evaluation are likely needed (not MLE, gradient, or
  contour functionality from the reference package).
- The reference `rvs` implementation is rejection-based and may be inefficient
  for high concentration parameter κ (i.e. narrow beams), which is a common
  XICSRT use case.
- The reference package emits noisy warnings/logging that conflict with the
  XICSRT `mirlogging` conventions and uses its own RNG separate from the rest
  of XICSRT.
- Open question: whether Kent (FB5, elliptical symmetric) alone is sufficient
  or whether the full FB8 asymmetry is needed.

---

## F001 - Performance enhancement for tools/xicsrt_voigt_multi.py
Started: 2026-07-17
Status: In Progress (2026-07-17: implemented jax-friendly Weideman kernel)

Goal: Faster multi_voigt / multi_voigt_cdf_tab / multi_voigt_random.

Implementation (2026-07-17):
- Chose Option D: replace scipy.special.wofz (non-jax-able C routine) with a
  pure-numpy Weideman (1994) rational Faddeeva approximation, default N=16
  (tunable keyword, not user-facing). N=16 max abs err ~3e-7 vs wofz.
- New shared kernel xicsrt/tools/xicsrt_faddeeva.py (voigt_profile, wofz_weideman,
  cached _weideman_coeffs). Pure array arithmetic -> future jax swap is trivial.
- xicsrt_voigt.voigt now wraps voigt_profile; original wofz version retained
  (unused) as voigt_wofz per request.
- xicsrt_voigt_multi.multi_voigt vectorized (broadcast over line axis, no
  per-line Python loop); N passthrough added to cdf_tab / random.
- Added tests/ (pytest): accuracy vs wofz (N=16/24/32), voigt==voigt_wofz,
  multi==sum-of-singles, CDF properties, sampler histogram-vs-PDF. 11 pass.
- setup.py: extras_require={'test': ['pytest']}.
- Modest CPU speedup today (~1.1-1.7x, grows with line count); main value is
  the jax-ready, differentiable, vectorized kernel.

Findings:
- ~94% of runtime is inside scipy.special.wofz. Cost scales with
  n_lines x n_grid evaluations; everything else (cumsum, bin widths,
  linspace, interp) is negligible.
- Rewriting the per-line Python loop as a 2D numpy broadcast gives NO
  meaningful gain (1.0-1.2x, only at tiny sizes) — wofz still runs on the
  same number of points; loop overhead is trivial. Not worth the added
  memory/complexity.
- Windowing (evaluate each line only on its ±cutoff grid slice via
  np.searchsorted) gives a real ~1.5x (~33%) speedup that scales with
  line count (5→300 lines: 1.49x→1.56x). Tradeoff: small tail-truncation
  error (tunable via window width). Deferred: not worth the accuracy
  tradeoff / added complexity.
- Micro-opts (constant bin width -> scalar dx, hoist sqrt(2)/sqrt(2pi))
  are all <5%; skip.
- Only path to a large win is replacing wofz with a pseudo-Voigt
  approximation — faster but changes accuracy; not pursued.

Decision: No code change at this time.
