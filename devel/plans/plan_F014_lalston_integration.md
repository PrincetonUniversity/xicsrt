# Plan F014-F019: integrate the devel_lalston SULI work into the optimized codebase

Status: approved by user 2026-07-31, not yet started.
Tracking: F014-F019 in devel/features_request.md.

This plan is self-contained: a fresh agent session should be able to implement
it starting from this file alone. Instruct the agent:
"Implement the lalston integration per devel/plans/plan_F014_lalston_integration.md,
starting with Phase 0."

Note: this plan was produced with AI assistance using Claude (Opus 5).

## Objective

Merge a colleague's (SULI 2026, L. Alston) W7-X XICS work into the optimized
`devel_npablant_accel` branch, so that the equivalent of these two notebooks
runs on the current codebase:

- `suli/suli2026_alston/xics_ml_pipeline/generate_training_configs.ipynb`
- `suli/suli2026_alston/xics_ml_pipeline/generate_training_images.ipynb`

The colleague worked from a pre-optimization commit, and several of their
features were re-implemented differently (and more aggressively) on the accel
branch. The user does not need the `devel_lalston` branches preserved as
history and does not want a real git merge; the goal is to identify the
meaningful changes and get them working on the current code.

## Repository map

Four repos are involved. The first three are on branch `devel_npablant_accel`.

- `xicsrt`            - core engine
- `xicsrt_contrib`    - `XicsrtPlasmaVmec` (DESC geometry)
- `xicsrt_analysis`   - `w7x_npablant` package (W7-X model + production driver)
- `suli/suli2026_alston/xics_ml_pipeline` - the colleague's own repo
  (github `lalston07/xics_ml_pipeline`, branch `main`). NOT previously known;
  this is where the two target notebooks actually live.

## Survey results (already done, do not repeat)

### The devel_lalston branches contain almost nothing

| Repo | devel_lalston vs devel_npablant_accel | Verdict |
|---|---|---|
| xicsrt | local branch is already an ANCESTOR of accel; remote has 1 new commit `3ba693d` | one change to evaluate |
| xicsrt_contrib | 90 lines, all of it REVERTING accel work (AI disclaimer, DESC `.h5` support, `_pad_masked`, equilibrium caching) | take nothing |
| xicsrt_analysis | `.gitignore` plus reverting LDRD-162 / `w7x_npablant` work | take nothing |

A `git merge` is the wrong tool: it would resurrect deleted code and revert
optimizations. The colleague's DESC coordinate-transform work was already
merged by the user in `xicsrt_contrib` commit `340961d`.

There is also a stale clone at `suli/suli2026_alston/xicsrt_contrib` that
appears to have 5 unpushed commits. `git cherry` confirms all 5 are
patch-equivalent to commits already merged. It is safe to delete.

### Blocking discovery: xicsrt_w7x_lalston does not exist

Both target notebooks do `from w7x_npablant import xicsrt_w7x_lalston`.
That file exists in NO repo, branch, or commit available locally (searched
all three repos across all refs, plus the whole mirproject tree). It exists
only on the colleague's Windows machine. Both notebooks are unrunnable as
written, independent of any merge. This plan drops it and extends the
existing `xicsrt_w7x_npablant` instead.

### The real gap: the spline bridge was never built on either side

`xicsrt/tools/xicsrt_spline.py` is already in the tree (commits `7c3bc21`,
`f8e0820`) and is a bug-fixed superset of the colleague's local copy. But it
is completely orphaned: zero importers, and no `profile_*` config option
exists anywhere in any repo. Correspondingly
`xicsrt_analysis/w7x_npablant/production/xicsrt_train_task.py:83`
`update_config_for_image` is an empty `return config` stub. These are the two
ends of the same missing bridge. Building it is the actual work.

### The one lalston commit worth evaluating: 3ba693d

Adjusts the wavelength sample count by the w-line intensity ratio. It does
not port directly (F010 deleted `random_wavelength_ar16_voigt` and moved the
Ar16+ physics to the per-bundle `get_line_parameters` hook), and its formula
is wrong. See Phase 4.

## Phase 0 - git reconciliation

- Delete ONLY the stale clone `suli/suli2026_alston/xicsrt_contrib`.
- KEEP every `devel_lalston` branch, local and remote, in all three repos.
  The user explicitly wants these left alone for now; deleting them is a
  deliberate unfinished step to be completed in a later session.
- Port the two comment typo fixes from `3ba693d`
  (`sinrce`->`since`, `Kev`->`keV`) into
  `xicsrt/sources/_XicsrtSourceGeneric.py`. Note the surrounding code has
  moved; apply the fixes to the equivalent current comments if present.
- Do NOT run `git merge` in any repo.

## Phase 1 - fix xicsrt/tools/xicsrt_spline.py

This module must be corrected before anything can consume it.

### 1a. Units and ranges (issue I-1, a latent 1000x bug)

The temperature generators currently emit keV (0.2-10.0) into a pipeline that
requires eV everywhere. See the unit audit in F018 below.

Emissivity and temperature are structurally DIFFERENT and must be treated
differently. This is important and was initially got wrong during planning:

- Emissivity is normalized BY DESIGN. `generate_random_emissivity:121`
  hardcodes `y_peak = 1.0`; only the profile SHAPE varies and the absolute
  scale comes from `emissivity_scale`. Leave this alone.
- Temperature is NOT normalized. `generate_random_temp:227` draws
  `core_temp = rng.uniform(y_min, y_max)`, i.e. the peak value is itself a
  randomized physical parameter. Normalizing temperature to a peak of 1.0
  would give every training profile the same core temperature and destroy
  the single most important ML label. DO NOT normalize temperature.

Changes:

- `generate_random_temp`: `y_min` / `y_max` in eV, documented as eV.
- `generate_random_ion_temp`: defaults `y_min=200.0, y_max=5000.0` (eV).
- `generate_random_electron_temp`: defaults `y_min=200.0, y_max=10000.0` (eV).
- Both remain caller-overridable; the range is a user knob, not a constant.
- `generate_random_perpendicular_velocity` and
  `generate_random_parallel_velocity`: replace the hardcoded, undocumented
  `rng.uniform(0.0, 20.0)` (`:334-337`, `:395`) with explicit `y_min`/`y_max`
  parameters in m/s, default +/-20e3, matching xicsrt's SI velocity
  convention.
- The consuming plasma class therefore sets `temperature_scale`,
  `temperature_e_scale` and `velocity_scale` to 1.0, since the splines return
  absolute physical units.

### 1b. Other fixes in the same file

- Remove `import plotly.graph_objects as go` (line 3). It is unused, and
  plotly is NOT in `setup.py` `install_requires` (`numpy, scipy, pillow,
  h5py`), so importing this module raises ImportError on a clean install,
  including on Stellar.
- Fix the hardcoded 5-element `x_free` / `y_free` / `deriv_zero` literals.
  Verified: `generate_random_temp(n_knots=7)` returns 7 knots but length-5
  mask arrays. Either derive them from `n_knots` or validate.
- Every generator docstring claims it returns `spline : callable
  interpolator`; none of them do. They build a spline locally and discard it.
  Fix the docstrings and add a `spline_from_profile(profile)` helper so the
  knots->callable logic lives in exactly one place.
- Add NumPy-style docstrings and the AI disclaimer per AGENTS.md.
- Add unit tests.

Known limitation, log but do NOT change:
`generate_random_perpendicular_velocity` hardcodes
`y_knots = [0, ion_root, 0, electron_root, 0]` and so can only produce
profiles with exactly one ion root and one electron root; it cannot produce
pure-ion-root or pure-electron-root cases. The file's own comment at `:299`
already notes this. Widening it is a physics-coverage decision for the user.

## Phase 2 - get_velocity signature change and unit fixes

### 2a. The hook contract

```python
def get_emissivity(self, rho):        # unchanged
def get_temperature(self, rho):       # unchanged
def get_temperature_e(self, rho):     # unchanged
def get_velocity(self, point_flx):    # CHANGED
```

Only `get_velocity` changes. The hooks are NOT required to share a signature:
nothing dispatches over them generically, each is called by name inside
`bundle_generate`, so the only constraint is per-hook agreement between base,
overrides and call sites.

The asymmetry is physically justified. The scalar profiles are flux-surface
functions and `rho` is genuinely all they need. Velocity is a VECTOR: even
under the flux-surface-function assumption for v_perp and v_par, converting
those magnitudes into a Cartesian vector requires the local B direction and
the grad(rho) direction, which require theta and zeta.

Do NOT add a `mask` argument. A hook that used it to compress its input would
reintroduce the variable-length shape that the padding exists to prevent.

Convention, to be stated once in `XicsrtPlasmaGeneric` and referenced
elsewhere: hooks receive full-length `bundle_count` arrays and return
full-length arrays; `bundle_generate` applies the mask to the result.
`point_flx` column 0 is rho, with NaN marking points outside the LCFS.

### 2b. Why this is necessary (measured, not cosmetic)

DESC's `eq.compute` retraces whenever its input shape changes: measured
1.8 s vs 0.11 s at 10,000 bundles. Today `bundle_generate` passes
`rho = rho_full[m]`, a mask-compressed array whose length changes every
iteration as different bundles fall outside the LCFS. `_XicsrtPlasmaVmec`
already solved this for `map_coordinates` with `_pad_masked`, but then
re-compresses with `[m]` before calling the hooks. Phase 5 needs DESC inside
a hook, so the fixed shape must extend through the hook boundary.

Note that `rho` is redundant with `point_flx[:, 0]`, which is why the hook
takes only `point_flx` and not both.

### 2c. Files to change

| File | Change |
|---|---|
| `xicsrt/sources/_XicsrtPlasmaGeneric.py:251` | signature; also I-2 and I-3 below |
| `xicsrt_contrib/.../_XicsrtPlasmaVmec.py:181-228` | pass full-length arrays, apply `[m]` to results |
| `xicsrt/sources/_XicsrtPlasmaToroidal.py:36-80` | same, plus the F019 rho fix |
| `xicsrt_analysis/w7x_npablant/sources/_XicsrtPlasmaW7x.py` (new) | DESC velocity implementation |

Also in this phase:

- I-2: `_XicsrtPlasmaGeneric.py:197-198` initializes `temperature` and
  `temperature_e` to `np.ones` (= 1.0 eV) although the documented default is
  0.0. Change to `np.zeros`.
- I-3: add `temperature_e_scale` where a class has `temperature_scale` but
  not the electron counterpart.
- Add a Doppler-sigma regression test pinning the eV convention on BOTH the
  numpy and jaxrt engines. There is currently zero test coverage on this
  convention despite four duplicated copies of the same formula
  (`_XicsrtSourceGeneric.py:372`, `:411`, `_XicsrtPlasmaGeneric.py:511`,
  `jaxrt/tools/_wavelength.py:102`). Suggested anchor: temperature=1000.0 eV,
  mass_number=39.948, wavelength=3.9492 A gives sigma = 6.474e-4 A.

Verified NOT affected (they never call the hooks, they write `bundle_input`
directly): `_XicsrtPlasmaCubic.py`, `_XicsrtPlasmaCylindrical.py`,
`_XicsrtPlasmaBundleSource.py`. Also not affected:
`_XicsrtPlasmaToroidalDatafile.py` and `_XicsrtPlasmaVmecDatafile.py`, which
only override the scalar hooks.

Deliberately out of scope: `_XicsrtPlasmaImas.py` (user: "dead code, I don't
care at all"), left on the old signature; and `xicsrt_iter`, a fourth repo
whose `get_velocity(self, rho, veloc_interp)` has already diverged.

No test breakage expected: all 55 tests exercise these paths through
`raytrace`, none call the hooks directly.

## Phase 3 - class hierarchy

Delete the existing dead `xicsrt_analysis/w7x_npablant/sources/_XicsrtPlasmaW7x.py`.
It contains `W7xPlasma`, which: cannot be loaded by the dispatcher (the class
name does not match the `Xicsrt<Name>` convention for its filename), imports
the nonexistent `xicsrt.sources.xicsrt_vmec`, uses the old camelCase API
(`getEmissivity`, `getTemperature`), and has zero references anywhere.
Reuse the filename for the new base class.

```
XicsrtPlasmaVmec (xicsrt_contrib)      - DESC geometry, no profiles
  |
  +- XicsrtPlasmaW7x                   - NEW base: W7-X geometry, Ar16+ line
     |                                   model, w-line normalization.
     |                                   NO PROFILES.
     +- XicsrtPlasmaW7xSimple          - polynomial profiles (existing)
     +- XicsrtPlasmaW7xProfile         - NEW: spline profiles, NO fallback
```

The user explicitly does not want a polynomial fallback in the spline class:
profiles must be class-specific with no default. `XicsrtPlasmaW7x` holds what
is genuinely shared (Ar16+ `get_line_parameters`, w-line normalization) and
defines no profiles at all.

`XicsrtPlasmaW7xProfile` details:

- Five options `profile_emissivity`, `profile_ion_temp`,
  `profile_electron_temp`, `profile_perpendicular_velocity`,
  `profile_parallel_velocity`.
- Default them to `None`, NOT `{}`. With a `{}` default,
  `strict_config_check` recurses into the user-supplied dict and rejects
  `x_knots` etc. as unrecognized options. See the same trap documented in
  `_XicsrtPlasmaBundleSource.py:105-110`.
- Build the splines once in `initialize()`, not per bundle-generate call.
- Propagate NaN for rho outside the LCFS (`extrapolate=False`).
  `_XicsrtPlasmaVmec.bundle_generate` relies on
  `m &= np.isfinite(bundle_input['temperature'])` to drop those bundles.
  Clamping or zeroing instead would silently create plasma outside the LCFS.
- Set `temperature_scale = temperature_e_scale = velocity_scale = 1.0`
  since the splines return absolute eV / m/s after Phase 1.

## Phase 4 - w-line emissivity normalization

Decision: `emissivity` for the W7-X Ar16+ model means the emissivity of the
'w' line ONLY, not the whole Ar16+ spectrum. Implement on the shared
`XicsrtPlasmaW7x` base class, with NO config option and no dedicated test
(user: "no backwards compatibility, clean code, human readability").

Mechanism: per bundle, multiply `bundle_input['emissivity']` by
`I_tot / I_w`, where both are evaluated from the Ar16+ line model at that
bundle's Te. This must be applied BEFORE the Poisson draw in
`XicsrtPlasmaGeneric.create_sources` (~line 365), because after that point
the ray count is already fixed. Since the factor is per-bundle
(Te-dependent), and plasma parameters are constant within a bundle but vary
between bundles, this is well defined.

Implementation note: override `bundle_generate` to evaluate the Ar16+ line
model ONCE per iteration, cache `(location, intensity, sigma, gamma)` on the
instance, apply the emissivity factor, and have `get_line_parameters` return
the cache. That keeps it to one atomic-model call per iteration rather than
two.

Photon statistics are preserved exactly: this scales the true expected number
of emitted photons before the Poisson draw. It is not a reweighting.

Verification already performed (do not need to repeat, but useful as a
regression reference): two bundles at Te = 1.0 and 4.0 keV have w-fractions
0.3633 and 0.6807. Scaling each bundle's draw count by `I_tot/I_w` and
counting rays that select the w line (widths zeroed so line selection is
exactly countable) gave 50,107 and 49,959 against a 50,000 target, i.e.
+0.48 sigma and -0.18 sigma. This works because
`multi_voigt_random_batched` normalizes each bundle row independently
(`cum /= cum[:, -1:]`).

The colleague's commit `3ba693d` used `size * (1 + I_w/I_notw)` =
`size * I_tot/I_notw`, which is NOT the same factor and differs by up to ~5x;
the two even cross over as a function of Te (at Te=0.5 keV: 1.20 vs 6.11;
at Te=4.0 keV: 3.13 vs 1.47). The user confirmed the intent is `I_tot/I_w`.
That commit also had a latent bug: it returned more wavelengths than `size`,
mismatching the origin/direction array lengths.

## Phase 5 - velocity via DESC

Replace the dead `if False:` stelltools code in
`_XicsrtPlasmaW7xSimple.get_velocity` (`:223-278`) with a DESC
implementation, on the new `XicsrtPlasmaW7x` base.

Confirmed equivalences (researched, high confidence):

| stelltools | DESC | notes |
|---|---|---|
| `gradrho_car_from_flx` | `compute('e^rho', basis='xyz')` | m^-1, outward, identical |
| `b_car_from_flx` | `compute('B', basis='xyz')` | Tesla, identical |
| `fsa_gradrho_from_s(s)` | `<\|grad(rho)\|>` | s = rho**2 |
| `fsa_modb_from_s(s)` | `<\|B\|>` | s = rho**2 |

Conversion hazard: stelltools flux coordinates use `s`, DESC uses
`rho = sqrt(s)`.

Two MANDATORY performance measures, both measured on a W7-X equilibrium:

1. Seed the iota profile: pass `data={'iota': eq.iota(rho)}` to
   `eq.compute`. Without it, DESC builds an internal override grid with one
   entry per UNIQUE rho, which at the production `bundle_count = 1e4`
   measured 90 s at 5,000 bundles and OOM-killed the process at 10,000.
   With the seed the output is bit-identical and takes 0.11 s warm.
   Guard on `eq.iota is not None` (true for `VMECIO.load` output).
2. Fixed-shape input, handled by the Phase 2 signature change. `eq.compute`
   retraces on shape change exactly like `map_coordinates`.

Other details: use `Grid(nodes, sort=False, jitable=False)`; `sort=True`
would reorder nodes and break the bundle correspondence; `jitable=True`
breaks `compute` outright. Build the `<|grad(rho)|>` / `<|B|>` FSA tables
once per equilibrium on a `LinearGrid` and `np.interp` them per bundle
(cached on the instance alongside `self.eq`), rather than requesting the FSA
quantities on the custom grid, which hits the same override-grid explosion.
All `eq.compute` outputs are jax arrays and need `np.asarray` before
in-place use, as at `_XicsrtPlasmaVmec.py:203`.

REQUIRED code comment: state clearly that treating perpendicular and
parallel velocity as flux-surface functions is an ASSUMPTION AND IS WRONG,
and that a better implementation must account for flow incompressibility and
Pfirsch-Schlueter flows. Do NOT implement the FSA-to-local-flow conversion;
that is substantial work, logged as F017.

## Phase 6 - logbook notebooks

Create two new notebooks in
`/u/npablant/code/notebooks/npablant-2019/logbook/`, continuing the existing
ML Training Set series (Part 1 exists but is an empty stub, so the series
continues cleanly):

- `2026-07-30 - W7-X XICS ML Training Set - SULI 2026 L. Alston, Part 2 - Generating config files`
- `2026-07-30 - W7-X XICS ML Training Set - SULI 2026 L. Alston, Part 3 - Generating training images`

These reimplement `generate_training_configs.ipynb` and
`generate_training_images.ipynb` against the current API. Required changes
from the colleague's originals:

- Drop `xicsrt_w7x_lalston`; use `xicsrt_w7x_npablant`.
- `wavelength_dist = 'multi_voigt'`, NOT `'ar16_voigt'` (that distribution
  was removed from public xicsrt by F010; the Ar16+ model now enters via the
  `get_line_parameters` hook).
- POSIX paths, not `C:\Users\...`.
- DESC `.h5` equilibrium
  (`/u/npablant/data/w7x/vmec/w7x_ref_172/wout_desc_solved.h5`).
- `raytrace_mp` for the image-generation stage.

The best template is the existing
`2026-07-30 - W7-X XICS Raytracing - SULI 2026 L. Alston, Part 7 - New acceleration updates.ipynb`,
which is already updated for F010.

Per AGENTS.md, notebooks must be fully output-stripped before staging.

## Phase 7 - wire update_config_for_image

Fill in the `return config` stub at
`xicsrt_analysis/w7x_npablant/production/xicsrt_train_task.py:83` so that it
generates the randomized spline profiles from the per-image seed and injects
them into the config. The notebook path and the SLURM path must share one
implementation. Without this, all 10,000 SLURM training images would use
identical plasma profiles and differ only by RNG seed.

## Phase 8 - verification

- `pytest tests/` in xicsrt. Baseline is 55 passed.
- One small-N end-to-end run of the Phase 6 notebooks.
- Statistical check that the detected w-line count scales as expected with
  the requested w-line emissivity.
- Confirm `devel/jaxrt_sync.md` handling. jaxrt has no plasma source
  (`jaxrt/_dispatch.py:56-60` supports only the three non-plasma sources),
  so this is expected to be a divergence-log note rather than a mirrored
  change, but CONFIRM rather than assume.

## Out of scope / decided against

- Deleting the `devel_lalston` branches (user wants them kept for now).
- Any real `git merge` of devel_lalston.
- `_XicsrtPlasmaImas` (dead code per user).
- `xicsrt_iter` (fourth repo, unrelated device).
- FSA-to-local flow conversion (F017).
- Widening perpendicular-velocity root coverage.
- A config option or test for the w-line normalization (user explicitly
  declined both).

## Order of execution

1. Phase 0 (git reconciliation; cheap, do first).
2. Phase 1 (spline fixes) with tests.
3. Phase 2 (hook signature, I-2, I-3, F019, eV regression test).
4. Phase 3 (class hierarchy).
5. Phase 4 (w-line normalization).
6. Phase 5 (DESC velocity).
7. Phase 6 (notebooks), Phase 7 (SLURM hook).
8. Phase 8 verification.
9. Session close: append to devel/devel_ai_log.txt; ask the user before
   marking any feature Done. Work spans three repos, so expect three
   separate commits; the commit-message approval gate in AGENTS.md applies
   to each.
