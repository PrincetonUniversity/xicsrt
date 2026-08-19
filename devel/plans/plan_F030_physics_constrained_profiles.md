# F030 - Physics-constrained spline profiles for W7-X ML training configs

Status: Done. Implementation below was built and tested 2026-08-18, then
reworked the same day so that `PhysicsConstraintOptions` toggles are
captured in `config['scenario']['physics_constraints']` (see "Rework:
config-driven physics constraints" below) instead of being passed only as
a Python function argument.

This file includes AI generated content using Claude (Sonnet 5).

## Goal

Add two optional, independently-toggleable, off-by-default physics
constraints to the randomized spline plasma profiles used to build W7-X
XICS ML training-set configs (`xicsrt_config_task.py` /
"Part 2b" notebook):

1. **Emissivity Te-cutoff.** The real Ar16+ line model
   (`XicsrtPlasmaW7x._full_spectrum_factor`) returns zero emissivity for
   `temperature_e <= te_min` (default 200 eV). Constrain the randomized
   emissivity profile's zero-point knot (`x_knots[-2]`) to land at or
   inside that same rho, so the label never claims nonzero emission in a
   region the physics model would have already zeroed.
2. **Ti <= Te.** By default Ti and Te are drawn fully independently
   (~25% of samples get Ti > Te; see F021). Optionally constrain Ti to
   never exceed Te: the core value is resampled below Te's core value, and
   from the first knot where Ti would exceed Te onward, Ti's knots are
   pinned to Te's spline value.

Depends on F029 (moving `xicsrt_spline.py` into `xicsrt_analysis` first).

## Scope

Entirely within `xicsrt_analysis`.

### New module: `w7x_npablant/xicsrt_spline_constrained.py`

- `DEFAULT_TE_MIN_EV = 200.0`, documented as needing to stay in sync with
  `XicsrtPlasmaW7x.default_config()['te_min']` (duplicated rather than
  imported: the real value lives on a DESC-backed plasma source class that
  is expensive to instantiate just to read one default).
- `find_te_cutoff_rho(te_profile, te_min)`: `scipy.optimize.brentq` on the
  Te spline; returns 0.0/1.0 directly for the profile-never-crosses edge
  cases.
- `generate_random_emissivity_constrained(...)`: duplicates
  `generate_random_emissivity`'s peak/y-value logic verbatim; replaces the
  x-knot construction. `zero_knot_range` is clipped to `[0, rho_cutoff]`
  (physics), then the lower bound is floored to leave room for the
  remaining interior knots at `min_spacing` (geometric feasibility) -
  deliberately with NO corresponding ceiling near rho=1, since forced
  physics placement takes priority over preserving a `min_spacing`-sized
  final gap. A single-point resulting range forces `x_knots[-2]` exactly
  and marks it not free; an empty range raises `ValueError`.
- `generate_random_temp_constrained(...)` /
  `generate_random_ion_temp_constrained(...)`: duplicates
  `generate_random_temp`'s decreasing-branch logic (the increasing branch
  is not ported - no call site needs it); core value drawn from
  `(y_min, min(y_max, te_core))`; after generating knots normally, clamps
  from the first knot where `Ti > Te` onward to Te's spline value at those
  same rho, marking them not free.
- `PhysicsConstraintOptions` frozen dataclass: `enable_emissivity_te_cutoff`,
  `te_min`, `emissivity_zero_knot_range`, `enable_ti_le_te` - all off/
  unconstrained by default.

### Orchestration: `xicsrt_w7x_npablant.py`

`generate_random_profiles(seed, physics_constraints=None)` generates
`profile_electron_temp` first (always), then conditionally routes
`profile_ion_temp`/`profile_emissivity` through the constrained generators
with `te_profile=profile_electron_temp` when the corresponding toggle is
set. `update_config_with_profiles` threads `physics_constraints` through.
Default (`None`, equivalent to `PhysicsConstraintOptions()`) reproduces
pre-F030 output bit-for-bit - verified directly (reordering electron-temp
generation first does not change any profile's content, since seeds are
independent per F021).

### CLI: `xicsrt_config_task.py`

New flags: `--enable-emissivity-te-cutoff`, `--te-min`,
`--emissivity-zero-knot-range LOW HIGH`, `--enable-ti-le-te`, assembled into
one `PhysicsConstraintOptions` passed through `build_training_set_configs`
-> `get_w7x_ml_config`.

### Tests

- `tests/test_xicsrt_spline_constrained.py` (17 tests): `find_te_cutoff_rho`
  correctness; forced placement via `(1.0, 1.0)`; default-range knot never
  exceeding the cutoff; the `min_spacing` feasibility floor (near-axis
  cutoff raises, near-edge cutoff is allowed); Ti(0) < Te(0) and
  Ti <= Te at every knot over many (Te seed x Ti seed) combinations;
  reproducibility; `PhysicsConstraintOptions` defaults/frozen-ness.
- `tests/test_xicsrt_w7x_npablant.py` (5 tests): the orchestration-level
  bit-identical-when-disabled regression, plus end-to-end checks of each
  constraint (and both together) through `generate_random_profiles`.
- Existing `tests/test_xicsrt_config_task.py` lambdas patched to accept the
  new `physics_constraints` keyword.

## Design notes / things found during implementation

- The `min_spacing` feasibility floor was NOT part of the original design:
  an early version only clipped `zero_knot_range` to the Te cutoff and drew
  `x_knots[-2]` directly, which raised inside `sample_interior_x` for ~47%
  of seeds when forcing `zero_knot_range=(1.0, 1.0)` with a Te cutoff close
  to rho=1 (no room left for `min_spacing`-separated interior knots below
  it). Fixed by flooring the draw range's lower bound at
  `(n_knots - 2) * min_spacing`, deliberately NOT capping the upper bound
  near rho=1 (physics placement wins over preserving a final gap there).
- The Ti<=Te clamp is an at-the-knots guarantee only: because knot values
  are pinned to Te exactly where clamped and interpolated (PCHIP/Hermite)
  elsewhere, the interpolated curve between two knots that straddle the
  clamp point is not proven to stay below the Te curve everywhere. This is
  documented in the function docstring as a deliberate approximation.

## Rework: config-driven physics constraints (2026-08-18)

The original design threaded `PhysicsConstraintOptions` only as a Python
function argument (`generate_random_profiles`/`update_config_with_profiles`
/`get_w7x_ml_config`/CLI), so the toggles never reached the saved
`xicsrt_config_NNNNNN.json`. Reworked so the toggles live in the config
itself, following the standard xicsrt `scenario` section convention
(`xicsrt/xicsrt_config.py::default_config()`):

- `xicsrt_w7x_npablant.get_config()` now populates
  `config['scenario']['physics_constraints']` with all four
  `PhysicsConstraintOptions` fields (all off by default,
  `emissivity_zero_knot_range` stored as a JSON-safe list).
- `update_config_with_profiles(config, seed)` dropped its
  `physics_constraints` parameter; it now builds a `PhysicsConstraintOptions`
  internally via the new private `_physics_constraints_from_config(config)`
  helper, reading from `config['scenario']['physics_constraints']`.
- `generate_random_profiles(seed, physics_constraints=None)` is unchanged:
  it stays a pure function taking an explicit `PhysicsConstraintOptions`,
  used directly by `update_config_with_profiles` and by tests.
- `xicsrt_config_task.py` (`get_w7x_ml_config`/`build_training_set_configs`)
  now takes `physics_constraints` as a plain dict of overrides, applied via
  `config['scenario']['physics_constraints'].update(...)` right after
  `get_config()`/`initialize()`. `main()` assembles the CLI flags into a
  plain dict instead of a `PhysicsConstraintOptions` instance.
- "Part 1" and "Part 2b" notebooks updated to demonstrate setting
  `config['scenario']['physics_constraints']` overrides.
- `verify_ray_calibration.py` required no change: it already called
  `update_config_with_profiles(config, seed)` with no constraints argument.

## Verification

- `xicsrt_analysis/w7x_npablant` `tests/`: 94 passed (89 pre-rework + 5 new
  covering `get_config()` scenario defaults, `_physics_constraints_from_config`,
  `update_config_with_profiles` reading scenario toggles, and the CLI
  building a plain dict).
- Manual bit-identical check: `generate_random_profiles(seed)` with no
  constraints matches direct `xicsrt_spline.generate_random_*` calls using
  `profile_seeds` children, for an arbitrary seed.
- CLI `--help` renders correctly with the four new flags.
