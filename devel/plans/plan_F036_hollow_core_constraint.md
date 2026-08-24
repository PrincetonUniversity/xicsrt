# Plan F036 - Hollow-core (off-axis-peaked) Ti/Te physics constraint

Status: Done (2026-08-25)
Repo: entirely in `xicsrt_analysis` (sibling repo); feature tracking lives
here in `xicsrt` per the F029/F030/F031 precedent.

## Files touched (all in `xicsrt_analysis/w7x_npablant/`)

- `xicsrt_spline_constrained.py`
- `xicsrt_w7x_npablant.py`
- `tests/test_xicsrt_spline_constrained.py`
- `tests/test_xicsrt_w7x_npablant.py`

No changes to `xicsrt_spline.py` (unconstrained generators untouched, per
project direction) or `production/xicsrt_config_task.py` (CLI flags for
`physics_constraints` toggles were removed in F033; these are
experiment-design choices set only in `user_config_update`).

## 1. `xicsrt_spline_constrained.py`

- New module-docstring section documenting the hollow-core constraint,
  alongside the existing emissivity Te-cutoff and Ti<=Te sections.
- New function `apply_hollow_core(profile)`: a pure post-processing step
  on an already-built profile dict (works regardless of which generator
  produced it, constrained or not).
  - `delta = y_knots[0] - y_knots[1]`
  - `y_knots[0] = max(0.0, y_knots[1] - delta)`
  - No other field (`x_knots`, `deriv_zero`, `x_free`, `y_free`) is
    modified.
  - No renormalization of the rest of the profile: the resulting maximum
    value is generally lower than the unhollowed profile's (explicit
    user decision).
- `PhysicsConstraintOptions` (frozen dataclass): add
  `enable_ti_hollow: bool = False`, `enable_te_hollow: bool = False`,
  with docstring entries following the existing `enable_*` naming
  convention.

## 2. `xicsrt_w7x_npablant.py`

- `get_config()`: add `enable_ti_hollow=False`, `enable_te_hollow=False`
  under `config['scenario']['physics_constraints']`.
- `_physics_constraints_from_config()`: read both new keys into the
  `PhysicsConstraintOptions(...)` construction.
- `generate_random_profiles()` dispatch order:
  1. Generate `profile_electron_temp`.
  2. If `enable_te_hollow`: apply `apply_hollow_core` to it -- BEFORE
     it is used as the `te_profile` reference for `enable_ti_le_te` /
     `enable_emiss_te_cutoff`, so those constraints see the
     already-hollowed Te profile.
  3. Generate `profile_ion_temp` (plain or Ti<=Te-constrained, using
     the possibly-hollowed Te).
  4. If `enable_ti_hollow`: apply `apply_hollow_core` to
     `profile_ion_temp`.
- Docstring of `generate_random_profiles` updated to describe the new
  options and the dispatch-order rationale.

## 3. Known interaction (not a bug, not fixed)

Hollowing can push the electron core temperature at or below the
ion-temperature generator's `y_min` floor (default 200 eV). This is the
same, already-documented infeasible-sampling `ValueError` raised by
`generate_random_ion_temp_constrained` for any low Te core (see
`test_ion_temp_constrained_raises_if_te_core_at_or_below_y_min`), not new
behavior introduced by this feature. Production code
(`xicsrt_config_task.py::build_training_set_configs`) already catches and
skips `ValueError` per sample. Measured over 200 seeds with
`enable_te_hollow=True, enable_ti_le_te=True`: ~20% raise.

## 4. Tests

- `test_xicsrt_spline_constrained.py`: `test_apply_hollow_core_formula`,
  `test_apply_hollow_core_clamped_at_zero`,
  `test_apply_hollow_core_no_renormalization_lowers_max`; extended
  `test_physics_constraint_options_defaults_are_off` for the two new
  fields.
- `test_xicsrt_w7x_npablant.py`:
  `test_generate_random_profiles_te_hollow_enabled`,
  `test_generate_random_profiles_ti_hollow_enabled`,
  `test_generate_random_profiles_te_hollow_seen_by_ti_le_te` (skips
  seeds that hit the known infeasible-sampling `ValueError`, asserting
  at least one seed succeeds); extended
  `test_get_config_scenario_physics_constraints_defaults_are_off` for
  the two new keys.

Verification: full `xicsrt_analysis` `w7x_npablant` pytest suite run via
a throwaway venv (`xicsrt`/`xicsrt_analysis` on `PYTHONPATH`): 99 passed
(2 pre-existing, unrelated collection errors from a missing `xarray`
dependency, per F020).
