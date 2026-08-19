# Plan F031 - Rename spline `min_spacing`->`min_dx`, add per-profile `min_dx`/`min_dy` config

Status: Done (2026-08-19)
Repo: entirely in `xicsrt_analysis` (sibling repo); feature tracking lives
here in `xicsrt` per the F029/F030 precedent.

## Files touched (all in `xicsrt_analysis/w7x_npablant/`)

- `xicsrt_spline.py`
- `xicsrt_spline_constrained.py`
- `xicsrt_w7x_npablant.py`
- `production/xicsrt_config_task.py`
- `tests/test_xicsrt_spline.py`
- `tests/test_xicsrt_spline_constrained.py`
- `tests/test_xicsrt_w7x_npablant.py`
- `tests/test_xicsrt_config_task.py`

Notebook `notebooks/.../Part 2b - Generating config files.ipynb`: no code
change needed. It already assigns directly into
`config['scenario']['physics_constraints'][...]`, so the new keys are usable
without any notebook edit once `get_config()` populates their defaults.

## 1. `xicsrt_spline.py`

- Add module constants `DEFAULT_MIN_DX = 0.05`, `DEFAULT_MIN_DY = 0.0`.
- `sample_interior_x`: rename `min_spacing` param to `min_dx`.
- `sample_monotone_y`: default of its existing `min_dy` param now references
  `DEFAULT_MIN_DY`; internal call into `sample_interior_x` updated to the
  renamed `min_dx` keyword.
- `generate_random_emissivity`: rename `min_spacing`->`min_dx` (already had
  `min_dy`).
- `generate_random_temp`: rename `min_spacing`->`min_dx`; add new `min_dy`
  parameter, threaded into both the decreasing and increasing
  `sample_monotone_y` calls.
- `generate_random_electron_temp`, `generate_random_ion_temp`: rename
  `min_spacing`->`min_dx`; add `min_dy` passthrough parameter.
- `generate_random_perpendicular_velocity`: rename `min_spacing`->`min_dx`
  only. No `min_dy` added: this generator draws two independent scalar
  amplitudes (ion-root, electron-root), not a monotonic interior sequence.
- `generate_random_parallel_velocity`: rename `min_spacing`->`min_dx`
  (already had `min_dy`).

## 2. `xicsrt_spline_constrained.py`

- Import `DEFAULT_MIN_DX`, `DEFAULT_MIN_DY` from `xicsrt_spline`.
- `generate_random_emissivity_constrained`: rename `min_spacing`->`min_dx`
  (docstring, error messages, feasibility-floor comments; already had
  `min_dy`).
- `generate_random_temp_constrained`: rename `min_spacing`->`min_dx`; add new
  `min_dy` parameter, threaded into its `sample_monotone_y` call.
- `generate_random_ion_temp_constrained`: rename `min_spacing`->`min_dx`; add
  `min_dy` passthrough.
- `PhysicsConstraintOptions` (frozen dataclass): add 10 new fields, defaults
  matching the unconstrained generator defaults:
  - `ti_min_dx`, `ti_min_dy`
  - `te_min_dx`, `te_min_dy`
  - `emiss_min_dx`, `emiss_min_dy`
  - `vperp_min_dx`, `vperp_min_dy` (`vperp_min_dy` unused/no-op; documented
    in the field's docstring)
  - `vpara_min_dx`, `vpara_min_dy`

## 3. `xicsrt_w7x_npablant.py`

- Import `DEFAULT_MIN_DX`, `DEFAULT_MIN_DY` from `xicsrt_spline`.
- `get_config()`: add the same 10 keys under
  `config['scenario']['physics_constraints']`.
- `_physics_constraints_from_config`: read all 10 new keys into the
  `PhysicsConstraintOptions(...)` construction.
- `generate_random_profiles`: thread `min_dx`/`min_dy` into every generator
  call (both the constrained and unconstrained branch for ion temp and
  emissivity), independent of whether the F030 toggles are enabled:
  - electron temp: `te_min_dx`/`te_min_dy`
  - ion temp (both variants): `ti_min_dx`/`ti_min_dy`
  - emissivity (both variants): `emiss_min_dx`/`emiss_min_dy`
  - perpendicular velocity: `vperp_min_dx` only
  - parallel velocity: `vpara_min_dx`/`vpara_min_dy`

## 4. `production/xicsrt_config_task.py`

- Import `DEFAULT_MIN_DX`, `DEFAULT_MIN_DY` from `xicsrt_spline`.
- Add 10 new argparse flags (`--ti-min-dx`, `--ti-min-dy`, `--te-min-dx`,
  `--te-min-dy`, `--emiss-min-dx`, `--emiss-min-dy`, `--vperp-min-dx`,
  `--vperp-min-dy`, `--vpara-min-dx`, `--vpara-min-dy`), each defaulting to
  the imported constants.
- Extend the `physics_constraints` dict built in `main()` with the 10 new
  keys from `args`.

## 5. Tests

- `test_xicsrt_spline.py`: rename `min_spacing=` kwargs to `min_dx=`; add
  `test_min_dy_enforced_on_interior_y_values` for the temperature
  generators.
- `test_xicsrt_spline_constrained.py`: rename `min_spacing=` kwargs to
  `min_dx=`; add `test_ion_temp_constrained_min_dy_enforced_below_clamp`.
- `test_xicsrt_w7x_npablant.py`: add
  `test_get_config_scenario_min_dx_dy_defaults`,
  `test_generate_random_profiles_te_min_dx_applied`,
  `test_generate_random_profiles_ti_min_dx_applied_constrained`.
- `test_xicsrt_config_task.py`: update
  `test_main_builds_physics_constraints_dict` to expect the 10 new keys.

Verification: full `xicsrt_analysis` `w7x_npablant` pytest suite, 101 passed.
