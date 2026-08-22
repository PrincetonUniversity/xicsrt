# F032 - CLI for combining W7-X per-sample results into one ML training-set file

Status: Done 2026-08-23

This file includes AI generated content using Claude (Sonnet 5).

## Goal

Give the "Part 4a - Combined ML training-set file" notebook's combining step
(F020) a scripted, version-controlled CLI entry point, matching how F023 gave
config generation and result generation their own production scripts
(`xicsrt_config_task.py`, `xicsrt_train_task.py`). Entirely within
`xicsrt_analysis`; no changes to core xicsrt.

## Scope

- Move `_is_dropped_value` and `combine_training_set` from the Part 4a
  notebook into `xicsrt_analysis/w7x_npablant/xicsrt_results_util.py`,
  verbatim aside from doc/import touch-ups (`xarray` import added to that
  module). Behavior is unchanged: glob `xicsrt_flat_*.hdf5` in an input
  directory, build the padded `(sample, ray, axis)` float32 ray array
  (NaN-filled beyond `sample_count`), drop config keys that are `None` or an
  empty `dict`/`list` in any sample, stack every remaining key with a leading
  `sample` dimension, use `config__scenario__sample_index` as the `sample`
  coordinate when present, verify consistent detector geometry across
  samples, and write summary `attrs`.
- Update the Part 4a notebook to import `combine_training_set` from
  `w7x_npablant.xicsrt_results_util` instead of defining it inline, so the
  notebook and the new CLI cannot diverge. Clear all cell outputs first per
  repo rule (notebook already has none).
- New `xicsrt_analysis/w7x_npablant/production/xicsrt_combine_task.py`: a
  thin `argparse` CLI wrapping `combine_training_set`, following the
  `xicsrt_config_task.py`/`xicsrt_manifest_task.py` conventions (module
  docstring, `sys.path.append` of `xicsrt`/`xicsrt_contrib`/`xicsrt_analysis`,
  `logging.basicConfig`). Options: `--input-path` (default
  `./training_results`, matching `xicsrt_train_task.py`'s `--output-path`
  default), `--output-filename` (default `<input-path>/xicsrt_training_set.nc`),
  `--overwrite`, `--max-samples`.
- New `xicsrt_analysis/w7x_npablant/tests/test_xicsrt_combine_task.py`:
  covers `combine_training_set`'s dropped-key logic, padded ray array and
  `sample_count`, `sample` coordinate sourced from
  `config__scenario__sample_index`, the detector-geometry-mismatch
  `ValueError`, `overwrite`/`max_samples`/missing-input-dir behavior, and the
  CLI's argument wiring into `combine_training_set`.

## Out of scope

- `RayReader`, `rasterize`, and the verification/plotting cells later in the
  Part 4a notebook: unrelated to generating the combined file and left as
  notebook-only reference material.
- Any change to the per-sample `.hdf5` format or `run_one_sample` (F023).

## Verification

- `xicsrt_analysis/w7x_npablant` `tests/`: full suite passes, including the
  new `test_xicsrt_combine_task.py`.
- Manual/CLI smoke check: running `xicsrt_combine_task.py` against a small
  set of real or synthetic per-sample `.hdf5` files reproduces the same `.nc`
  output as calling `combine_training_set` directly (same function either
  way, so this mainly checks the CLI's argument plumbing and defaults).
