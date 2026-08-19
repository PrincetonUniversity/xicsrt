# F029 - Move orphaned `xicsrt_spline.py` from `xicsrt` core into `xicsrt_analysis`

Status: Done 2026-08-18.

This file includes AI generated content using Claude (Sonnet 5).

## Goal

`xicsrt/xicsrt/tools/xicsrt_spline.py` has exactly one consumer
(`xicsrt_analysis/w7x_npablant`) and is imported nowhere else in the public
`xicsrt` package. Move it to live alongside its only caller, so that F030's
new physics-constrained sibling module does not have to straddle a repo
boundary or duplicate private helpers (`sample_interior_x`,
`sample_monotone_y`, `_deriv_zero_mask`, `_interior_x_free_mask`) across it.

## Scope

- Move `xicsrt/xicsrt/tools/xicsrt_spline.py` ->
  `xicsrt_analysis/w7x_npablant/xicsrt_spline.py` verbatim (module docstring
  gets a short note about the move; no other content change).
- Move `xicsrt/tests/test_spline.py` ->
  `xicsrt_analysis/w7x_npablant/tests/test_xicsrt_spline.py`, fixing only
  the import (`from xicsrt.tools import xicsrt_spline` ->
  `from w7x_npablant import xicsrt_spline`).
- Update the two `xicsrt_analysis` importers:
  `xicsrt_w7x_npablant.py`, `sources/_XicsrtPlasmaW7xProfile.py`.
- Update the two notebooks that import `xicsrt.tools.xicsrt_spline`
  directly ("Part 1", "Part 2b"), clearing all cell outputs first per repo
  rule.
- No `_version.py` bump: per user direction, this is an internal/private
  dependency removal, not a change to any documented public API surface
  used outside `xicsrt_analysis`.
- `devel/jaxrt_sync.md`: dated entry noting no divergence (never a trigger
  file, jaxrt has no plasma sources).

## Verification

- `xicsrt` `tests/`: 64 passed (down from 108 baseline minus the ~44 moved
  `test_spline.py` tests, all of which now live in `xicsrt_analysis` as
  `test_xicsrt_spline.py`).
- `xicsrt_analysis/w7x_npablant` `tests/`: 89 passed after the move (F030
  additions included).
- `python examples/example_00/example_00.py` runs end to end in `xicsrt`.
- `grep -rl xicsrt_spline xicsrt/` finds only historical doc/plan/log
  references (F014/F021/F022 plans, features_request.md, jaxrt_sync.md,
  git internals), no live code.

## Consequences

None for existing saved configs or results: `xicsrt_spline.py`'s public
API (`generate_random_*`, `spline_from_profile`, `profile_seeds`,
`make_profile_dict`) is unchanged, only its import path moved.
