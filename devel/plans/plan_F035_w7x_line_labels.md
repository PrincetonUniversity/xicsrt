# F035 - Always-computed per-ray line label for `multi_voigt` sampling

Status: Done (2026-08-26). See `devel/features_request.md` for the
implementation summary and verification notes.

This file includes AI generated content using Claude (Sonnet 5).

Depends on F034 (`devel/plans/plan_F034_ray_label_infrastructure.md`) for
the `rays['label']` convention.

## Goal

Whenever a source samples wavelengths from a `multi_voigt` mixture
distribution, always populate `rays['label']` with the index of the
specific spectral line each ray was drawn from. No config option is
needed to enable this: it is simply always computed as part of
`multi_voigt` sampling, exactly like the wavelength itself. Every other
`wavelength_dist` (`monochrome`, `uniform`, `voigt`) continues to never
touch `rays['label']`, so the presence/absence of the key is itself the
opt-in signal, matching F034's convention.

Motivating use case: `xicsrt_analysis/w7x_npablant`'s `XicsrtPlasmaW7x`
Ar16+ model already evaluates multiple atomic emission lines (w, z, etc.)
per bundle and mixes them via `multi_voigt`. Labeling each ray with its
originating line will let `xicsrt_ml` training scripts apply per-line
augmentations (e.g. dropping a line, nudging a poorly-atomic-physics-
constrained line) directly on already-raytraced photon samples.

## Scope

### Core `xicsrt`

- `xicsrt/tools/xicsrt_voigt_multi.py`:
  - `multi_voigt_random(...)`: change to always return
    `(random_x, line_index)` instead of just `random_x`. No
    backwards-compatible flag; update every call site in the same change
    (per repo's no-backwards-compatibility mandate).
  - `multi_voigt_random_batched(...)`: same change, always returns
    `(random_x, line_index)`.
  - Update both docstrings' `Returns` sections; both already carry an "AI
    generated" attribution — append the current tool/version.
- `xicsrt/sources/_XicsrtSourceGeneric.py`:
  - `random_wavelength_multi_voigt` (single-bundle, non-plasma path):
    receive `line_index` from `multi_voigt_random` and set
    `rays['label'] = line_index` in `generate_rays` when
    `wavelength_dist == 'multi_voigt'`. Update the call site around line
    380-401.
- `xicsrt/sources/_XicsrtPlasmaGeneric.py`:
  - `_generate_wavelengths`: in the `multi_voigt` branch, receive
    `line_index` from `multi_voigt_random_batched` and return/store it so
    `create_sources` can set `rays['label'] = line_index`. Every other
    branch (`monochrome`, `uniform`, `voigt`) must NOT set `rays['label']`.
- `xicsrt/jaxrt/tools/_wavelength.py`:
  - Mirror the signature change for jaxrt's non-plasma `multi_voigt`
    support (jaxrt has no plasma sources, so only the single-source path
    applies there — see `devel/jaxrt_sync.md`). Set the jaxrt ray bundle's
    `label` field (added unconditionally by F034) from `line_index` when
    `multi_voigt` is selected.
- `devel/jaxrt_sync.md`: add a divergence-log entry for the
  `xicsrt_voigt_multi.py` signature change (a listed trigger file):
  batched/plasma form has no jaxrt equivalent (no plasma sources, per the
  existing F014-F019/F028 divergence entries); single-source form is
  mirrored in `jaxrt/tools/_wavelength.py` per above.
- Update existing call sites/tests for the new two-value return:
  `tests/test_voigt.py`, `tests/test_voigt_direct.py`,
  `tests/test_plasma_wavelength_line_filter.py`,
  `tests/test_plasma_bundle_source.py`.
- New/expanded tests: `line_index` distribution matches line-intensity
  mixture weights (reuse existing KS-test patterns already present in
  `tests/test_voigt_direct.py`/`tests/test_plasma_bundle_source.py`), and
  per-bundle indexing correctness for the batched form (each ray's
  `line_index` is consistent with the bundle it was drawn from and that
  bundle's own line parameters).

### `xicsrt_analysis/w7x_npablant`

- `sources/_XicsrtPlasmaW7x.py`:
  - After `_eval_line_model`/`line_table.labels` is available, record the
    int-to-name mapping once into `config['scenario']['line_labels']` as
    a plain list indexed by int (e.g. `['w', 'z', ...]`), matching the
    `line_index` values that end up in `rays['label']`. Confirm
    `line_table.labels` ordering/content is stable across calls within a
    run before deciding exactly where (e.g. `check_param`, first
    `bundle_generate` call) this write happens, and guard against
    redundant/conflicting writes across iterations.
- Tests (in whichever test suite `xicsrt_analysis/w7x_npablant` uses):
  verify `rays['label']` values index correctly into
  `config['scenario']['line_labels']` for a small `XicsrtPlasmaW7x`
  scenario, and that the label distribution across a large ray sample
  matches the expected per-line intensity ratios.
- Feature tracking: since most of this feature's code lives in
  `xicsrt_analysis` (a separate repo with no `devel/features_request.md`
  of its own as of this writing), continue tracking status/completion
  notes here in `xicsrt`'s `devel/features_request.md`, per the F033
  precedent.

## Out of scope

- Any change to `wavelength_dist` values other than `multi_voigt`.
- Any new user-facing config option to enable/disable labeling — the
  `wavelength_dist` choice itself is the opt-in signal.
- Non-Ar16+/non-W7-X consumers of `multi_voigt` labeling (e.g. other
  plasma sources) — they get the label field "for free" once this lands,
  since it's a property of `multi_voigt` sampling itself, but no other
  source's config/tests are touched by this feature.

## Verification

- `pytest tests/` (core `xicsrt`) passes, including updated/new
  `multi_voigt` tests.
- `xicsrt_analysis/w7x_npablant`'s own test suite passes, including the
  new label/`line_labels` tests.
- Manual check: a small `XicsrtPlasmaW7x` raytrace produces a `label`
  array whose value counts, normalized, approximately match the relative
  line intensities reported by the Ar16+ atomic model at a representative
  temperature.
