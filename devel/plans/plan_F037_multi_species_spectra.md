# Plan F037 - Multi-species composite spectra; S XV / S XVI in the Ar16+ model

Status: Pending (not started); reviewed 2026-09-04, see
`review_F037_multi_species_spectra.md`. Review changes are folded in
below and marked "(review)".
Date: 2026-09-03, revised 2026-09-04

Repos touched (feature tracking lives here in `xicsrt` per the
F029/F030/F031 precedent; no core xicsrt code changes):

- `xics_jax` (branch `devel_npablant`) -- tracked there as FR-008
- `xicsrt_analysis` (branch `devel_npablant_accel`)
- `xicsrt_ml` (branch `devel_npablant`) -- F038 portion only

Related: F038 (per-ray label plumbing) is a hard prerequisite for the
augmentation strategy and is implemented as Part 3 of this plan.
F039 (`li_fraction`) is independent and NOT addressed here.


## 0. Context and motivation

Two contaminating sulfur features appear in real W7-X Ar16+ spectrometer
data near 3.98-4.00 A, strong enough to require inclusion in the ML
training set. Each is an unresolved doublet. They land on top of the
Li-like j/k satellites and the z line, i.e. exactly the
satellite/resonance region that a Te-from-line-ratios model keys on.

(review) The 3.998 A feature (S XV 1s2-1s5p) is one member of a Rydberg
series. The next member, **1s2-1s6p at 3.95012 A, is on the detector
0.92 mA redward of the Ar w line** at ~0.57x the 5p intensity, and
1s2-1s7p (3.92193 A) is partially on-detector. 6p is therefore a direct
contaminant of the primary Ti observable and must be included. The
actual Ar16+ detector span is 3.9154-4.0239 A (all rows; 3.9406-4.0239
on the mid row), computed from `xics_jax.geometry` for three
calibration blocks.

The same machinery is required for the future Ar17+ model (simultaneous
Fe24+, Mo32+ and a further unidentified S line), so this is built as a
general multi-species composite capability rather than a sulfur
special case.

### Naming, and the easiest way to get this wrong

Spectroscopic notation is one greater than the charge number:

- **S XV**  = He-like sulfur = **S14+** -> `xics_jax` spectrum name `s14`
- **S XVI** = H-like sulfur  = **S15+** -> `xics_jax` spectrum name `s15`

`xics_jax` consistently names spectra by charge number (`ar16` = Ar16+ =
Ar XVII). Every reference in the NIST source table is in Roman numerals.
Confirm the mapping once, at the top of each new data file header, and
never re-derive it inline.


## 1. Decisions already made (do not relitigate without asking)

| Decision | Value | Rationale |
|---|---|---|
| Lines included (review) | Full NIST S XV 1s2-1s np series n=4..8 (singlets + 4p/5p triplet partners) and both S XVI Lyb components; off-detector members filtered as for Ar16 | 6p sits on the Ar w line; the series is physically tied together; user wants the wider set with filtering downstream |
| S XV series ratios (review) | Frozen at NIST A-value ratios (electron-impact scaling, ~n^-3); one amplitude per ion | No data yet constrain 5p:6p:7p; revisit if recombination/CX flattening is found |
| S XVI branching | gA-weighted, Lyb 1/2 = 0.4996 x Lyb 3/2 | `Rel. Int.` (130:70 = 1.857) are rounded observational estimates; gA is the statistical result for Lyman-beta fine structure, good to ~1% |
| S XV triplet branching | NIST rel. int. (5p: 2:50, 4p: 5:100) | No A value published; ratio is actually Te-dependent (2-10%), immaterial for a nuisance line |
| Amplitude values | `s14_ratio = s15_ratio = 1.0`, fixed | Maximizes downward-augmentation headroom; 0.5-1.0 x w has been observed at campaign startup, so 1.0 is physical |
| Amplitude definition (review) | photons in one named **reference line** / Ar16+ w-line photons. S XV ref = 1s5p 1P1; S XVI ref = Lyb 3/2 | Same convention as Ar `w_intensity`; survives any wavelength filter; directly measurable |
| Ratio storage (review) | YAML `static_ratios` + `static_reference` per component, NOT the `.xc_param` `K` column | Ratios are a provisional model assumption, not atomic data; avoids relabelling the dielectronic `K` |
| Emissivity calibration | S excluded from `I_tot` | Keeps Ar16+ photon statistics identical to existing training sets |
| S radial profile | Inherits Ar16+ w-line shape | Accepted v1 limitation; S XV Ti error ~15-30% in high-Te cores, small at startup; S XVI <~10% |
| Breaking API change | Approved | `SpectrumConfig.atomic_mass` scalar -> per-line |

### Why the amplitudes are fixed rather than randomized

Line-intensity augmentation in the training loop can only go *downward*:
removing a fraction of a line's rays is a per-ray Bernoulli trial and
preserves exact Poisson statistics. Scaling *upward* requires
duplicating rays, which does not, and would violate the project's
"photon statistics are sacrosanct" mandate.

Therefore every sample is generated at the maximum intended sulfur level
(ratio 1.0), and the training loop augments down to any lower level.
Randomizing the raytrace ratio over, say, 0.1-1.0 would leave most
samples stuck near the bottom of the range with no headroom, and would
require regenerating the dataset to change the range later.

Consequence: the raw training set is *not* representative of any single
plasma; it is deliberately sulfur-rich. Augmentation is mandatory for it
to be used correctly. This must be stated in the combined-file
`description` attribute.

(review) Two constraints on the training loop follow and must also be
stated there:

1. Any per-sample normalization (total counts, max pixel, ...) must be
   computed **after** thinning, never cached from the raw sulfur-rich
   file; otherwise low-sulfur samples are mis-normalized in a way that
   correlates with the nuisance.
2. Thinning must select rays by `label` (F038), never by wavelength or
   detector region: 6p is inside the w line.


## 2. Reference data

### 2.1 Lines (NIST ASD ver. 5.12, S XIV-XVI, 3.90-4.10 A, vacuum) (review)

All NIST-tabulated lines in the window. Labels are the suggested
`.xc_param` labels. "Det." is whether the line lands on the Ar16+
detector (span 3.9154-4.0239 A over all rows).

| Ion | Label | Ritz lambda (A) | unc. | Transition | A_ki (1/s) | g_k | Rel. Int. | Det. |
|---|---|---|---|---|---|---|---|---|
| S XV (S14+) | He8s | 3.903850 | 3e-6 | 1s2 1S_0 - 1s8p 1P_1 | 9.21e11 | 3 | 20 | no |
| S XV (S14+) | He7s | 3.921932 | 3e-6 | 1s2 1S_0 - 1s7p 1P_1 | 1.38e12 | 3 | 25 | high-y rows |
| S XV (S14+) | He6s | 3.950117 | 3e-6 | 1s2 1S_0 - 1s6p 1P_1 | 2.19e12 | 3 | 35 | **yes, on Ar w** |
| S XVI (S15+) | Lyb1 | 3.99080124 | 5e-8 | 1s 2S_1/2 - 3p 2P_3/2 | 1.0949e13 | 4 | 130 | yes, on k |
| S XVI (S15+) | Lyb2 | 3.99194352 | 8e-8 | 1s 2S_1/2 - 3p 2P_1/2 | 1.0941e13 | 2 | 70 | yes, on j |
| S XV (S14+) | He5s | 3.997757 | 3e-6 | 1s2 1S_0 - 1s5p 1P_1 | 3.82e12 | 3 | 50 | yes, z red wing |
| S XV (S14+) | He5t | 3.998749 | 3e-6 | 1s2 1S_0 - 1s5p 3P_1 | -- | 3 | 2 | yes |
| S XV (S14+) | He4s | 4.088498 | 3e-6 | 1s2 1S_0 - 1s4p 1P_1 | 7.53e12 | 3 | 100 | no |
| S XV (S14+) | He4t | 4.090551 | 3e-6 | 1s2 1S_0 - 1s4p 3P_1 | -- | 3 | 5 | no |

Use the **Ritz** wavelengths rather than the observed values (S XV
observed values are 4 figures). Include the off-detector lines in the
files; `exclude_outside_lines` / the detector range filter them, as for
the Ar16 file's own out-of-range members.

The Ar w line is at 3.9492 A (MZ file). He6s is +0.92 mA from it, about
2 pixels at 0.426 mA/pixel, i.e. fully blended at any W7-X Ti.

### 2.2 Fixed intra-ion ratios (YAML `static_ratios`) (review)

Each ion has one free amplitude equal to the photon count of its
**reference line**; every other line is a fixed multiple. Ratios live in
the component YAML (section 3.5), not in the `.xc_param` `K` column.

S XV, reference `He5s` = 1.0. Singlet ratios are NIST A-value ratios
(equivalent to electron-impact excitation scaling, ~n^-3, to ~1%);
triplet ratios are NIST relative intensities scaled to the same
reference:

    He4s  7.53e12 / 3.82e12 = 1.9712
    He5s                    = 1.0
    He6s  2.19e12 / 3.82e12 = 0.5733
    He7s  1.38e12 / 3.82e12 = 0.3613
    He8s  9.21e11 / 3.82e12 = 0.2411
    He5t  2/50              = 0.0400
    He4t  (5/100) * 1.9712  = 0.0986

S XVI, reference `Lyb1` = 1.0, gA-weighted:

    gA(3/2) = 4 * 1.0949e13 = 4.3796e13
    gA(1/2) = 2 * 1.0941e13 = 2.1882e13
    Lyb2 / Lyb1 = 2.1882e13 / 4.3796e13 = 0.4996

Record the derivation arithmetic in the YAML comments so it can be
re-checked. State there that the S XV n-series ratio assumes
electron-impact excitation and is Te-independent by assumption
(excitation thresholds rise ~37 eV 5p->6p and ~60 eV 6p->7p, so 6p/5p is
~7% lower at 500 eV than at 3 keV; recombination/CX at low Te could
flatten the series further -- both unmodelled), and that the triplet
ratios are Te-dependent in reality (2-10% plausible) and frozen.

### 2.3 Other per-species constants

- S atomic mass: 32.06 amu (Ar: 39.948), both standard atomic weights;
  be consistent (do not mix with isotope masses). Doppler width ratio
  `sqrt(39.948/32.06) = 1.11626`, i.e. S lines are 11.6% broader than Ar
  at equal Ti.
- `atomic_number`: S = 16, Ar = 18.
- `LINE_WIDTH` = A_ki (nominally the total upper-level decay rate; for
  S XVI 3p the 3p->2s channel adds ~10%, immaterial). The S XV triplet
  lines have no published A value; use the corresponding singlet's.
  (review) Natural width here is ~1e-3 to 5e-3 of the Doppler width
  over Ti = 0.2-4 keV (gamma = 0.0016 mA vs sigma = 0.33-1.46 mA), so
  the substitution is immaterial -- note it and the convention in the
  header.
- `K`, `QD`, `ES` are all `0.0` for `STATIC` lines and unused; say so in
  the header.


## 3. Part 1 -- `xics_jax` (FR-008)

### 3.1 New line data files

`xics_jax/data/lines/2026-09_NIST_S14.xc_param`
`xics_jax/data/lines/2026-09_NIST_S15.xc_param`

Existing 13-column whitespace format, `;` comment header. Columns:
`LABEL CHARGE_STATE TYPE WAVELENGTH LINE_WIDTH K QD ES N CONF_UPPER
TERM_UPPER CONF_LOWER TERM_LOWER`.

Header must record: NIST ASD ver. 5.12 as the source with the full
query URL and retrieval date; the Roman-numeral-to-charge-number
mapping; that the file is pure atomic data and the intra-ion intensity
ratios live in the spectrum YAML (review: NOT in `K`); that `K`, `QD`,
`ES` are `0.0` and unused for `STATIC` lines; the `LINE_WIDTH`
convention and the A-value substitution for the S XV triplets; and a
note that the Ar16 file's k line must stay at the MZ value 3.9900 A
because the historical +0.3/+0.6 mA "corrections" are consistent with an
unmodelled Lyb1 blend (see review section 4).

Labels (review): `Lyb1`/`Lyb2` (S XVI); `He4s`/`He4t`/`He5s`/`He5t`/
`He6s`/`He7s`/`He8s` (S XV). These get `element:` prefixes on
composition, so they need only be unique within their own file.

Note: these are the first `.xc_param` files in the project not generated
by the IDL `XC_BUILD_ATOMIC_DATA_FILES_*` routines. State that in the
header so provenance is not misattributed.

### 3.2 New `TYPE = STATIC`

Third line type alongside `DIRECT` and `DIELECTRONIC` (review: ratio
source changed from `K` to a new `static_factor`):

    intensity = slot_value * static_factor

Applied identically in both `use_te` modes. This is what collapses each
sulfur ion to a single free amplitude (the reference line's photon
count) while preserving its internal line-ratio structure.

Implementation:
- `model/lines.py`: add `BRANCH_STATIC = 5` to the `BRANCH_*` constants.
  In `_resolve_use_te_branch`, return `(BRANCH_STATIC, RATE_NONE)` for
  `line_type == "STATIC"`.
- `model/lines.py`: add `static_factor: np.ndarray` to `LineTable`
  (include in `_hash_key`). In `build_line_table`, for `STATIC` lines
  look the label up in `config.static_ratios`; the reference label gets
  1.0; a `STATIC` label absent from `static_ratios` is an error (no
  silent zero). Non-`STATIC` lines get `static_factor = 1.0`. Set
  `intensity_factor = static_factor` for `STATIC` lines (`DIELECTRONIC`
  keeps `qd/1e13`, `DIRECT` keeps 1.0), so the `use_te=False` path works
  with no further change.
- `model/lines.py`: guard -- `k` must never reach a `STATIC` line's
  intensity by any path. Assert `k == 0.0` for `STATIC` lines on read of
  the vendored files, or document and test that no branch multiplies
  `k` for `BRANCH_STATIC`.
- `model/excitation.py`: in `compute_use_te_intensities`, add a
  `BRANCH_STATIC` arm to the nested `jnp.where` chain computing
  `slot_value * static_factor`. Note the existing asymmetry -- the
  `use_te=True` path does not apply `intensity_factor` -- so
  `static_factor` must be applied explicitly here.
- `io/line_file.py`: `type` is already `.upper()`-ed on read; no reader
  change needed beyond confirming `STATIC` is accepted. The file format
  is unchanged (review: no new column).

### 3.3 Per-line `atomic_mass` (breaking change)

`SpectrumConfig.atomic_mass` is currently a scalar used at
`model/forward.py` for the Doppler width. A composite spectrum needs a
per-line value.

- Add `atomic_mass: np.ndarray` to `LineTable`; include it in
  `_hash_key` (the table is a jit static arg and must stay hashable).
- `build_line_table` broadcasts the component's scalar `atomic_mass`
  across its lines.
- `_compute_line_params` reads `line_table.atomic_mass` instead of
  `config.atomic_mass`.
- The YAML key stays `atomic_mass` on each component; only the plumbing
  changes.

Decide explicitly: keep `SpectrumConfig.atomic_mass` as the per-component
declaration (recommended -- it is still meaningful for a single-species
config) but ensure nothing downstream reads it for physics. Per the
no-backwards-compatibility mandate, update every call site in the same
change; do not leave a scalar fallback.

Call sites to update (found by search; re-verify):
`model/forward.py`, `model/registry.py`, `notebooks/01_quickstart.ipynb`,
`tests/`, `benchmarks/`. Note the notebook must have outputs cleared
before editing.

### 3.4 New `atomic_number` column

Add `atomic_number` to `LineTable`, sourced from a new required YAML key
per component.

**This is a correctness fix, not bookkeeping.** `_resolve_use_te_branch`
selects `BRANCH_LILIKE` via `charge_state == config.li_like_charge_state`
(= 15 for the Ar16 config). S XVI is *also* charge state 15. If branches
were resolved globally after concatenation, H-like sulfur would be
silently misclassified as Li-like argon and fed Marchuk argon excitation
rates. Two independent defenses:

1. Resolve branches per component *before* concatenation (3.5).
2. Match on `(atomic_number, charge_state)`, never `charge_state` alone.

Add a regression test asserting S XVI lines resolve to `BRANCH_STATIC`
in the composite.

### 3.5 `CompositeSpectrumConfig`

New frozen dataclass in `model/registry.py`:

    @dataclass(frozen=True)
    class CompositeSpectrumConfig:
        name: str
        components: tuple[SpectrumConfig, ...]

New YAML `xics_jax/data/spectra/ar16_s.yaml`:

    name: ar16_s
    components: [ar16, s14, s15]

New component YAMLs `s14.yaml` / `s15.yaml` (review) carry the fixed
ratios alongside the usual keys, e.g. for `s14`:

    name: s14
    line_source_file: 2026-09_NIST_S14.xc_param
    atomic_mass: 32.06
    atomic_number: 16
    wavelength_ref: 3.997757
    use_te_supported: true      # STATIC-only, no rates needed
    intensity_param_names: [he5s_intensity]
    intensity_rules:
      - [STATIC, He5s, he5s_intensity]
      - [STATIC, He5t, he5s_intensity]
      - [STATIC, He4s, he5s_intensity]
      ...                       # every S XV label -> the one slot
    static_reference: He5s
    static_ratios:              # relative to static_reference (= 1.0)
      He4s: 1.9712              # 7.53e12 / 3.82e12  (NIST A ratio)
      He6s: 0.5733              # 2.19e12 / 3.82e12
      He7s: 0.3613              # 1.38e12 / 3.82e12
      He8s: 0.2411              # 9.21e11 / 3.82e12
      He5t: 0.0400              # NIST rel. int. 2/50
      He4t: 0.0986              # (5/100) * 1.9712

and for `s15`: `static_reference: Lyb1`, `static_ratios: {Lyb2: 0.4996}`
(gA-weighted; derivation in a comment). `SpectrumConfig` gains
`static_reference: str` and `static_ratios: tuple[tuple[str, float],
...]` (tuple, so it stays hashable). Both are optional for components
with no `STATIC` lines. Keep the physics-assumption comments from
section 2.2 in these files: this is where a future reader will look
when the n-series ratio is revisited.

`load_spectrum_config` detects the `components` key and returns a
composite; keep the `lru_cache` (frozen and hashable). Decide how
`use_te` composes -- recommendation: it is a property of the *composite*,
and `use_te_supported` is the AND over components that contain any
non-`STATIC` line. `STATIC`-only components (s14, s15) have no rate
requirement and must not veto `use_te` for the composite.

`build_composite_line_table(composite_config)`:

1. Build each component's `LineTable` independently, via the existing
   `build_line_table`, so branch resolution and intensity-parameter
   resolution use that component's own config.
2. Prefix labels with `element:` (`ar16:W`, `s15:Lyb1`).
3. Concatenate arrays; offset each component's
   `intensity_param_index` by the running total of preceding components'
   `intensity_param_names` lengths (preserving -1 as -1).
4. Concatenate `intensity_param_names` with the same prefixing so
   `ModelParams.from_dict` can address slots unambiguously.

The rate table stays per-component. Simplest approach: keep the existing
single `rate_table` argument and have `STATIC`/no-rate components carry
`rate_index = RATE_NONE`. Verify the composite does not need more than
one rate table -- with only Ar16 requiring Marchuk rates today it does
not, but the Ar17+Fe24 case may, and the interface should not preclude
it.

### 3.6 Regression guard (do this first)

Before any refactor, capture the current pure-`ar16` output as a golden
regression. The IDL reference values in `tests/idl_reference/` already
serve this purpose; confirm they are exercised and passing on the
current tree, then require bit-for-bit agreement after 3.3 and 3.5.

This is the safety net for the whole part. If it cannot be made to pass
bit-for-bit, stop and report rather than loosening the tolerance.

### 3.7 `xics_jax` tracking

Add FR-008 to `xics_jax/devel_ai/features_request.md`, following that
repo's per-entry format (Date/Status/Description/Key prompt/Plan/
Implementation notes). Note that repo does not use the Pending/Done
summary tables.


## 4. Part 2 -- `xicsrt_analysis`

All in `w7x_npablant/`.

| Site | Change |
|---|---|
| `sources/_XicsrtPlasmaW7x.py` `_eval_line_model`, `get_line_labels` | Replace the two hardcoded `load_spectrum_config('ar16', use_te=True)` calls with a `spectrum_name` config option (default `'ar16_s'`) |
| `sources/_XicsrtPlasmaW7x.py` `default_config` | Add `spectrum_name`, `s14_ratio`, `s15_ratio`; required because `strict_config_check` is True. (review) Document them as I(S XV He5s)/I(Ar w) and I(S XVI Lyb1)/I(Ar w) -- reference-line photon counts, not ion totals |
| `sources/_XicsrtPlasmaW7x.py` `_eval_line_model` | `w_mask = labels == 'W'` -> `'ar16:W'` |
| `sources/_XicsrtPlasmaW7x.py` `_eval_line_model` | Pass `s14_ratio`/`s15_ratio` into `ModelParams.from_dict` as the `s14:he5s_intensity` / `s15:lyb1_intensity` slots, alongside `scale_factor` |
| `sources/_XicsrtPlasmaW7x.py` `_full_spectrum_factor` | `I_tot` must sum **Ar components only** (see 4.1) |
| `sources/_XicsrtPlasmaW7x.py` `check_param` | Update the "models the Ar16+ spectrum" message |
| `xicsrt_w7x_npablant.py` `get_config` | **Delete** `mass_number`, `wavelength`, `linewidth` (see 4.2) |

No change to `_PROFILE_NAMES` or `generate_random_profiles`: the sulfur
ratios are fixed, not sampled. This is deliberate -- it means existing
per-profile seed streams are untouched and previously generated training
sets remain exactly reproducible.

### 4.1 Keeping sulfur out of the emissivity calibration

`_full_spectrum_factor` returns `I_tot/I_w`, which multiplies the bundle
emissivity before the Poisson draw, and also drives `shape_integral` ->
`calibrate_emissivity_scale`.

If sulfur entered `I_tot`, adding sulfur would *reduce* `emissivity_scale`
and therefore reduce the Ar16+ ray count, making Ar photon statistics a
function of the sulfur nuisance parameter. That is a confound and would
break comparability with existing training sets.

Implementation: compute the Ar-only sum for the calibration factor, and
add the sulfur contribution as a separate additive term that does not
feed `shape_integral`. Use the `atomic_number` / label prefix to select,
never a wavelength range or a hardcoded index.

Expect the total ray count per sample to rise. (review) With the
reference-line definition and the full line set, the S photon load in
w-line units is `s14_ratio * sum(static_ratios on detector)` +
`s15_ratio * 1.4996`; at ceiling that is roughly 1.6-1.9 (S XV,
depending on how much of He7s is on-detector) + 1.5 (S XVI) ~ 3.3
w-equivalents, against an Ar `I_tot/I_w` of ~2.75 at 1 keV -- i.e. up
to ~+100% rays at 1 keV, proportionally less at low Te where Ar
satellites dominate `I_tot`. Confirm the measured increase matches this
and that run time / `max_rays` headroom is still acceptable.

### 4.2 Deleting the inert options

`mass_number`, `wavelength` and `linewidth` in
`config['sources']['plasma']` are **verified inert** when
`wavelength_dist == 'multi_voigt'`: the dispatch in
`xicsrt/sources/_XicsrtPlasmaGeneric.py._generate_wavelengths` reads them
only in the `monochrome` and `voigt` branches, and `XicsrtPlasmaW7x.
check_param` hard-requires `multi_voigt`. They are nonetheless written
into every output file, where `mass_number = 39.948` now actively implies
argon-only physics for what is a multi-species spectrum.

Removing them changes the flat-config key set, so old and new per-sample
hdf5 files can no longer be combined together (`combine_training_set`
raises if key sets differ across samples). Acceptable, but plan the
regeneration and note it in the feature entry when closing.


## 5. Part 3 -- F038 per-ray label plumbing (prerequisite)

Without this the augmentation strategy in section 1 cannot work at all.

### 5.1 Verify the failure first

`combine_training_set` special-cases only `intersect` for ragged padding.
`label` has the same per-sample ragged length and falls through to
`np.stack(...)`, which should raise `ValueError` on inhomogeneous shapes.
No test covers it and no combined `.nc` exists in the repo, so this has
apparently never been exercised since F035.

Reproduce it explicitly (two small flat files with different found-ray
counts, both containing `label`) before changing anything, and record the
actual failure mode. Do not assume.

### 5.2 Fix

- `xicsrt_results_util.combine_training_set`: pad `label` into a
  `(sample, ray)` int16 array filled with `-1`, mirroring the existing
  `intersect` padding. `-1` rather than NaN so the array stays integral;
  document that padding is `-1` and valid entries are
  `label[ii, :sample_count[ii]]`.
- Handle `config__scenario__line_labels` (a per-sample list of strings).
  Verify whether stacking to a `(sample, n_lines)` `<U` array survives
  `to_netcdf` with the `h5netcdf` engine. If not, store it as a dataset
  attribute after asserting it is identical across samples -- which it
  should be, and asserting that is itself a useful guard.
- Add the missing test in `tests/test_xicsrt_results_util.py`.

### 5.3 `xicsrt_ml`

- `schema.py`: add a `label` field to `XarraySchema` (the module docstring
  designates this the single place for data-format changes).
- `dataset.py`: `RayReader` reads labels alongside `intersect`.
- Beware the documented past bug: per-sample scalars must use
  `_bulk_scalar`, not `_bulk_row`. `label` is per-ray, so it follows the
  `intersect` path, but any new per-sample scalar (e.g. the S ratios) must
  use `_bulk_scalar`.

The augmentation itself (downward Bernoulli ray dropping) is explicitly
**out of scope** here -- see section 7.


## 6. Part 4 -- core `xicsrt`

No changes. `multi_voigt` sampling
(`xicsrt/tools/xicsrt_voigt_multi.py`) and the jaxrt path
(`xicsrt/jaxrt/tools/_wavelength.py`) both operate on arbitrary
location/sigma/gamma/intensity arrays and are already species-agnostic.

**No `devel/jaxrt_sync.md` entry is required** (verified: the numpy and
jax engines share the same species-agnostic interface, and no physics
change is being made on either side).


## 7. Out of scope

- **Training-loop intensity augmentation.** Downward-only Bernoulli ray
  dropping. Preserves Poisson statistics exactly; upward scaling would
  not and must never be added. File separately once F038 lands.
- **F039 `li_fraction`.** Independent. (review) Which of F037/F039 is
  the larger error depends on the target: for **Te**, F039 is larger
  (q/r/s/t satellites are comparable to w at 200 eV-1 keV) and must be
  fixed before Te training; for **Ti**, F037 is larger (He6s directly
  broadens the w line) and must be fixed, with 6p included, before Ti
  training.
- **Sulfur-specific radial emissivity profiles.** Accepted v1 limitation
  (S XV Ti error ~15-30% in high-Te cores; S XVI <~10%).
- **S XV n-series ratio flexibility.** One amplitude per ion, ratios
  frozen at electron-impact scaling. Promote to one amplitude per upper
  n only if data show recombination/CX flattening.
- **Lyb dielectronic satellites; Te dependence of the S XV triplet
  ratios.** Few-percent effects; accepted omissions.
- **The unidentified Ar17+ sulfur line.** Not yet identified. (review)
  Lead: S XVI Lyg (1s-4p) scales to ~3.784 A, inside the Ar17+ window;
  verify against NIST when that model is built.
- **Fe24+ / Mo32+ composites.** Enabled by this work but not built here;
  Mo32 line data does not exist yet and the Ar17 `use_te` path is
  unsupported. Mo is traces-only on W7-X and not diagnostic.
- **Research items surfaced by the review** (not F037): test whether
  unmodelled sulfur explains (a) the historical apparent k-line shift,
  (b) the z/k vs w/n=3 vs Thomson Te disagreement, (c) the w-line Ti
  being ~165 eV high in many discharges (prediction: a red asymmetry of
  w that scales with the 3.998 A feature). PHA and HEXOS can bound
  typical post-startup S levels. The S line files from this feature are
  the tool for that investigation.


## 8. Verification

1. `pytest` green in `xics_jax`, including the bit-for-bit `ar16` golden
   test (3.6), and in `xicsrt_analysis` (`w7x_npablant/tests/`).
   Note the 2 pre-existing `xarray` collection errors per F020.
2. `evaluate_lines('ar16_s')` (unfiltered): exactly 9 sulfur lines at
   the tabulated Ritz wavelengths (7 S XV, 2 S XVI); S line sigma / Ar
   line sigma = 1.11626 at equal Ti; per-line intensities equal
   `slot * static_ratios[label]` with the reference line equal to the
   slot value exactly; S XVI Lyb1/Lyb2 = 2.0015; He6s/He5s = 0.5733.
   (review) Then with a 3.9154-4.0239 A filter: 6 sulfur lines remain
   (He7s, He6s, Lyb1, Lyb2, He5s, He5t) and the reference lines' values
   are unchanged by the filter.
3. Composite line count = 184 (ar16) + 7 + 2 = 193 unfiltered; labels
   all prefixed; no duplicate labels.
4. S XVI resolves to `BRANCH_STATIC`, not `BRANCH_LILIKE` (the charge
   state 15 collision, 3.4). `k` is `0.0` on every `STATIC` line and no
   branch multiplies it.
5. `verify_ray_calibration.py`: Ar16+ ray-count relative std unchanged
   from the current 0.10 over 50 seeds.
6. End-to-end: 5-sample training set -> combine -> confirm `label`
   present and `-1`-padded, `line_labels` retrievable, and S-labelled
   rays land in the expected detector region. (review) Specifically
   check that `s14:He6s` rays land within ~2 pixels of `ar16:W` rays --
   this is the blend the whole review turned on.
7. Confirm the combined-file `description` attribute states that the set
   is deliberately sulfur-rich, requires augmentation by `label`, and
   that per-sample normalization must be computed post-thinning.
8. (review) Environment note: `jax`/`xics_jax` are not on the default
   pyenv interpreter on the development Mac; the `desc` pyenv has them.
   Run the `xics_jax` checks there.


## 9. Review

Reviewed 2026-09-04. Brief: `review_request_F037_multi_species_spectra.md`.
Findings and the decisions log: `review_F037_multi_species_spectra.md`.
All review outcomes are folded into this plan and marked "(review)".
The one blocking finding was the omission of S XV 1s6p on the Ar w line
(section 2.1); the other changes are the reference-line amplitude
definition, moving the fixed ratios from the `K` column to YAML
`static_ratios`, the corrected natural-width magnitude, and the Ti/Te
ordering relative to F039.
