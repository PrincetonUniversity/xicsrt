# Plan F037 - Multi-species composite spectra; S XV / S XVI in the Ar16+ model

Status: Pending (not started)
Date: 2026-09-03

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
training set. Each is an unresolved doublet, so four NIST lines are
involved. They land on top of the Li-like j/k satellites and the z line,
i.e. exactly the satellite/resonance region that a Te-from-line-ratios
model keys on.

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
| Lines included | All 4 | Doublet splitting is comparable to the instrumental width |
| S XVI branching | gA-weighted 2.0015:1 | `Rel. Int.` (130:70 = 1.857) are rounded observational estimates; gA is the statistical result expected for Lyman-beta fine structure |
| S XV branching | NIST rel. int. 50:2 | No A value published for the triplet component |
| Amplitude values | `s14_ratio = s15_ratio = 1.0`, fixed | Maximizes downward-augmentation headroom |
| Amplitude definition | total ion photons / Ar16+ w-line photons | Composes with the existing w-line emissivity convention (F015); directly measurable from real data |
| Emissivity calibration | S excluded from `I_tot` | Keeps Ar16+ photon statistics identical to existing training sets |
| S radial profile | Inherits Ar16+ w-line shape | Accepted v1 limitation; documented |
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


## 2. Reference data

### 2.1 Lines (NIST ASD, 3.98-4.00 A, vacuum)

| Ion | Ritz lambda (A) | unc. | Transition | A_ki (1/s) | g_k | Rel. Int. |
|---|---|---|---|---|---|---|
| S XVI (S15+) | 3.99080124 | 5e-8 | 1s 2S_1/2 - 3p 2P_3/2 | 1.0949e13 | 4 | 130 |
| S XVI (S15+) | 3.99194352 | 8e-8 | 1s 2S_1/2 - 3p 2P_1/2 | 1.0941e13 | 2 | 70 |
| S XV  (S14+) | 3.997757   | 3e-6 | 1s2 1S_0 - 1s5p 1P_1  | 3.82e12   | 3 | 50 |
| S XV  (S14+) | 3.998749   | 3e-6 | 1s2 1S_0 - 1s5p 3P_1  | --        | 3 | 2 |

Use the **Ritz** wavelengths (8 significant figures) rather than the
observed values (S XV observed is given only as 3.9983, 4 figures).

### 2.2 Derived branching ratios for the `K` column

S XVI, gA-weighted:

    gA(3/2) = 4 * 1.0949e13 = 4.3796e13
    gA(1/2) = 2 * 1.0941e13 = 2.1882e13
    ratio   = 2.00146
    K       = 0.666829  (3.99080124 A)
              0.333171  (3.99194352 A)

S XV, from relative intensities 50:2:

    K       = 0.961538  (3.997757 A)
              0.038462  (3.998749 A)

`K` is normalized to sum to 1.0 within each ion, so the intensity slot
value equals that ion's total photon count directly. Record the derivation
arithmetic in the file header so it can be re-checked.

### 2.3 Other per-species constants

- S atomic mass: 32.06 amu (Ar: 39.948). Doppler width ratio
  `sqrt(39.948/32.06) = 1.11626`, i.e. S lines are 11.6% broader than Ar
  at equal Ti.
- `atomic_number`: S = 16, Ar = 18.
- `LINE_WIDTH` = A_ki. The S XV triplet has no published A value; use the
  singlet's 3.82e12. Natural width here is ~1e-4 of the Doppler width, so
  this substitution is immaterial -- but note it in the header.


## 3. Part 1 -- `xics_jax` (FR-008)

### 3.1 New line data files

`xics_jax/data/lines/2026-09_NIST_S14.xc_param`
`xics_jax/data/lines/2026-09_NIST_S15.xc_param`

Existing 13-column whitespace format, `;` comment header. Columns:
`LABEL CHARGE_STATE TYPE WAVELENGTH LINE_WIDTH K QD ES N CONF_UPPER
TERM_UPPER CONF_LOWER TERM_LOWER`.

Header must record: NIST ASD as the source with the full query URL and
retrieval date; the Roman-numeral-to-charge-number mapping; that `K`
carries fixed intra-ion branching (NOT a dielectronic branching ratio);
the gA derivation for S XVI; the A-value substitution for the S XV
triplet; and that `QD`/`ES` are unused (set 0.0) for `STATIC` lines.

Suggested labels: `Lyb1`/`Lyb2` (S XVI), `He5s`/`He5t` (S XV). These get
`element:` prefixes on composition, so they need only be unique within
their own file.

Note: these are the first `.xc_param` files in the project not generated
by the IDL `XC_BUILD_ATOMIC_DATA_FILES_*` routines. State that in the
header so provenance is not misattributed.

### 3.2 New `TYPE = STATIC`

Third line type alongside `DIRECT` and `DIELECTRONIC`:

    intensity = slot_value * K

Applied identically in both `use_te` modes. This is what collapses each
sulfur ion to a single free amplitude while preserving its internal
doublet structure.

Implementation:
- `model/lines.py`: add `BRANCH_STATIC = 5` to the `BRANCH_*` constants.
  In `_resolve_use_te_branch`, return `(BRANCH_STATIC, RATE_NONE)` for
  `line_type == "STATIC"`.
- `model/lines.py`: in `build_line_table`, set `intensity_factor = K` for
  `STATIC` lines (`DIELECTRONIC` keeps `qd/1e13`, `DIRECT` keeps 1.0).
  This makes the `use_te=False` path work with no further change.
- `model/excitation.py`: in `compute_use_te_intensities`, add a
  `BRANCH_STATIC` arm to the nested `jnp.where` chain computing
  `slot_value * k`. Note the existing asymmetry -- the `use_te=True` path
  does not apply `intensity_factor` -- so `K` must be applied explicitly
  here rather than relying on `intensity_factor`.
- `io/line_file.py`: `type` is already `.upper()`-ed on read; no reader
  change needed beyond confirming `STATIC` is accepted.

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

`load_spectrum_config` detects the `components` key and returns a
composite; keep the `lru_cache` (frozen and hashable). Decide how
`use_te` composes -- recommendation: it is a property of the *composite*,
and `use_te_supported` is the AND over components that actually need
rates. `STATIC`-only components (s14, s15) have no rate requirement and
must not veto `use_te` for the composite.

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
| `sources/_XicsrtPlasmaW7x.py` `default_config` | Add `spectrum_name`, `s14_ratio`, `s15_ratio`; required because `strict_config_check` is True |
| `sources/_XicsrtPlasmaW7x.py` `_eval_line_model` | `w_mask = labels == 'W'` -> `'ar16:W'` |
| `sources/_XicsrtPlasmaW7x.py` `_eval_line_model` | Pass `s14_ratio`/`s15_ratio` into `ModelParams.from_dict` alongside `scale_factor` |
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

Expect the total ray count per sample to rise. Confirm the increase is
consistent with `s14_ratio + s15_ratio` relative to the w line and that
run time / `max_rays` headroom is still acceptable.

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
- **F039 `li_fraction`.** Independent, and a larger error on the primary
  observable than sulfur is. Should be fixed before Te training.
- **Sulfur-specific radial emissivity profiles.** Accepted v1 limitation.
- **The unidentified Ar17+ sulfur line.** Not yet identified.
- **Fe24+ / Mo32+ composites.** Enabled by this work but not built here;
  Mo32 line data does not exist yet and the Ar17 `use_te` path is
  unsupported.


## 8. Verification

1. `pytest` green in `xics_jax`, including the bit-for-bit `ar16` golden
   test (3.6), and in `xicsrt_analysis` (`w7x_npablant/tests/`).
   Note the 2 pre-existing `xarray` collection errors per F020.
2. `evaluate_spectrum('ar16_s')` over 3.94-4.08 A: exactly 4 sulfur lines
   at the tabulated Ritz wavelengths; S line sigma / Ar line sigma =
   1.11626 at equal Ti; S XVI doublet intensity ratio = 2.0015; each S
   ion's total = its slot value.
3. Composite line count = 184 (ar16) + 2 + 2 = 188; labels all prefixed;
   no duplicate labels.
4. S XVI resolves to `BRANCH_STATIC`, not `BRANCH_LILIKE` (the charge
   state 15 collision, 3.4).
5. `verify_ray_calibration.py`: Ar16+ ray-count relative std unchanged
   from the current 0.10 over 50 seeds.
6. End-to-end: 5-sample training set -> combine -> confirm `label`
   present and `-1`-padded, `line_labels` retrievable, and S-labelled
   rays land in the expected detector region.
7. Confirm the combined-file `description` attribute states that the set
   is deliberately sulfur-rich and requires augmentation.


## 9. Open questions for the reviewer

Flagged in detail in `review_F037_multi_species_spectra.md`. In brief:
the gA-vs-observed branching choice, the missing S XV triplet A value,
the shared-emissivity-profile approximation, the "sulfur-rich then
augment down" dataset design, and whether excluding sulfur from the
emissivity calibration is the right normalization convention.
