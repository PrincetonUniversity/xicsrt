# Review - F037 multi-species composite spectra

Status: **Complete** (2026-09-04). Responds to
`review_request_F037_multi_species_spectra.md`. Conducted interactively
with the domain expert over two sessions; decisions taken during the
review are recorded inline as "Decision:" and have been folded into
`plan_F037_multi_species_spectra.md` (see section 7 of this file for the
list of plan changes).

Reviewer: Claude (Fable 5.1). Physics judgements below are the
reviewer's; atomic data were checked against NIST ASD ver. 5.12 (saved
query supplied by the user; live access to physics.nist.gov was not
available). Geometry numbers were computed directly from
`xics_jax.geometry` (`w7x_ar16`, calibration blocks 180707017,
220901001, 240901001; all agree to <1 mA).


## 1. Verdict

The physics is sound enough to implement **after one blocking
correction**: the plan's line set is incomplete. The S XV 1s2-1s5p line
is one member of a Rydberg series whose next member, **1s2-1s6p at
3.95012 A, lands on the Ar16+ detector 0.92 mA redward of the Ar w line
(3.9492 A)**. Its intensity is ~0.57x the 5p line (A-value / n^-3
scaling), so whenever the 5p feature is visible, 6p is a direct
contaminant of the primary Ti observable. The 1s7p line (3.92193 A) is
also partially on-detector. Neither is in the plan. With this fixed, and
with the amplitude redefined as a named reference line (so it survives
wavelength filtering), the remaining decisions -- gA-weighted Lyb
branching, the triplet A-value stand-in, frozen intra-ion ratios,
shared emissivity profile, w-line normalization with S excluded from the
calibration, sulfur-rich-then-thin dataset design -- are all correct or
acceptable for a v1 nuisance model. No other blocking errors.

A secondary outcome of the review is that unmodelled sulfur is a
plausible common cause of three long-standing anomalies in the W7-X
Ar16+ analysis (k-line apparent shift, z/k vs w/n=3 Te disagreement,
w-line Ti ~165 eV high). These become testable once the S line files
exist; they are recorded in section 5 as research items, not F037 scope.


## 2. Per-item findings

### P1. Atomic data selection and accuracy -- **Disagree (incomplete); transcription correct**

Transcription: all four rows of plan section 2.1 match NIST ASD 5.12
exactly (Ritz lambda, uncertainty, A_ki, g_k, Rel. Int.). Ritz over
observed is correct: the S XV upper levels are semi-empirical, and the
observed 3.9983 is 4 significant figures. The `g` suffix on the NIST
intensities flags a ground-term transition, not data quality.

Completeness: the 3.90-4.10 A NIST query for S XIV-XVI returns 9 lines,
not 4. The extra five are the S XV `1s2 1S0 - 1s np 1P1` series:

| S XV upper level | Ritz (A) | A_ki (1/s) | Rel. Int. | A/A(5p) | (5/n)^3 |
|---|---|---|---|---|---|
| 1s4p 1P1 | 4.088498 | 7.53e12 | 100 | 1.971 | 1.953 |
| 1s4p 3P1 | 4.090551 | -- | 5 | -- | -- |
| 1s5p 1P1 | 3.997757 | 3.82e12 | 50 | 1 | 1 |
| 1s5p 3P1 | 3.998749 | -- | 2 | -- | -- |
| 1s6p 1P1 | 3.950117 | 2.19e12 | 35 | 0.573 | 0.579 |
| 1s7p 1P1 | 3.921932 | 1.38e12 | 25 | 0.361 | 0.364 |
| 1s8p 1P1 | 3.903850 | 9.21e11 | 20 | 0.241 | 0.244 |

A-values follow n^-3 to ~1%, and electron-impact excitation from the
1s2 ground state scales the same way (proportional to oscillator
strength), so the emitted intensities of the series members are tied
together. 5p cannot be included without 6p and 7p.

Actual Ar16+ detector span (computed, not the plan's implicit line-file
span of 3.9449-4.0732): **3.9154-4.0239 A** over all rows and modules;
3.9406-4.0239 A on the mid-y row; 0.426 mA/pixel. The user's earlier
"3.4-4.1 A" was a typo. Against this span:

| Line | Ritz (A) | On detector | Lands on |
|---|---|---|---|
| S XV 8p | 3.90385 | no | -- |
| S XV 7p | 3.92193 | high-y rows only | clean region shortward of the Li3 satellites |
| S XV 6p | 3.95012 | **yes** | **Ar w (3.9492), +0.92 mA, ~2 pixels** |
| S XVI Lyb 3/2, 1/2 | 3.99080, 3.99194 | yes | k (3.9900), j (3.9941) |
| S XV 5p 1P, 3P | 3.99776, 3.99875 | yes | z (3.9944) red wing |
| S XV 4p 1P, 3P | 4.08850, 4.09055 | no (65 mA past edge) | -- |

Magnitude of the 6p/w blend: a Gaussian contaminant of fractional
intensity p at offset Delta inflates apparent variance by
p(1-p) Delta^2. With Delta = 0.92 mA and sigma(w, 1 keV) = 0.655 mA,
p = 0.03 gives ~+5.7% apparent Ti at 1 keV (+2.9% at 2 keV) and a red
centroid shift of p*Delta. The user reports the 5p feature at 0.5-1.0x
w in startup plasmas, implying 6p at 0.3-0.6x w there -- a first-order
distortion of w, consistent with the user's separate recollection that
w is "anomalously broad in many, but not all" discharges.

n-distribution caveat: electron-impact scaling assumes collisional
excitation dominates. At low Te with high neutral density, radiative
recombination onto S16+ and charge exchange feed high n preferentially
and could flatten 5p:6p:7p. No data currently constrain this.

**Decision:** assume electron-impact excitation; freeze the S XV
series ratios at the NIST A-value ratios; one free amplitude per ion.
Revisit if data constrain the ratio. Include the wider NIST set (4p-8p,
both Lyb components) in the line files; off-detector members are
filtered as for Ar16.

**Decision (P1-a):** amplitude = photon count of one named reference
line, K/ratio = 1.0 for the reference and relative for all others
(same convention as Ar `w_intensity`). References: S XV 1s5p 1P1;
S XVI Lyb 3/2. This replaces the plan's "total ion photons, K sums to
1" definition, which breaks as soon as a wavelength filter removes any
line.

### P2. S XVI Lyb doublet branching -- **Agree** (gA, 2.0015:1)

Excitation from 1s to the 3p 2P fine-structure levels is statistical in
(2J+1) to well under 1% at Z = 16; cascades and radiative recombination
are likewise statistical; the two components have equal decay branching
(A = 1.0949 vs 1.0941e13, and the 3p->2s channel is J-independent).
Opacity is negligible (S density far too low; f(Lyb) ~ 0.079). No
plausible W7-X mechanism produces 1.86. NIST Rel. Int. are rounded
source-dependent estimates; 130:70 is consistent with 2:1 within their
precision.

Does 7% matter? Not for Ti (blend centroid moves 0.02 mA). Possibly for
Te: Lyb1 sits on k and cannot be separated from it by position; Lyb2
(on j) is the only handle, and the frozen ratio converts measured Lyb2
into inferred Lyb1 contamination of k. A ratio error eps becomes a
k-intensity error eps*(Lyb1/k). gA gives ~1%; 1.857 would be a 7%
systematic on the primary Te satellite. Use gA and state in the header
that the ratio is known to ~1%.

Unmodelled and accepted: He-like S satellites to Lyb (1s nl 3p ->
1s2 nl, n >= 3), unresolved just redward of Lyb, few-percent level,
Te^-3/2 exp(-Es/Te) dependence. Note in the header.

**Decision (P2-b):** reference line is Lyb 3/2 alone (K = 1.0,
Lyb 1/2 = 0.4996), consistent with the S XV convention.

### P3. S XV triplet A-value substitution -- **Agree**, with an interpretation caveat

Natural width: gamma(A = 3.82e12, lambda = 3.998 A) = 0.0016 mA versus
sigma_Doppler = 0.33-1.46 mA over Ti = 0.2-4 keV, so gamma/sigma ~ 1e-3
to 5e-3 -- **not ~1e-4 as the plan states** (plan correction). The
physically expected A(3P1 -> ground) ~ 0.04 x A(1P1) would make gamma
smaller still. Conclusion unchanged: immaterial; use the singlet value
and say why.

The 50:2 ratio is an emitted-intensity ratio from NIST's source, not an
A-ratio. In a collisional plasma the 3P1 intensity is set by exchange
excitation (falls relative to singlet excitation as Te rises) times the
branch of 3P1 to ground versus to 1s nl 3S/3D (Delta-n >= 1 E1 rates
~1e10-1e11 at this Z, comparable to the ~1.5e11 intercombination rate).
So it is Te-dependent, plausibly 2-10% over W7-X conditions. Needs data
if it ever matters. It does not for a nuisance line: a 4% component
1.0 mA redward shifts the 5p blend centroid by ~0.04 mA. Freeze 50:2,
state the Te dependence as unmodelled. Same for 4p 3P1 (5:100), which is
off-detector anyway.

### P4. Fixed intra-ion branching via the K column -- **Disagree with K overloading**

The labelling risk is real: the Ar16 header defines `K` as the
dielectronic "Branching Ratio" and `BRANCH_LILIKE` multiplies
`rate * k`. A reader seeing `K = 0.573` on an S XV 6p line would infer
an autoionization branch that does not exist. The plan already leaves
`QD`/`ES` as dead placeholders on STATIC lines; a fourth reused meaning
compounds it.

Adding a column to the `.xc_param` format is awkward: `io/line_file.py`
is positional (13 columns, last four optional strings), so inserting
breaks the Ar16/Ar17/Fe24 files and appending puts a numeric column
after optional strings.

**Decision (P4-a):** the fixed ratio is a *model assumption*
(electron-impact scaling, Te-independent, explicitly provisional), not
an atomic constant. It lives in the per-component spectrum YAML as
`static_ratios: {label: value}` plus `static_reference: label`, next to
the other model assumptions (`intensity_rules`, `li_like_charge_state`).
`build_line_table` fills a new `LineTable.static_factor` array from it;
`BRANCH_STATIC` and the `use_te=False` `intensity_factor` path multiply
`static_factor`, never `k`. The `.xc_param` files stay pure atomic data
with `K = QD = ES = 0.0` on STATIC lines and a header note "unused for
STATIC". Add a guard so `k` cannot reach a STATIC line's intensity by
any path.

### P5. Shared radial emissivity profile -- **Agree** (accepted v1 limitation), magnitude:

Coronal charge-state shells (ionization potentials: S14+ 3.22 keV,
S15+ 3.49 keV; Ar15+ 0.92, Ar16+ 4.12 keV). S XV fractional abundance
peaks near Te ~ 0.6-0.7 keV (half-width ~0.35-1.5 keV); Ar16+ peaks
~1.5-2 keV and dominates from ~1 to >4 keV. S XVI peaks ~1.2-1.5 keV
and tracks the Ar16+ shell closely. Excitation weighting
(exp(-3.1 keV/Te)) pulls both S shells inward by a few hundred eV.

For a W7-X core with Te(0) ~ 2-4 keV, S XV emission is weighted
~0.2-0.4 further out in rho than Ar w, where Ti is 15-30% lower, so the
modelled 5p/6p Doppler width is 15-30% high in Ti (7-15% in sigma).
S XVI: <~10%. In the low-Te startup plasmas where S is bright, all
shells collapse together and the error is small. Transport shifts the
shells (generally inward for low-Z), estimate good to a factor ~1.5.

Spurious-cue risk: low. 6p's width is a mis-specified nuisance, not a
Ti cue; the k/Lyb blend uses the S XVI shell, which matches Ar16+. S is
excluded from `I_tot`, so the Ar-side calibration is unaffected.
Document the 15-30% S XV figure in the plan's limitation note.

### P6. Amplitude definition and normalization -- **Agree**

With P1-a: `s14_ratio` = I(S XV 5p 1P1)/I(w), `s15_ratio` =
I(S XVI Lyb 3/2)/I(w). Unambiguous under any filter; directly measurable;
composes with F015. Excluding S from `I_tot`: correct -- Ar photon
statistics and `verify_ray_calibration` stay identical to prior sets.

Ceiling 1.0 is physical: the user has seen the 5p feature at 0.5-1.0x w
at campaign startup. Not "wildly unphysical". Ray-count consequence at
ceiling: on-detector S XV total ~ 1.0 x (1 + 0.573 + 0.36 partial +
0.04) ~ 1.6-1.9 w-equivalents; S XVI ~ 1.5. Against I_tot/I_w ~ 2.75 at
1 keV this is roughly +60% rays at 1 keV, less at low Te where Ar
satellites dominate. Check `max_rays` headroom as the plan says.

### P7. Generate sulfur-rich, thin downward -- **Agree**, two constraints to record

Bernoulli thinning of a Poisson process with probability p is exactly
Poisson with mean p*mu (thinning theorem); per-ray independent dropping
is exactly that, so per-pixel counts after thinning are exactly Poisson
with the correct reduced mean. Duplication gives variance > mean and is
not Poisson. The plan's claim is correct.

Constraints on the training loop (record in the plan and in the
combined-file description):
1. Any per-sample normalization (total counts, max pixel, etc.) must be
   computed **after** thinning, never cached from the raw sulfur-rich
   file; otherwise every low-sulfur sample is mis-normalized in a way
   correlated with the nuisance.
2. Thinning must select rays by `label` (F038), never by wavelength or
   detector region -- 6p is inside the w line.

### P8. `li_fraction` (F039) interaction -- **Agree** on ordering, sharpened

Separable at generation: S is STATIC, Ar satellites are
`BRANCH_LILIKE`/`DIELECTRONIC`; nothing sulfur-related is redone after
F039. Which is the larger error depends on the target:

- **Te**: F039 is larger. `li_fraction` sets q/r/s/t, comparable to w at
  200 eV-1 keV, versus a contaminant that reaches 1.0 x w only at
  startup and is probably a few percent otherwise.
- **Ti**: F037 is larger. 6p-on-w directly broadens the width
  observable; `li_fraction` touches w's width only through much weaker
  satellite blends.

So: F039 before Te training; F037 (with 6p) before Ti training. The
plan should state both, not just "F039 is larger".

Compensating-error risk: a network trained with frozen `li_fraction`
and correct sulfur could use Lyb2 (on j) as a proxy for missing
satellite Te dependence. Unlikely; full-range sulfur augmentation is
the mitigation and is already in the design.

### Section 3 items

- Charge-state collision: per-component resolution plus
  `(atomic_number, charge_state)` matching is sufficient.
  `_resolve_use_te_branch` is the only place species is inferred from
  charge; `li_like_charge_state` is per-component YAML, so safe.
- Doppler factor: `sqrt(39.948/32.06) = 1.11626` correct;
  `_compute_line_params` scales sigma as 1/sqrt(mass). 32.06 and 39.948
  are standard atomic weights; dominant isotopes (32S 31.972, 40Ar
  39.962) change the factor by 0.12%. Irrelevant, but be consistent
  (both standard weights or both isotopes) and say which.
- `use_te` composition: no case where a STATIC-only component should
  veto. Rule: `use_te_supported` = AND over components that contain any
  non-STATIC line.


## 3. Corrections to plan section 2

1. Line set is incomplete; add S XV 4p, 6p, 7p, 8p singlets and the 4p
   3P1 partner (table in P1). 6p and 7p are on-detector.
2. Natural-width magnitude: gamma/sigma ~ 1e-3 to 5e-3, not ~1e-4.
3. `K` normalization "sums to 1.0 within each ion" is replaced by the
   reference-line convention (P1-a/P2-b); ratios move out of the
   `.xc_param` file into YAML `static_ratios` (P4-a).
4. `LINE_WIDTH` convention: nominally the total upper-level decay rate
   (sum of A_kj), not the single A_ki; for S XVI 3p the 3p->2s channel
   adds ~10%. Immaterial, but state the convention in the headers since
   these are the first hand-built line files.
5. On-detector S line count in the verification list is no longer 4:
   Lyb x2, 5p x2, 6p x1, 7p x1 (rows-dependent) = 5-6, plus off-detector
   4p x2 and 8p x1 present in the table when unfiltered.
6. Doppler factor and mass values are correct; no change.


## 4. Missed physics

1. **S XV 1s6p on the Ar w line** (blocking; P1).
2. **S XV Rydberg-series n-distribution** may not follow electron-impact
   scaling at low Te (recombination, CX). Frozen by decision; flagged
   for future data.
3. **He-like S satellites to Lyb** (n >= 3 spectators), redward of Lyb,
   few percent, Te-dependent. Accepted omission.
4. **Te dependence of the S XV 3P1/1P1 ratio** (exchange excitation and
   competing triplet decay channels). Accepted omission.
5. **Series-member excitation thresholds**: 5p->6p->7p thresholds rise
   by ~37 and ~60 eV, so 6p/5p is ~7% lower at 500 eV than at 3 keV.
   Not modelled; note in the YAML.
6. **Ar17+ channel lead**: S XVI Lyg (1s-4p) scales to ~3.784 A, inside
   the Ar17+ window near Ar Lya (3.731/3.737). Candidate for the
   "unidentified S line" in the future Ar17+ model. Unverified -- check
   NIST.
7. **Testable predictions of the sulfur hypothesis** (research, not
   F037): (a) the two commented-out k-line corrections in
   `2011-04_MZcode_Ar16.xc_param` (3.9900 -> 3.9903 -> 3.9906) are
   consistent with a k + Lyb1 blend rather than a real k shift -- keep k
   at the MZ value and record why in the header; (b) z/k Te vs w/n=3 Te
   vs Thomson disagreement; (c) w-line Ti often ~165 eV high, with a
   predicted red asymmetry. The user has the archive data to test (b)
   and (c); PHA and HEXOS can bound typical post-startup S levels.


## 5. Other

- `2011-04_MZcode_Ar16.xc_param` has the X and Y natural widths swapped
  (X = 2 3P2 -> 1 1S0 is M2, A ~ 3e8; Y = 2 3P1 is the E1
  intercombination line, A ~ 1.7e12; the file has X = 1.673e12,
  Y = 3.212e8). Immaterial to any spectrum; fix opportunistically.
- Mo: traces only on W7-X, Ar17+ system, not diagnostic. Out of scope
  for the Ar16+ channel.
- The `xics_jax` geometry check required the `desc` pyenv; `jax` is not
  on the default interpreter. Worth noting in the plan's verification
  section so the 6p/w test is run in the right environment.


## 6. Decisions log (user)

| Id | Decision |
|---|---|
| P1 | Electron-impact scaling assumed; S XV series ratios frozen at NIST A-value ratios; one amplitude per ion; include 4p-8p in the file |
| P1-a | Amplitude = named reference line; S XV ref = 1s5p 1P1; ratios relative to it |
| P2 | gA-weighted Lyb, 2.0015:1 |
| P2-b | S XVI ref = Lyb 3/2 alone |
| P3 | Singlet A-value stand-in for 3P1; 50:2 frozen |
| P4-a | Ratios in YAML `static_ratios` + `static_reference`; `.xc_param` stays pure atomic data; new `LineTable.static_factor` |
| P5 | Shared profile accepted; 15-30% Ti error for S XV documented |
| P6 | w-normalized amplitudes, S excluded from `I_tot`, ceiling 1.0 |
| P7 | Thin downward by `label`; normalize post-thinning |
| P8 | F039 first for Te, F037 first for Ti; state both |
| Q1-Q3 | w red-asymmetry, S origin/persistence, n-distribution: research items, out of scope |


## 7. Plan changes made as a result

Applied to `plan_F037_multi_species_spectra.md` on 2026-09-04:

- Section 0/1: 6p-on-w motivation; amplitude definition changed to the
  reference-line convention; decisions table updated.
- Section 2.1: full S XV 4p-8p + S XVI Lyb table; detector span stated.
- Section 2.2: ratios re-derived relative to the reference lines and
  moved to YAML; K removed from that role.
- Section 2.3: natural-width magnitude corrected; LINE_WIDTH convention.
- Section 3.1/3.2/3.5: `static_ratios`/`static_reference` YAML keys,
  `LineTable.static_factor`, K guard for STATIC lines.
- Section 4: w_mask, s14/s15 definitions, ray-count estimate.
- Section 7 (out of scope): Ti/Te ordering vs F039; research items.
- Section 8: verification counts and the `desc` pyenv note.
