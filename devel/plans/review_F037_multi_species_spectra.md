# Review instructions - F037 multi-species composite spectra

Companion to `plan_F037_multi_species_spectra.md`. This file tells a
reviewing agent what to review, what to skip, and what a useful answer
looks like.

**Nothing has been implemented yet.** This is a review of a plan, not of
code. Do not write or modify code.


## 0. What this feature does, in one paragraph

Real W7-X Ar16+ spectrometer data contains two contaminating sulfur
features near 3.98-4.00 A (four NIST lines, two unresolved doublets)
that overlap the Li-like satellites and the He-like z line. They must be
added to the ML training set. Rather than special-casing sulfur, the plan
generalizes the `xics_jax` forward model to composite multi-species
spectra, which is also required for a future Ar17+ model containing
Fe24+, Mo32+ and a further sulfur line.


## 1. How to review efficiently

Read in this order. Budget most of your effort on section 2.

1. `devel/plans/plan_F037_multi_species_spectra.md` -- the plan. Sections
   1-2 carry the physics decisions.
2. `devel/features_request.md`, entries F037, F038, F039.
3. Only if needed for a specific question, these source files:
   - `xics_jax/xics_jax/model/lines.py` -- `_resolve_use_te_branch`,
     `build_line_table`, the `BRANCH_*` constants
   - `xics_jax/xics_jax/model/excitation.py` --
     `compute_use_te_intensities`, the four existing branches
   - `xics_jax/xics_jax/model/forward.py` -- `_compute_line_params`,
     the Doppler-width expression
   - `xics_jax/xics_jax/model/registry.py` -- `SpectrumConfig`
   - `xics_jax/xics_jax/data/lines/2011-04_MZcode_Ar16.xc_param` --
     the existing format and header conventions (header only, ~60 lines)
   - `xicsrt_analysis/w7x_npablant/sources/_XicsrtPlasmaW7x.py` --
     `_eval_line_model`, `_full_spectrum_factor`, `shape_integral`

Repository roots:
- `/Volumes/Data HD/u/npablant/code/mirproject/xics_jax`
- `/Volumes/Data HD/u/npablant/code/mirproject/xicsrt_analysis`
- `/Volumes/Data HD/u/npablant/code/mirproject/xicsrt_ml`

You do not need to read the ML training code, the DESC geometry code, the
raytracing core, or the SLURM production scripts.


## 2. Physics decisions -- REVIEW THESE

These are the points where the plan could be wrong in a way that testing
would not catch. Please address each explicitly, and say plainly when you
think a choice is wrong.

### P1. Atomic data selection and accuracy (highest priority)

The four NIST lines and their parameters are tabulated in plan section
2.1. Please verify independently:

- Are these the correct and complete set of sulfur lines in 3.98-4.00 A
  that would be visible in a W7-X Ar16+ spectrum? Are any comparably
  strong S lines (or lines of other plasma-facing-component or
  intrinsic impurities) omitted?
- Are the Ritz wavelengths and A-coefficients correctly transcribed?
- Is preferring Ritz over observed wavelengths correct here?
- S XV 1s5p is an n=5 transition. Is a bare 1s2-1s5p treatment adequate,
  or are there unresolved n=5 satellites or blends that matter at this
  resolution?

### P2. S XVI doublet branching: gA vs observed

The plan uses gA-weighting (2.00146:1) rather than the NIST tabulated
relative intensities (130:70 = 1.857:1), on the grounds that `Rel. Int.`
values are rounded observational estimates while gA is the statistical
result expected for a Lyman-beta fine-structure doublet.

- Is that reasoning sound for H-like Ly-beta in a W7-X-like plasma?
- Could the observed 1.857 reflect something real (opacity, a
  non-statistical upper-level population, blending) rather than
  measurement scatter?
- Is the 7% difference large enough to matter for a network learning
  line ratios in this region?

### P3. S XV triplet A-value substitution

NIST publishes no A_ki for the S XV 1s2 1S_0 - 1s5p 3P_1 line at
3.998749 A. The plan sets its `LINE_WIDTH` to the singlet's 3.82e12 and
argues the natural width is ~1e-4 of the Doppler width so the choice is
immaterial.

- Confirm or refute that magnitude estimate at W7-X Ti (roughly
  0.2-4 keV).
- Is the 50:2 relative-intensity ratio for this intercombination line
  credible, and is it expected to be Te- or ne-sensitive? If it is, a
  fixed `K` may be the wrong model.

### P4. Fixed intra-ion branching via the `K` column

The plan collapses each sulfur ion to a single free amplitude, with the
internal doublet ratio frozen in the `K` column.

- Is freezing the intra-ion ratios correct, i.e. are they genuinely
  Te/ne-independent over the W7-X operating range?
- The `K` column currently means "branching ratio" for dielectronic
  satellites. Is overloading it for a different quantity on `STATIC`
  lines acceptable, or should a separate column be added? (This is
  partly a design question, but the physics-labelling risk is real:
  someone could later apply dielectronic logic to these values.)

### P5. Shared radial emissivity profile (accepted limitation - sanity-check the magnitude)

There is only one emissivity profile in the model, so the sulfur lines
inherit the Ar16+ w-line radial shape. Physically S XVI (H-like) peaks
further into the core and S XV further out. The S lines therefore get an
Ar-weighted Ti along the line of sight rather than their own
emission-weighted Ti.

The user has accepted this for v1. Please do not re-argue the decision;
instead estimate the size of the error:

- Roughly how different are the S XV / S XVI and Ar16+ emission shells
  for a W7-X plasma?
- How large an apparent Ti error does that produce for the sulfur lines?
- Could a network plausibly latch onto this as a spurious Ti cue, given
  the S lines sit on top of the Ar satellites?

### P6. Amplitude definition and normalization convention

Sulfur amplitude is defined as (total ion photons)/(Ar16+ w-line
photons), fixed at 1.0, and sulfur is deliberately **excluded** from the
`I_tot/I_w` emissivity calibration so that Ar16+ photon statistics stay
identical to existing training sets (plan sections 1 and 4.1).

- Is normalizing to the w line the right convention, given the existing
  F015 w-line emissivity definition?
- Is excluding sulfur from the calibration right, or does holding total
  ray count fixed matter more?
- Sanity-check the ratio 1.0: that means sulfur emits as many photons as
  the Ar16+ w line. Combined with the observation that these are
  "contaminating" lines, is 1.0 a plausible upper bound, or is it far
  outside anything physical? (It is intentionally an upper bound for
  augmentation headroom, not a typical value -- but if it is wildly
  unphysical, say so, since augmentation cannot fix a training set whose
  ceiling is unreachable.)

### P7. Dataset design: generate sulfur-rich, augment downward

Because ray-dropping augmentation preserves Poisson statistics only in
the downward direction, every sample is generated at maximum sulfur and
varied downward in the training loop.

- Is the claim correct that per-ray Bernoulli dropping preserves exact
  Poisson statistics, while upward duplication does not?
- Does generating a deliberately unphysical (sulfur-rich) raw dataset
  create problems beyond the need to always augment? Consider in
  particular that the Ar16+ rays are unaffected but the *total* detector
  count, and hence any global normalization or noise characteristic the
  network sees, is not.

### P8. `li_fraction` (F039) interaction

F039 records that `li_fraction` is frozen at 0.1 and Te-independent,
making the satellite/resonance ratio wrong in a Te-dependent way. The S
lines sit directly on those satellites.

- Is it sound to add sulfur before fixing `li_fraction`, or does the
  overlap mean the two must be fixed together to avoid the network
  learning a compensating error?
- Is the plan's claim right that F039 is the larger error?


## 3. Physics-adjacent correctness -- worth a look

Not judgement calls, but places where a wrong assumption would be
subtle:

- **Charge-state collision (plan 3.4).** S XVI is charge state 15; so is
  Li-like argon, which `_resolve_use_te_branch` keys `BRANCH_LILIKE` off.
  The plan defends with per-component branch resolution plus an
  `atomic_number` column. Is that sufficient? Any other place where
  species is inferred from charge state alone?
- **Doppler width factor.** `sqrt(39.948/32.06) = 1.11626`. Confirm the
  masses and that the expression in `_compute_line_params` scales as
  `1/sqrt(mass)`.
- **`use_te` composition.** `STATIC`-only components have no excitation
  rate data. The plan says they must not veto `use_te` for the composite.
  Is there a case where a component genuinely should?


## 4. Explicitly NOT under review

Do not spend effort on these; they are settled or are routine
engineering:

- Dataclass/field layout, naming, file organization, caching strategy.
- The `element:label` prefixing scheme and parameter-index offsetting.
- Whether the breaking `SpectrumConfig.atomic_mass` change is acceptable
  (the user has approved it; the project mandates no backwards
  compatibility, no shims, no deprecated aliases).
- The F038 label-plumbing mechanics (`-1` padding, schema fields).
- Test framework choice, file naming, docstring style, AI disclaimers.
- Deleting the inert `mass_number`/`wavelength`/`linewidth` options
  (verified inert by code inspection).
- Whether to use `s14`/`s15` naming (matches the existing charge-number
  convention).

If you spot an outright bug in a non-reviewed area, mention it briefly at
the end, but do not let it displace section 2.


## 5. Output format

Please structure your response as:

1. **Verdict** -- one paragraph: is the physics sound enough to
   implement, and are there any blocking errors?
2. **Per-item findings** -- P1 through P8, each marked
   `Agree` / `Disagree` / `Needs data`, with reasoning. Cite sources for
   any atomic-data claim.
3. **Corrections** -- any numbers in plan section 2 that are wrong, with
   the correct values.
4. **Missed physics** -- anything the plan does not consider at all.
5. **Other** -- brief, optional.

Be direct. If a decision is wrong, say so and say why; do not soften it.
If you are uncertain, say "needs data" rather than guessing -- a
confident wrong answer about an A-coefficient is worse than an
acknowledged gap. Prefer citing NIST ASD, published references, or the
repository's own vendored data over recollection.
