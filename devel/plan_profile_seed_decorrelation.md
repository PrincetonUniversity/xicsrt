# F021 - Decorrelate randomized spline profile seeds

Status: Planned, not yet implemented.
Plan approved 2026-08-03. See `devel/features_request.md` F021 for the defect
analysis and measured correlation matrix.

This file includes AI generated content using Claude (Opus 5).

## Goal

Make the five randomized spline plasma profiles statistically independent of
each other, so the W7-X ML training-set labels span the intended parameter
space instead of a ~3-dimensional slice of it.

## Scope

In scope: the seed derivation only.

Explicitly OUT of scope (user direction, keep this change focused):
- peak-vs-volume emissivity normalization (157x ray-count spread),
- the Te-dependent I_tot/I_w full-spectrum restoration (26x spread),
- the `te_min` 200 eV cliff and the resulting sample-99 crash,
- the missing `xicsrt.tools.xicsrt_spline` Sphinx apidoc `.rst`,
- regenerating the 1000 configs / 199 samples.

None of the above are to be filed as feature requests at this time.

## Repos touched

- `xicsrt`            - helper, module docstring, tests, feature request.
- `xicsrt_analysis`   - the calling function.

Separate commits; these are separate repos.

## Step 1 - `xicsrt/tools/xicsrt_spline.py`: add `profile_seeds`

    def profile_seeds(seed, names):
        """Derive independent child seeds, one per named profile."""
        parent = np.random.SeedSequence(seed)
        return {name: np.random.SeedSequence(parent.entropy, spawn_key=(ii,))
                for ii, name in enumerate(names)}

NumPy-style docstring; mark as AI generated per the repo directives.

Returns `SeedSequence` objects. Every generator already accepts these unchanged
(verified) because they only call `np.random.default_rng(seed)`. No generator
signature changes.

Resolve `parent.entropy` before spawning children so that `seed=None` behaves
correctly and all five children derive from one resolved parent.

### Why `spawn_key` and not the alternatives

|                                         | spawn_key=(ii,) | SeedSequence.spawn(5) | shared Generator |
|-----------------------------------------|-----------------|-----------------------|------------------|
| stable if a 6th profile is added        | yes             | append-only           | NO               |
| stable if a generator's draw count changes | yes          | yes                   | NO               |
| independence (max cross-|r|, N=500)     | 0.089           | 0.089                 | 0.083            |

A shared `Generator` is fragile for exactly the reason that caused this bug:
any change to one generator's internal draw count shifts every subsequent
profile. `spawn_key` pins each profile to its own named stream.

### Verified properties

- Reproducible for a fixed int seed.
- All five children distinct for seeds 0, 1234, 2**63.
- Zero collisions across 2000 seeds x 5 profiles.
- `seed=None`: children distinct within a call, and different between calls.

## Step 2 - `xicsrt_spline.py` module docstring: add a "Seeding" section

State that generators sharing a seed produce CORRELATED profiles because their
draw sequences align, and that independent profiles require `profile_seeds`.
This is the non-obvious fact that caused the bug and is the single most
valuable thing to document.

## Step 3 - `xicsrt_analysis/w7x_npablant/xicsrt_w7x_npablant.py`

Rewrite `generate_random_profiles` to call
`xicsrt_spline.profile_seeds(seed, _PROFILE_NAMES)` and pass each child seed to
its generator. Keep the public signature `generate_random_profiles(seed)`
unchanged, so `update_config_with_profiles`, `xicsrt_train_task.py:96` and all
notebooks inherit the fix with no call-site edits.

## Step 4 - Physics-constraint note

Add as a code comment in `generate_random_profiles` and as a paragraph in the
`generate_random_temp` docstring:

  Ti and Te are drawn FULLY INDEPENDENTLY. This maximizes label coverage but
  admits combinations that are not physical for a given heating scheme. With
  the default ranges roughly 24% of samples get Ti > Te and 3.3% get
  Ti > 5*Te. A more physics-constrained formulation would couple the two
  profiles and enforce Te >= Ti, which is the expected ordering for Electron
  Cyclotron Resonance Heating, where power is deposited on the electrons and
  reaches the ions only through collisional transfer. Other heating schemes
  (NBI, ICRH) are less clear-cut and can drive Ti > Te. Deliberately not
  imposed here.

Percentages are measured, not estimated.

Context worth keeping in mind: the pre-fix buggy code happened to satisfy
Te >= Ti in 500/500 seeds (Ti/Te fixed at ~0.54). The fix deliberately trades
an accidentally-physical but zero-information coupling for full coverage that
includes unphysical corners.

## Step 5 - Regression tests, appended to `tests/test_spline.py`

No new test file; one source module, one test module.

1. `test_profile_seeds_are_distinct` - deterministic.
   - the five children give distinct draws,
   - a fixed int seed reproduces exactly,
   - same-seed generators DO alias (pins the library contract that motivates
     the helper's existence).

2. `test_profile_seeds_decorrelate_profiles` - statistical.
   - over N=500 seeds build the ~10 scalar labels
     (x_knots[1] and y_knots[0] for each of the five profiles),
   - assert max cross-profile |r| < 0.20.

   Measured max is 0.089; null 3-sigma is 0.135, giving ~4.5 sigma of margin.
   The current bug at r = 1.0 would fail loudly.

   CRITICAL: exclude SAME-profile pairs. Ordered knots within one profile
   legitimately correlate - `emiss_x1`/`emiss_x3` reaches 0.29 - so a naive
   all-pairs check fails spuriously.

## Verification

- `pytest tests/` in the `xicsrt` repo (108 tests currently pass; expect 110).
- Re-measure the correlation matrix through `generate_random_profiles` and
  confirm both rank-1 blocks are gone (expect max cross-profile |r| ~ 0.09,
  down from 1.000).

## Versioning

Pure addition, no signature changes, so no `_version.py` minor bump is required
under the documented major/minor/revision semantics.

## Consequences

Profiles change for every seed. The 199 existing samples in
`/u/npablant/analysis/xicsrt_results/w7x_ar16_ml_training/` and the 1000 saved
configs become inconsistent with newly generated ones. The user has explicitly
accepted this: correct statistics for new runs matter, previous runs do not.
