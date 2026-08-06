# Plan F010: W7-X ML training-set acceleration

Status: approved by user 2026-07-30, not yet started.
Tracking: F010 in devel/features_request.md.

This plan is self-contained: a fresh agent session should be able to implement
it starting from this file alone. Instruct the agent:
"Implement F010 per devel/plans/plan_F010_w7x_training_accel.md, starting with Phase 1a."

## Objective

Generate 10,000 independent W7-X XICS training images, ~1e6 detected counts
each, on Princeton Stellar (96 cores/node, up to 4096 cores CPU-only, 24-48 h).
Each image varies plasma profiles (temp/emissivity/velocity) only; geometry and
equilibrium fixed. Requires ~2x speedup vs baseline. Exact per-image photon
statistics are mandatory (reweighting/importance-sampling was considered and
rejected by user). Readability is priority one; speed second.

## Baseline

Stellar run_08: 12 workers x 8 threads, 96 cores, wall 6m45s, 2.1e9 generated /
5.1e5 detected rays => ~21 core-h per 1e6-count image.

Worker core-time breakdown (~78 min total):

- generate_wavelength (ar16_voigt per-bundle CDF tables):  19m (24%)
- optics tracing (Dispatcher: raytrace):                   31m (39%)
- generate_direction + generate_origin:                    12m (15%)
- DESC map_coordinates ("Fluxspace from Realspace"):       11m (14%)
- bundle loop overhead + collection:                       ~5m  (6%)

Reference runs: /u/npablant/remote/stellar_princeton_edu/scratch/xicsrt/w7x
(run_08 = best CPU; run_11 = 96x1 layout, worse; run_12 = GPU+MPS hybrid, no
benefit because generation dominates and one A100 is shared by 12 workers).

Model entry point: xicsrt_analysis/w7x_npablant/xicsrt_w7x_npablant.py
(XicsrtPlasmaW7xSimple -> XicsrtPlasmaVmec (xicsrt_contrib) ->
XicsrtPlasmaGeneric (xicsrt)).

## Phase 1 - numpy-path acceleration (implement)

### 1a. Direct Voigt sampling (xicsrt/tools/)

A Voigt variate is exactly center + Normal(0, sigma) + Cauchy(0, gamma);
a multi-Voigt spectrum is a mixture: draw line index from intensity weights
(searchsorted on normalized cumulative intensities), then that line's variate.
This eliminates per-bundle CDF grid construction entirely (currently 184 lines
x 8192 grid points of Faddeeva evaluations per bundle x ~156k bundles) and is
MORE exact than the table (no truncation at `cutoff`, no interpolation error).

- xicsrt_voigt.py: rewrite voigt_random() as direct sampling. Keep voigt(),
  voigt_cdf_tab() (used by jaxrt setup / pdf evaluation).
- xicsrt_voigt_multi.py: rewrite multi_voigt_random() as vectorized mixture
  sampling; add a batched form taking per-bundle line-parameter arrays
  (n_bundles, n_lines) plus a per-ray bundle index so all bundles sample in
  one call. Keep multi_voigt(), multi_voigt_cdf_tab(). Remove gridsize/cutoff
  from sampler signatures; update all callers (no compat shims).
- Delete xicsrt_voigt_multi_jax.py, xicsrt_faddeeva_jax.py (verify no other
  users first), tests/test_voigt_multi_jax.py, and the _USE_JAX_VOIGT_MULTI
  toggle in sources/_XicsrtSourceGeneric.py:41. Note obsolescence under F004.
- NumPy-style docstrings (Sphinx-rendered) documenting the decomposition and
  exactness. Note: results not bit-identical to old runs for a given seed
  (statistically identical).

### 1b. Vectorized create_sources (xicsrt/sources/_XicsrtPlasmaGeneric.py)

Clean redesign (approved): replace the per-bundle XicsrtSourceFocused
instantiation loop (create_sources, ~line 294) with batched generation:

- np.random.poisson vectorized over per-bundle intensities (exact counts),
  bundle_index = np.repeat(arange(n_bundles), counts).
- Origins vectorized (point bundles: repeat bundle origins; voxel: add
  uniform offsets).
- Directions: per-ray focused normal (target - origin, normalized) + cone
  sampling vectorized over per-ray spread. Requires vectorizing
  xicsrt_spread.vector_dist_isotropic and solid_angle over array-valued
  spread (removes loop at _XicsrtPlasmaGeneric.py:233). Other distributions
  keep scalar signatures.
- Wavelengths via 1a batched sampler (per-bundle sigma from temperature, or
  per-bundle line tables via the 1c hook); Doppler shift vectorized with
  per-bundle velocity.
- XicsrtSourceGeneric/Focused unchanged as standalone sources.
- Update all plasma subclasses in the same change: Toroidal,
  ToroidalDatafile, Cubic, Cylindrical (xicsrt) + XicsrtPlasmaVmec
  (xicsrt_contrib) + XicsrtPlasmaW7xSimple (xicsrt_analysis/w7x_npablant).

### 1c. Relocate ar16_voigt out of public xicsrt (approved for Phase 1)

- Remove the xics_jax import (line 31) and random_wavelength_ar16_voigt from
  sources/_XicsrtSourceGeneric.py; drop 'ar16_voigt' from public
  wavelength_dist options.
- Add a plasma-source hook, e.g. get_line_parameters(bundle_input) returning
  per-bundle (n_lines,) location/intensity/sigma/gamma arrays, consumed by
  the batched multi_voigt sampler.
- Implement the Ar16+ case as an override in w7x_npablant that calls
  xics_jax.evaluate_lines batched (vmap) over all bundle (Ti, Te) pairs at
  once - one call per iteration instead of one per bundle.
- This also removes the F009 (xics_jax import perturbs global RNG) exposure
  from the public repo.

### 1d. DESC mapping hygiene (xicsrt_contrib/_XicsrtPlasmaVmec.py)

Keep exact Newton root-finding (user rejected rho interpolation tables).

- Cache the loaded equilibrium across bundle_generate calls (currently
  initialize_vmec reloads the file every iteration, line ~160).
- Investigate/fix per-call jax recompilation of eq.map_coordinates caused by
  varying masked-point counts: pad inputs to fixed bundle_count shape.
  Target: 13.5 s/call -> sub-second warm.

### jaxrt sync (per devel/jaxrt_sync.md)

Mirror direct sampling into jaxrt/tools/_wavelength.py (jax.random.normal +
jax.random.cauchy; removes the numpy cdf_tab live-coupling point listed at
jaxrt_sync.md:70) if trivial; otherwise record divergence in the log.
Plasma sources remain out of jaxrt scope.

### Housekeeping

- AI-disclaimer header on every modified python file.
- Update F004 notes (jax voigt sampler deleted as obsolete).
- Minor version bump in xicsrt/_version.py at completion.

### Verification gate (Phase 1)

1. New statistical-equivalence tests (tests/, alongside test_voigt.py):
   KS / chi-square of the direct sampler against (a) the analytic multi_voigt
   pdf and (b) large samples from the existing CDF sampler (before deleting
   it, or against a pinned reference sample), at production Ar16+ scale
   (~184 lines) and single-line Voigt. Tolerances must account for the CDF
   table's own truncation/interpolation error.
2. End-to-end 5-sigma statistical comparison (detected counts, image moments)
   old vs new on the w7x config, separate processes per engine variant
   (see F007 warm-allocator caveat).
3. pytest tests/ (incl. tests/jaxrt), examples/example_00/example_00.py.
4. Stellar re-profile with run_08 layout; re-tune MP split (try 24x4, 48x2)
   since the generation/tracing balance shifts toward tracing.

## Phase 2 - production template (implement, template only)

One SLURM job-array batch script + minimal per-task python driver, in the
style of the existing run_NN directories: SLURM_ARRAY_TASK_ID -> distinct
random_seed + output suffix -> N images per task -> per-image hdf5/tif.
Node layout from the Phase 1 re-tune. The user's separate generation
framework owns parameter sweeps; do not build sweep/combining tooling.

## Phase 3 - hybrid GPU (document only, DO NOT implement)

Write devel/plans/plan_F010_hybrid_gpu.md describing: numpy plasma generation -> padded
fixed-capacity ray handoff -> jaxrt jit'd optics chain on GPU (jaxrt's
_build_trace boundary already separates generation from tracing). Include:
Amdahl ceiling ~2-2.4x/node post-Phase-1 (tracing ~60% share); fleet
economics (GPU jobs capped at 2 nodes vs 42+ CPU nodes); prerequisite first
real A100 benchmark via testing/benchmark_jaxrt.py; statistics preserved via
capacity+mask (no approximation). Gated on Phase 1 production experience.

## Out of scope / decided against

- Reweighting / importance sampling across images (violates exact per-image
  statistics; user rejected).
- rho(x,y,z) interpolation table replacing DESC mapping (user rejected).
- Full jaxrt port of VMEC/W7-X plasma sources (sources still in flux;
  ar16_voigt and VMEC source must stay out of the public codebase).
- Velocity-profile port (stelltools -> DESC) deferred; first training set
  uses enable_velocity=False.
- Multi-node MPI; use SLURM job arrays instead.

## Order of execution

1. Implement 1a + statistical-equivalence tests (tests must pass against the
   CDF sampler BEFORE it is deleted).
2. Implement 1b, then 1c (touches xicsrt, xicsrt_contrib,
   xicsrt_analysis/w7x_npablant), then 1d (xicsrt_contrib).
3. Run the verification gate; Stellar re-profile.
4. Phase 2 template; Phase 3 design note.
5. Session close: append to devel/devel_ai_log.txt; ask user before marking
   F010 done; commit-message approval gate applies (see AGENTS.md).
