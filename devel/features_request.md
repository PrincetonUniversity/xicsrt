# XICSRT Feature Requests

## F012 - Randomize found-ray order in the history (`shuffle_history`)
Started: 2026-07-30
Status: Done (2026-07-30)

Plasma sources emit rays in contiguous per-bundle blocks
(`XicsrtPlasmaGeneric.create_sources`: `bundle_index = np.repeat(np.arange(len(counts)), counts)`),
and this block order survives unshuffled through `_sort_raytrace` and
`combine_raytrace` into the returned `found` history. A user taking a naive
subset (e.g. `history['detector']['origin'][:1000]`) would get rays from only
a handful of bundles instead of a statistically representative sample of the
plasma.

Implementation: new `general.shuffle_history` option (default `True`).
`_sort_raytrace` permutes `w_found` (the found-ray index array) with a
dedicated `rng_shuffle` generator, independent from the existing `rng` used
for lost-ray subsampling, so enabling/disabling the shuffle cannot change
which lost rays are retained and the shuffle cannot perturb the global
`np.random` stream that generates the rays. The `lost` rays are already
unordered (`rng.choice(..., replace=False)`) and are not reshuffled.
`raytrace_single` derives `rng_shuffle` via `np.random.SeedSequence(seed,
spawn_key=(1,))`, keeping `rng_lost`'s stream byte-identical to before this
change. `xicsrt/jaxrt/_engine.py` mirrors the same generator setup since it
imports `_sort_raytrace` directly.

Shuffling is done at the found-ray-selection stage, not in the source: a
benchmark of shuffling inside `XicsrtPlasmaGeneric.create_sources` (or the
`bundle_index` array feeding it) cost 13-31% of total raytrace time for the
'voigt' wavelength distribution, and >100% for 'multi_voigt' (random gather
into the per-bundle line table destroys locality). Shuffling only the found
rays in `_sort_raytrace` costs approximately 0.005% of raytrace time at a
typical x-ray efficiency (found << traced), rising to ~2% only when a large
fraction of traced rays are found. Cross-iteration/run shuffling in
`combine_raytrace` was also rejected: each iteration is already an unbiased
sample of the same config, so within-iteration shuffling alone makes any
prefix of the found history a fair sample.

New tests: `tests/test_history_shuffle.py` (unit-level `_sort_raytrace`
behavior on synthetic bundle-blocked data, plus an end-to-end check with a
real point-bundle plasma source). `testing/compare_raytrace_regression.py`
sets `shuffle_history=False` in its baseline scenario (for the Tier A exact
comparison against pre-F012 source trees) and `_call_sort` now explicitly
disables shuffling for Tier B, which tests lost-ray subsampling and is
orthogonal to this feature.

No bundle id is stored in the ray arrays, so shuffling is currently the only
way to lose track of bundle membership; recovering it (if ever needed) would
require a separate per-ray bundle-index feature, deliberately left out of
this change.


## F011 - XicsrtPlasmaCubic ignores the `velocity` option (no Doppler shift)
Started: 2026-07-30
Status: Pending (pre-existing bug, found while verifying F010 addendum)

`XicsrtPlasmaGeneric.create_sources` applies a Doppler shift from
`bundle_input['velocity']`, but `XicsrtPlasmaCubic.bundle_generate` only
fills `temperature` and `emissivity`; it never copies `self.param['velocity']`
into `bundle_input['velocity']`. The array therefore stays at its zero
initialization and the configured `velocity` is silently ignored.

Reproduce (monochrome line, temperature 0, so the shift is unambiguous):

    velocity = [0, 0, 3e6] -> mean wavelength shift 0.0 (expected ~ -0.0395 A)

`XicsrtPlasmaToroidal` does this correctly (`bundle_input['velocity'][m] = ...`),
as does the new `XicsrtPlasmaBundleSource`, so the bug is specific to
`XicsrtPlasmaCubic`. Verified present on 6a94671 (pre-dates the F010
addendum work). Fix is a one-line addition to `bundle_generate`, but it
changes results for any existing `XicsrtPlasmaCubic` config that sets
`velocity`, so it is left for explicit approval rather than folded into an
unrelated change.


## F010 - W7-X ML training-set acceleration (numpy path)
Started: 2026-07-30
Status: Phase 1 implemented 2026-07-30 (verification notes below); Phases 2-3
pending.
Plan: devel/plan_w7x_training_accel.md

Addendum (2026-07-30, Done): Phase 1b's vectorization of
`XicsrtPlasmaGeneric.create_sources` removed the ability to model each
bundle with an arbitrary, user-selectable ray source (the old code
instantiated a fresh `XicsrtSourceFocused` per bundle). This capability was
reintroduced as a new example class, `XicsrtPlasmaBundleSource`
(`xicsrt/sources/_XicsrtPlasmaBundleSource.py`), which loops over bundles
and dispatches a ray source chosen by the config option
`bundle_source_class` (resolved through the dispatcher's plugin search
paths, so a user's own source class works). `XicsrtPlasmaGeneric` remains
the fast, production default; the new class is documented as a worked
example, not a performance-equivalent replacement.

Enabling this required a small framework change: elements previously had
no way to see the plugin search paths used to find them (`general.pathlist`
+ `general.pathlist_default`), since only their own element-level config is
passed down. `Dispatcher._instantiate_single` now sets
`obj.param['pathlist']` on every element it constructs, and
`ConfigObject.__init__` seeds a builtin-only default so directly
constructed elements (e.g. via `xicsrt_public.get_element`) still work. The
class lookup half of `_instantiate_single` was factored out into
module-level `xicsrt.objects._Dispatcher.find_xicsrt_class`, reused by the
new plasma class. `pathlist` is a `param`-only key (never added to
`default_config`) so it is never written into a saved config file, which
would leak machine-specific absolute paths.

Phase 1 implementation notes (2026-07-30):
- 1a: voigt_random / multi_voigt_random rewritten as exact direct sampling
  (Normal + Cauchy; mixture by intensity weights). New batched sampler
  multi_voigt_random_batched (per-bundle line tables, one call per iteration).
  ~108x faster than the per-bundle CDF-table build at Ar16+ scale, and more
  exact (no tail truncation at `cutoff`, no interpolation error). Deleted
  xicsrt_voigt_multi_jax.py, xicsrt_faddeeva_jax.py, tests/test_voigt_multi_jax.py,
  _USE_JAX_VOIGT_MULTI. Removed multi_gridsize/multi_cutoff config options.
  New statistical tests: tests/test_voigt_direct.py (KS vs analytic pdf; KS vs
  the retained CDF tables with tolerances above the tables' own ~0.2-0.6%
  truncation bias). jaxrt _wavelength.py mirrored to direct sampling.
- 1b: XicsrtPlasmaGeneric.create_sources vectorized: per-bundle Poisson draws,
  bundle_index = repeat(arange, counts), vectorized origins/directions/
  wavelengths/Doppler. XicsrtSourceFocused no longer instantiated per bundle.
  vector_dist_isotropic and solid_angle_isotropic accept array spread.
  Verified 5-sigma statistical equivalence vs baseline (testing/
  compare_f010_plasma.py: generated 0.55 sigma, detected 0.66 sigma,
  centroids <0.05 px over 5 seeds).
- 1c: ar16_voigt and the xics_jax import removed from public xicsrt.
  New hook XicsrtPlasmaGeneric.get_line_parameters (default: broadcast static
  line_* config); XicsrtPlasmaW7xSimple overrides it with a vmapped+jit'd
  xics_jax._compute_line_params over all bundle (Ti, Te) pairs (one call per
  iteration). W7X config now uses wavelength_dist='multi_voigt'. Also removes
  the F009 xics_jax-import exposure from the public repo (F009 itself still
  open for the analysis repo).
- 1d: XicsrtPlasmaVmec caches the loaded DESC equilibrium across iterations;
  map_coordinates now always called with fixed bundle_count-shaped arrays
  (masked rows padded) to avoid jax recompilation on masked-count changes.
- Measured: W7X model (10k bundles, 3 iter, single process, M1) wall
  138.8s -> 27.4s (5.1x). Direct sampling changes per-seed results
  (statistically identical, not bit-identical).
- Version bump 0.8.13 -> 0.9.0 (config options removed; behavior change).

Goal: generate 10,000 W7-X training images (~1e6 detected counts each) on the
Princeton Stellar cluster within a ~4096-core x 24-48 h envelope. Requires
roughly 2x end-to-end speedup of the numpy engine plasma path relative to the
Stellar run_08 baseline (12 workers x 8 threads, 21 core-h per 1e6-count image).

Scope (Phase 1, implement):
1a. Direct Voigt sampling: replace CDF-table sampling in voigt_random /
    multi_voigt_random with exact Normal+Cauchy mixture sampling; delete
    xicsrt_voigt_multi_jax.py / xicsrt_faddeeva_jax.py (obsoletes F004 sampler).
1b. Vectorized create_sources in XicsrtPlasmaGeneric (remove per-bundle
    XicsrtSourceFocused loop); vectorize xicsrt_spread over array spread.
1c. Relocate ar16_voigt / xics_jax out of public xicsrt into w7x_npablant via
    a per-bundle line-parameter hook (also resolves F009 exposure).
1d. DESC equilibrium caching + fixed-shape map_coordinates (xicsrt_contrib).
Phase 2 (implement): SLURM job-array production template.
Phase 3 (document only): hybrid numpy-generation -> jaxrt GPU optics design
    note in devel/plan_hybrid_gpu.md.

Constraints: exact photon statistics (no reweighting; per-image independent
sampling), readability first, no backwards-compat shims, jaxrt sync per
devel/jaxrt_sync.md, minor version bump on completion.


## F009 - `xics_jax` import consumes global RNG stream on first use in a process
Started: 2026-07-28
Status: Pending

`import xics_jax` at module scope in `sources/_XicsrtSourceGeneric.py:30` draws
from the global `np.random` stream as a side effect of the import itself:

    np.random.seed(12345) -> stream position 624
    import xics_jax       -> stream position 10

The plugin dispatcher imports element modules lazily on the first
`instantiate()`, so that draw lands *between* `np.random.seed()` and ray
generation on the first raytrace in a process, but not on later ones. Two
identical `raytrace()` calls in one process therefore give different results
(measured: nfound = 27, then 29).

Consequences:
- `random_seed` does not currently guarantee reproducibility, contrary to its
  documentation in `xicsrt_config.default_config`.
- Any A/B comparison must import `xics_jax` up front or run each case in a
  fresh process, or the confounder swamps the signal.

Suggested fix: save and restore the global `np.random` state around the
`xics_jax` import, or move the import so it cannot occur after seeding.


## F008 - Raytrace not reproducible on the first run of a process (seed 0)
Started: 2026-07-28
Status: Pending

Discovered incidentally while validating F007; NOT caused by F007 (reproduced
on clean HEAD with no modifications).

Symptom: with `random_seed=0`, running the same config three times in a single
process gives detector counts 41456, 41661, 41661. Run 1 differs from runs 2
and 3; runs 2+ are stable. Seeds 1-7 are stable from the first run.

Implication: the first raytrace in a fresh process does not see the configured
seed state, so a single-shot script is not reproducible against a repeated one.
This silently undermines any "set random_seed for reproducibility" workflow and
would also affect multiprocessing workers, which each run exactly one first run.

Suspected cause: seed application order in `xicsrt_raytrace` / config setup,
where `0` may be being treated as falsy somewhere, or the global `np.random`
stream is touched before seeding. Not yet diagnosed.

Next step: bisect where the global RNG is first consumed relative to seeding.


## F007 - Acceleration of einsum and vector operations (numpy engine)
Started: 2026-07-28
Status: Implemented, pending review
Baseline commit: 7390575

Goal: reduce single-core cost of the numpy raytracing hot path.

IMPORTANT framing correction. This entry was opened on the premise that
`np.einsum` is not parallelized over CPUs and that switching to BLAS calls
would recover multicore scaling. That premise is wrong and was disproved by
measurement:

- Every hot array in XICSRT has shape (N,3). Such operations are
  memory-bandwidth-bound, not FLOP-bound.
- `a @ M` for a (2e6,3) @ (3,3) takes 16.5 ms on 1 thread and 16.4 ms on 10
  threads, i.e. zero scaling, while a 2000^3 gemm in the same process scales
  59.8 -> 31.3 ms. The BLAS alternative is therefore just as serial as einsum.
- This independently corroborates the F006 finding that einsum does not use
  threaded BLAS, and extends it: the BLAS replacement does not either.

Cores must therefore continue to come from multiprocessing over runs (F006).
What this entry actually buys is single-core efficiency.

Key corollary: `np.einsum('ij,ij->i', a, b)` is the FASTEST available row-wise
dot product (8.0 ms vs 15.9 ms for `(a*b).sum(axis=1)` at N=2e6). Those call
sites are already optimal. This is a targeted change, not an einsum purge;
"de-einsumming" the row-wise dots would be a ~2x regression.

Measured hot spots (1e7 rays, spherical crystal, keep_history=False; c_einsum
itself is only ~7% of runtime):

| operation                                   | current  | replacement | gain |
|---------------------------------------------|----------|-------------|------|
| `np.linalg.norm(a, axis=1)`                  | 14.1 ms  | 7.5 ms      | 1.9x |
| `einsum('ij,ki->kj', M, a)`                  | 28.8 ms  | 16.0 ms     | 1.8x |
| `einsum('ij,ijk->ik')` + (N,3,3) assembly    | 58.3 ms  | 19.8 ms     | 2.9x |

Planned work (hot path only, by user decision):
1. `tools/xicsrt_math.py` `magnitude`/`normalize`: `np.linalg.norm(v, axis=1)`
   -> `np.sqrt(np.einsum('ij,ij->i', v, v))`, with a comment.
2. `objects/_GeometryObject.py` `vector_to_external`/`vector_to_local`:
   einsum -> `vector @ orientation` / `vector @ orientation.T`. Preserve the
   existing `copy=` handling and the `vector[:]` in-place writeback.
3. `sources/_XicsrtSourceGeneric.py` `random_direction`: drop the (N,3,3)
   rotation-matrix buffer in favour of the component-sum form already used by
   the jax engine. Single code path retained.
4. `make_normal` in `_XicsrtSourceGeneric.py` and `_XicsrtSourceDirected.py`:
   normalize the constant axis once instead of N times. Both files need the
   edit; the dispatcher loads element modules by file path, so patching the
   base class alone does not cover the override.

Out of scope: all `einsum('ij,ij->i')` row-wise dots (already optimal); mesh,
torus, cylinder, mosaic, xicsrt_spread and filter call sites; the
`location_from_distance` masked-gather rewrite (71 -> 33 ms via `where=`,
real but a masking-semantics change rather than an einsum one -- candidate for
its own entry).

Measured result (all four changes applied).

Per-function, N=2e6, base -> new:

| function            | base     | new      | gain |
|---------------------|----------|----------|------|
| `xm.magnitude`      | 14.0 ms  | 7.7 ms   | 1.8x |
| `xm.normalize`      | 20.5 ms  | 14.4 ms  | 1.4x |
| `make_normal`       | 29.8 ms  | 9.1 ms   | 3.3x |
| `random_direction`  | 219.8 ms | 155.7 ms | 1.4x |

End-to-end (1e7 rays, spherical crystal, keep_history=False), separate
processes, best of 3, run against a clean git worktree for the baseline:

| source           | base    | new     | gain  |
|------------------|---------|---------|-------|
| SourceDirected   | 5.26 s  | 4.90 s  | 1.07x |
| SourceFocused    | 2.80 s  | 2.61 s  | 1.07x |
| SourceGeneric    | 4.71 s  | 4.29 s  | 1.10x |

MEASUREMENT CAVEAT, recorded so it is not repeated. An earlier figure of 1.24x
for this same change set was wrong. It came from monkeypatching the baseline
and the optimized version into a *single* process and timing them one after
the other; the second measurement benefits from warmed allocator and page
cache state. Measuring each variant in its own process, with the baseline
taken from a clean `git worktree`, gives ~1.07-1.10x. Always benchmark engine
changes in separate processes.

So the honest gain is ~7-10% end-to-end, not the ~24% first estimated. The
per-function speedups are real and reproducible; they are simply a smaller
share of total runtime than the microbenchmarks suggested, because the hot
path is spread across many masked-gather and reduction operations that this
change does not touch (see `location_from_distance`, `check_bounds`,
`make_image` in the profile).

Correctness: detector images are bit-identical to the baseline across 28
cases (4 source configurations x 7 seeds), compared via a position-weighted
image checksum. `pytest tests/` 36 passed. Photon statistics are untouched:
no sampling, masking, or RNG-consumption changes; only last-bit float
reassociation. No config, API, or output-dict change, so no version bump.

Also fixed here (pre-existing, unrelated to the optimization):
`examples/example_01/example_01.py` referenced the stale class name
`XicsrtOpticCrystalSpherical`, which no longer exists, so the example raised
"Could not find ... in available objects" on the baseline commit. Renamed to
`XicsrtOpticSphericalCrystal` to match `optics/_XicsrtOpticSphericalCrystal.py`.
The companion `example_01.ipynb` already used the correct name, which is why
the drift went unnoticed. All three examples now run, and example_01 is usable
as a regression check again.


## F006 - Raytrace memory and multiprocessing instrumentation
Started: 2026-07-28
Status: Implemented (2026-07-28), pending user verification

Full approved plan: devel/plan_raytrace_memory.md
Baseline commit: e89a3bd

Implementation summary (2026-07-28):
- Regression harness `testing/compare_raytrace_regression.py` (temporary, not
  for master): Tier A bit-exact on found history / images / meta, Tier B
  statistical on lost rays, over 4 scenario variants (history, nohistory,
  multi-run, multiprocessing), each at `number_of_iter=3`. Self-tested by
  injecting real source mutations: a 1 ULP ray perturbation and a global-RNG
  stream shift both correctly FAIL the harness.
- Instrumentation (all verified bit-identical): real `mp: pool_join` /
  `mp: result_transfer` / `mp: combine` timers, confirming defect 3 (the old
  `mp: gathering` measured 12 us against 1.27 s of real cost);
  `profiler.getResults`/`profiler.merge` plus a pool initializer so worker
  timings surface in the parent under `spawn` as well as `fork`; per-iteration
  found/lost/history-bytes/peak-RSS logging.
- Fix 4 (`_sort_raytrace`): dedicated `np.random.Generator` + `choice`,
  measured 22.2x faster (0.0787 s -> 0.0035 s at 4.4e6 rays). This shifts the
  global RNG stream for `num_iter > 1`, contrary to the plan's Finding 6; see
  Finding 8 in the plan for the correction and a four-part proof that the
  change is purely a stream shift (compensation test is bit-exact; detected
  counts and per-ray distributions statistically identical, all p > 0.6).
  User-approved 2026-07-28.
- Fix 5 (`make_image`): `np.bincount` scatter-add, bit-identical, 75x faster
  than the per-ray loop and 13x faster than `np.add.at`.
- Fix 6 (`raytrace_single`): releases the dispatcher history each iteration.
  Bit-identical; measured 1629 MB -> 1175 MB peak RSS (-27.9%) at 2.44e6
  rays/iter, matching one full history copy (453 MB predicted) to 0.2%.
- Fix 7 (`combine_raytrace`): allocates from the *input* ray keys with
  `np.empty`, which fixes the dropped `weight` array; frees inputs as they are
  consumed behind a new opt-in `consume_input` flag (default False, so the
  documented public usage does not have its inputs destroyed). Verified
  `weight` is now present and correct in both engines and survives the hdf5
  round-trip.
- jaxrt: fixes 4 and 7 propagate automatically via the direct import in
  `jaxrt/_engine.py`; the "known benign quirk" entry in devel/jaxrt_sync.md is
  resolved and deleted. No divergence introduced.
- Verification: pytest tests/ (36 passed), pytest tests/jaxrt/ (10 passed),
  example_00 and example_02, and both `python -m xicsrt` and `--mp` CLI paths.

Pre-existing defects found during implementation, NOT fixed (out of scope,
reported only):
- The `xics_jax` import perturbs the global RNG stream, so `random_seed` does
  not guarantee reproducibility. Tracked as F009; see also Finding 9 in the
  plan.
- `examples/example_01/example_01.py` is broken on baseline: it requests
  `XicsrtOpticCrystalSpherical`, but the class is `XicsrtOpticSphericalCrystal`.

Goal: enable ~1e9 generated / ~1e6 detected ray runs on the Princeton Stellar
cluster (768 GB, 96 cores), scaling up from a working 5.295e7 / 1.272e4 run.
Originally motivated by a suspicion that `combine_raytrace` was the
bottleneck; that hypothesis was disproved by measurement (see below). The
remaining work stands on its own merits: one correctness bug, one real memory
win, and two speedups.

Findings (all measured, details and tables in the plan):
- `combine_raytrace` is not the bottleneck. ~195 MB combined history at 1e6
  detected rays, and with `keep_history=False` its history block is skipped
  entirely by the guard at `xicsrt_raytrace.py:359` (same for
  `_sort_raytrace` at line 254).
- Memory is not the constraint. Measured 195 B/ray of history (65 B/ray x 3
  elements) + ~110 B/ray temporaries + a ~2x transient => ~26 GB at 12
  workers x 4.4e6 rays/iter with history on; ~30x headroom at 768 GB.
- `np.einsum` does NOT use threaded BLAS: measured zero speedup from 1 to 8
  threads on `'ij,ij->i'` and `'ij,ijk->ik'` (the entire hot path), while a
  2000^3 gemm scaled 62 -> 33.5 ms. A 12-runs x 8-threads layout therefore
  uses ~12 of 96 cores. Retained by user decision; documented, not changed.
- `_sort_raytrace` shuffles millions of indices to keep ~833 lost rays:
  0.0734 s vs 0.0035 s for `Generator.choice(replace=False)` (21x), with
  uniformity verified (chi2=9961.9, dof=9970, p=0.52).
- `np.random.shuffle` perturbs the *global* RNG stream in an N-dependent way;
  a dedicated `np.random.Generator` avoids this (verified), keeping found
  rays bit-identical while only lost-ray selection changes.

Pre-existing defects found:
- `combine_raytrace` silently drops the `weight` array from combined
  histories (`RayArray.zeros` omits it; the copy loop iterates output keys).
  This is the "known benign quirk" at `devel/jaxrt_sync.md:164-166`; fixing
  it resolves that entry for both engines.
- Worker profiling is invisible: `profiler_results` is a per-process global
  that is never returned, so `profiler.report()` shows only parent timings.
- `mp: gathering` (`xicsrt_multiprocessing.py:58`) times `.get()` on
  already-completed results and measures nothing; the real cost is inside the
  untimed `pool.join()`.
- `raytrace_single` holds ~2x history (previous iteration stays live while
  the next allocates).

Planned work: build a two-tier regression harness first (Tier A bit-identical
for found rays / images / meta over `number_of_iter=3`; Tier B statistical
for lost rays), then add multiprocessing + worker-profiler instrumentation,
then four fixes: `_sort_raytrace` sampling, `make_image` vectorization,
history release in `raytrace_single`, and the `combine_raytrace` rewrite.

Out of scope: per-element history compaction (~3x further ceiling gain, needs
its own feature entry); worker/thread layout changes.

Note: fix 7 changes the output dict structure (histories gain `weight`),
warranting a minor version bump at release time.

---

## F005 - Allow XicsrtPlasmaVmec to load either VMEC or saved DESC equilibria
Started: 2026-07-26
Status: Done (2026-07-26)

Implementation: `XicsrtPlasmaVmec.initialize_vmec` (in
`xicsrt_contrib/xicsrt_contrib/sources/_XicsrtPlasmaVmec.py`) now picks the
loader from the `wout_file` extension: `.nc` -> `VMECIO.load`, `.h5` ->
`desc.io.load`; any other extension raises `ValueError`. No other methods
needed changes since they only call `self.eq.map_coordinates(...)`, which is
identical for both equilibrium types. Verified with a standalone script
loading both `wout.nc` and `wout_desc_solved.h5` and checking
flux/Cartesian round-trip error (~1e-6 m or better for both); also ran the
full `pytest tests/` suite (36 passed) and `examples/example_00/example_00.py`
end-to-end. Updated the SULI Part 5 logbook notebook to point
`wout_file` at `/u/npablant/data/w7x/vmec/w7x_ref_172/wout_desc_solved.h5`.

Request: update the W7-X SULI Part 5 logbook notebook to use a pre-solved DESC
equilibrium file (`wout_desc_solved.h5`, produced by a separate converter
notebook via `VMECIO.load` + `solve_continuation_automatic` + `eq.save`)
instead of the original VMEC `wout.nc`. `desc.io.load` on a native DESC `.h5`
skips the VMEC spectral re-fit and is much faster to load than `VMECIO.load`
on a `wout.nc` (see the "Desc Coordinate Transform Performance Minimal
Example" notebooks). `XicsrtPlasmaVmec.initialize_vmec` currently only
supports `VMECIO.load` (`.nc`), so both VMEC and DESC equilibrium inputs
need to be valid.

Decision (user-approved plan): keep the `wout_file` config key name and the
`XicsrtPlasmaVmec` class name unchanged; detect the equilibrium format from
the file extension (`.nc` -> `VMECIO.load`, `.h5` -> `desc.io.load`) inside
`initialize_vmec`. No jaxrt sync needed (plasma sources are out of scope per
`devel/jaxrt_sync.md`). Scope limited to the canonical
`xicsrt_contrib/xicsrt_contrib/sources/_XicsrtPlasmaVmec.py` (the separate
`suli/suli2026_alston/xicsrt_contrib` clone is left untouched).

## F004 - Exploratory JAX-accelerated tools_jax for the numpy OO engine
Started: 2026-07-21
Status: Complete (2026-07-25). No clear advantage for CPU-based computation.
Update 2026-07-30 (F010, 1a): the opt-in JAX CDF sampler was made obsolete by
direct Voigt sampling (exact Normal+Cauchy mixture, no CDF tables at all).
`xicsrt_voigt_multi_jax.py`, `xicsrt_faddeeva_jax.py`,
`tests/test_voigt_multi_jax.py`, and the `_USE_JAX_VOIGT_MULTI` toggle were
deleted.
Manual benchmarking under realistic `raytrace_multiprocessing` usage (the
actual way production W7-X jobs are run) showed no net speedup from the
opt-in JAX path once multiprocessing already saturates the CPU; see "Final
disposition: manual multiprocessing benchmark" below. The single-process-only
benchmark from the 2026-07-21 session (below) is superseded by this finding.
The opt-in code (`_USE_JAX_VOIGT_MULTI`, default `False`) is left in place,
disabled by default, in case a future GPU target changes the conclusion.

Final disposition: manual multiprocessing benchmark (2026-07-25):
- The 2026-07-21 sessions below only benchmarked single-process
  (`raytrace`/`raytrace_single`) runs, where the JAX path showed a real
  ~1.3-2x speedup on the 184-line Ar16+ workload. That is not how production
  jobs are actually run.
- Manual benchmarking (M1 MacBook, CPU only) comparing plain numpy under
  `raytrace_multiprocessing` (10 runs, all cores) against the JAX opt-in path
  run with an equivalent multiprocess/iteration split found the two
  statistically indistinguishable (~5m33s vs ~5m37s for matched configs,
  efficiency/ray counts consistent within statistics). Once multiprocessing
  already parallelizes across all CPU cores, the single-process JAX
  speedup is not additive and provides no net benefit.
- Follow-up tuning of the gridsize-bucketing strategy (fixed gridsize 1024,
  fixed 4096, and a smaller-bucket/lower-max-gridsize variant) was also
  tried and gave no further improvement over the power-of-two/8192-max
  scheme already implemented; performance returned to baseline in each case.
- Conclusion: on CPU, for this workload and at this problem scale, there is
  no configuration of the JAX opt-in path that outperforms plain numpy once
  multiprocessing is used as it is in practice. The code is retained
  (disabled by default) rather than removed, since a GPU target (not tested
  here) remains a plausible future path per F003's Princeton Stellar A100
  reference; any future revisit should start from a GPU benchmark rather
  than further CPU tuning.

Second follow-up: revisited at production scale (2026-07-21):
- Prompted by: "I would like to start a new feature to try to accelerate
  plasma bundle generation with tools/xicsrt_voigt_multi... For this feature
  I want to accelerate the Object Oriented python+numpy code; not move fully
  to the jaxrt code... I want to be able to turn on and off jax acceleration
  (hard-coded changes such as commented in and out code)."
- The original abandonment (below) benchmarked a toy ~15-line spectrum.
  Re-benchmarking at the true production scale (184-line Ar16+ table, the
  actual `ar16_voigt` line count) gave the opposite conclusion: JIT dispatch
  overhead is no longer dominant once the per-call Faddeeva work is large
  enough, and a JAX path is faster than plain numpy for this workload
  (measured ~2x on `generate_wavelength` wall-clock, ~1.3x on full
  `raytrace`, in a real W7-X `ar16_voigt` run with `bundle_count` reduced to
  ~1000 for fast iteration).
- Two retracing hazards from the original attempt were both fixed
  differently this time:
  - `gridsize` (auto-computed per bundle from local sigma/gamma, so it
    varies continuously bundle-to-bundle): now rounded up to the next
    power-of-two "bucket" (floor 128) before being passed as a
    `static_argnames` argument to `jax.jit`, bounding the number of distinct
    compiled shapes across a whole run to a handful instead of ~1 per bundle.
  - `size` (the Poisson-derived ray count per bundle, also varying
    bundle-to-bundle): this is no longer a jit argument at all. The final
    uniform draw + inverse-CDF `numpy.interp` sampling step happens in plain
    numpy on the host (as it always did in the numpy engine), so `size`
    never touches the jit trace signature. This also means results stay
    reproducible via `numpy.random.seed` (no RNG deviation from the numpy
    engine, unlike the mirror-directory design considered in the original
    session).
- Implementation: two new modules, `xicsrt/tools/xicsrt_faddeeva_jax.py` and
  `xicsrt/tools/xicsrt_voigt_multi_jax.py`, API-compatible drop-ins for
  `xicsrt_faddeeva.py`/`xicsrt_voigt_multi.py` (not a `tools_jax/`
  subpackage this time; they live directly in `xicsrt/tools/` since `jax`
  via `xics_jax` is already a mandatory import of
  `_XicsrtSourceGeneric.py`). A single hard-coded module-level flag,
  `_USE_JAX_VOIGT_MULTI` in `_XicsrtSourceGeneric.py` (default `False`),
  switches `random_wavelength_multi_voigt` and `random_wavelength_ar16_voigt`
  between the two backends; there is no config option and no runtime
  detection/fallback, per the "readability first, hard-coded toggle" request.
- Also fixed in the same session (pre-existing, unrelated bug):
  `random_wavelength_ar16_voigt` was reading `self.param.get("gridsize")` /
  `self.param.get("cutoff", 1e-4)` instead of the actual config keys
  `self.param['multi_gridsize']` / `self.param['multi_cutoff']`, silently
  ignoring those two config options for `ar16_voigt` (they happened to
  default correctly by luck: `None`/`1e-4`). Now reads the correct keys,
  matching `random_wavelength_multi_voigt`.
- Tests: `tests/test_voigt_multi_jax.py` (`pytest.importorskip('jax')`),
  covering Faddeeva/Voigt numeric agreement with the plain-numpy kernel,
  CDF/PDF agreement at a matching gridsize, gridsize-bucketing correctness
  (including that two different requested gridsizes landing in the same
  bucket give bit-identical CDF tables), sampling histogram-vs-PDF
  statistics, and `numpy.random.seed` reproducibility. Full suite (36 tests)
  and `examples/example_00/example_00.py` pass with the (default) flag left
  `False`.
- Caveat (documented in the new module's docstring): this is a workload-
  dependent opt-in, not a strict upgrade. It is only faster for large line
  lists (validated at 184 lines); for small hand-specified `multi_voigt`
  line lists (the original F004 toy-benchmark regime) plain numpy is still
  expected to be faster, matching the original conclusion below.

Original session, root cause and benchmarks (2026-07-21 follow-up session):
- Root cause of the hang: the (uncommitted) edit wiring
  `xicsrt/sources/_XicsrtSourceGeneric.py` to import `xicsrt.tools_jax`
  instead of `xicsrt.tools` exposed the plain numpy engine's per-bundle
  Python loop (`XicsrtPlasmaGeneric.create_sources`, one
  `multi_voigt_random`/`ar16_voigt` call per bundle) to `jax.jit`. Each
  bundle has its own temperature-dependent `gridsize` (auto-computed in
  `_prepare_bounds` when `gridsize=None`) and its own Poisson-derived
  `size`; both are part of the `jax.jit(static_argnames=('N','size'))`
  trace/shape signature, so XLA recompiled on nearly every bundle
  (measured ~150-900ms/compile in isolation). For a realistic plasma
  (`bundle_count ~ 1e4`, see `xicsrt_w7x_npablant.get_config()`) this
  compounds to tens of minutes to hours. `tests/test_voigt_jax.py` never
  hit this because it always called with fixed/repeated shapes (jit cache
  hit), which is why "pytest passes but the notebook hangs".
- Benchmark (CPU, ~15-line Ar16+ spectrum, `bundle_count=1e4`, ~8-30 rays/
  bundle, matching the real w7x_npablant production config):
  - Plain numpy (`xicsrt.tools.xicsrt_voigt_multi`, pre-jax behavior):
    ~5.3s for all 10,000 bundles.
  - Per-bundle jax with `gridsize`/`size` rounded to fixed power-of-two
    "buckets" (fixes the retracing/hang): ~7.5s -- correctness fixed, but
    still *slower* than plain numpy.
  - Fully batched `jax.vmap` over all 10,000 bundles in a single jit call
    (fixed common grid, padded per-bundle ray capacity): ~12.6s warm
    (~20s incl. compile) -- also slower than plain numpy.
  - Conclusion: for this workload (many small independent per-bundle
    draws, ~15 lines, single-digit-to-tens of rays/bundle) on CPU, JAX
    dispatch/compile overhead is not recovered by the small amount of
    vectorizable math, regardless of whether the shape/retracing bug is
    fixed. There is currently no `tools_jax`-based approach that beats
    plain numpy for `XicsrtPlasmaGeneric.create_sources` at this problem
    scale on CPU.
  - Open question for a future session: whether a GPU target (e.g.
    Princeton Stellar A100s, per F003) and/or a much larger line list/
    grid/ray-count per bundle shifts this crossover point in JAX's favor.
    Not evaluated in this session.
- Disposition: `xicsrt/tools_jax/` and `tests/test_voigt_jax.py` were
  deleted (not merely left unwired) since the benchmarks show no viable
  path to a speedup for the actual production use case as currently
  architected; reintroducing this approach should start from a fresh
  GPU/large-scale benchmark rather than re-adding the CPU-oriented
  mirror-directory code as-is.

Original session notes (2026-07-21, historical; describes code that has
since been deleted -- see "Disposition" above):

Goal: Let the ordinary object-oriented numpy+scipy code opt in to targeted
JAX acceleration of `tools/xicsrt_voigt_multi.py` (and
`xicsrt_voigt.py`/`xicsrt_faddeeva.py`) for exploratory use, without touching
the numpy engine or the separate `xicsrt.jaxrt` raytracing engine. Follows
up on F001 (~94% of multi_voigt runtime is in the Faddeeva/wofz kernel) and
is unrelated to F003/jaxrt (not tracked in `devel/jaxrt_sync.md`).

Implementation (2026-07-21):
- Chose the "mirror directory" option (over adding an `xp=numpy` array-
  namespace parameter to the existing `xicsrt/tools/*.py` files) to keep
  zero blast radius on the numpy engine, at the cost of a duplicated (but
  small) copy of the Weideman/voigt math.
- New subpackage `xicsrt/tools_jax/` (not added to `setup.py`/
  `extras_require`; requires `jax`/`jaxlib` installed manually; importing it
  without jax raises a plain `ModuleNotFoundError`):
  - `xicsrt_faddeeva.py`: jax/jit port of `wofz_weideman`/`voigt_profile`
    (`_weideman_coeffs` setup reused unchanged from `xicsrt.tools`, plain
    numpy, `lru_cache`d).
  - `xicsrt_voigt_multi.py` / `xicsrt_voigt.py`: drop-in API-compatible
    ports of `multi_voigt`/`multi_voigt_cdf_tab`/`multi_voigt_random` and
    `voigt`/`voigt_cdf_tab`/`voigt_random`; return plain `numpy.ndarray`.
- Jit boundary: data-dependent grid/domain sizing (min/max over line
  widths, `warnings.warn`, validity checks) stays host-side Python/numpy,
  duplicated from the numpy engine (cannot be traced under `jax.jit`); the
  grid evaluation, cumsum/normalize, and (for the `_random` functions) the
  random draw + inverse-CDF interpolation are fused into one `jax.jit` call.
- RNG deviation (deliberate, user-approved): the `_random` functions use an
  internal `jax.random.PRNGKey` auto-seeded from OS entropy (`secrets.
  randbits`) each call, not the global `numpy.random` state, so they are
  NOT reproducible via `numpy.random.seed` unlike the numpy-engine versions.
  Documented in each function's docstring and covered by a dedicated test.
- Tests: `tests/test_voigt_jax.py` (`pytest.importorskip('jax')`), checking
  numeric agreement with `xicsrt.tools` for deterministic functions and
  histogram-vs-PDF statistics for the random samplers. Full suite (30
  tests) and `examples/example_00/example_00.py` verified passing.

Key prompt from user: "I would like to use jax to accelerate the code in
xicsrt_voigt_multi.multi_voigt_random when running a 'normal' object
oriented (OO) python+numpy code ... I just want a way to temporarily test
out some targeted jax accelerations." Later confirmed: mirror-directory
approach (not xp-injection), jax.random for sampling (auto-seeded from OS
entropy, not a `key` parameter), jit fused across CDF-build + sampling,
`xicsrt/tools_jax/` as an importable subpackage, port `xicsrt_voigt.py` too,
no benchmark script, no `setup.py` changes.

---

## F003 - JAX-accelerated raytracing engine (xicsrt.jaxrt)
Started: 2026-07-17
Status: In Progress (2026-07-17: phase 1 implemented, pending user verification)

Goal: A parallel JAX-based engine in a new subpackage `xicsrt/jaxrt/`, enabling
jit/vmap acceleration on CPU now and GPU (Princeton Stellar cluster, A100s)
later. The numpy engine remains completely untouched; `jax` is an optional
dependency. Priorities, in order: (1) readability for doctorate-level physics
researchers without CS background, (2) exact photon statistics at every
element (core XICSRT tenet, no exceptions), (3) acceleration.

Full approved plan: devel/plan_jaxrt.md

Key decisions (approved 2026-07-17):
- Parallel subpackage `xicsrt/jaxrt/`, pure functions + config dicts (no
  mutable mixin classes); same JSON configs as the numpy engine.
- float64 everywhere (jax x64 mode enabled on import).
- Explicit jax.random key threading; statistically equivalent to the numpy
  engine, not bit-identical (approved).
- Poisson ray counts via capacity + mask: exact Poisson draw N, fixed
  capacity = mean + 10 sigma, rays beyond N masked from birth; error on
  overflow (~1e-23 probability). Photon statistics exactly preserved.
- Phase 1 scope: Generic/Directed/Focused sources; Plane/Sphere/Cylinder/
  Torus shapes; None/Mirror/Crystal/MosaicCrystal interactions; rocking
  curves step/gaussian/file. Sequential runs, single process, single device.
- Deferred: plasma sources, mesh optics, filters, multi-GPU pmap/sharding,
  any multiprocessing interplay (intentionally excluded).

---

## F002 - Kent / FB8 directional distribution for angular spread
Started: 2026-07-17
Status: Pending

Goal: Implement the Kent (FB5) and/or FB8 family of directional distributions
as a proper angular-spread option in `xicsrt/tools/xicsrt_spread.py`, replacing
the small-angle Gaussian approximation for anisotropic emission. This should
include at minimum a sampler (analogous to existing `vector_dist_*` functions)
and a corresponding `solid_angle` calculation.

The FB8 family is the mathematically correct set of distributions for
anisotropic/elliptical angular emission on the unit sphere:
FB8 (8-param) ⊃ FB6 ⊃ FB5 / Kent (5-param) ⊃ von Mises–Fisher.
The Kent distribution is the natural spherical analogue of the bivariate normal.

Reference implementation (`fb8` v1.2.2, MIT, Tianlu Yuan):
- https://pypi.org/project/fb8/
- https://github.com/tianluyuan/sphere

Findings / notes (from code review, 2026-07-17):
- Only sampling and pdf evaluation are likely needed (not MLE, gradient, or
  contour functionality from the reference package).
- The reference `rvs` implementation is rejection-based and may be inefficient
  for high concentration parameter κ (i.e. narrow beams), which is a common
  XICSRT use case.
- The reference package emits noisy warnings/logging that conflict with the
  XICSRT `mirlogging` conventions and uses its own RNG separate from the rest
  of XICSRT.
- Open question: whether Kent (FB5, elliptical symmetric) alone is sufficient
  or whether the full FB8 asymmetry is needed.

---

## F001 - Performance enhancement for tools/xicsrt_voigt_multi.py
Started: 2026-07-17
Status: In Progress (2026-07-17: implemented jax-friendly Weideman kernel)

Goal: Faster multi_voigt / multi_voigt_cdf_tab / multi_voigt_random.

Implementation (2026-07-17):
- Chose Option D: replace scipy.special.wofz (non-jax-able C routine) with a
  pure-numpy Weideman (1994) rational Faddeeva approximation, default N=16
  (tunable keyword, not user-facing). N=16 max abs err ~3e-7 vs wofz.
- New shared kernel xicsrt/tools/xicsrt_faddeeva.py (voigt_profile, wofz_weideman,
  cached _weideman_coeffs). Pure array arithmetic -> future jax swap is trivial.
- xicsrt_voigt.voigt now wraps voigt_profile; original wofz version retained
  (unused) as voigt_wofz per request.
- xicsrt_voigt_multi.multi_voigt vectorized (broadcast over line axis, no
  per-line Python loop); N passthrough added to cdf_tab / random.
- Added tests/ (pytest): accuracy vs wofz (N=16/24/32), voigt==voigt_wofz,
  multi==sum-of-singles, CDF properties, sampler histogram-vs-PDF. 11 pass.
- setup.py: extras_require={'test': ['pytest']}.
- Modest CPU speedup today (~1.1-1.7x, grows with line count); main value is
  the jax-ready, differentiable, vectorized kernel.

Findings:
- ~94% of runtime is inside scipy.special.wofz. Cost scales with
  n_lines x n_grid evaluations; everything else (cumsum, bin widths,
  linspace, interp) is negligible.
- Rewriting the per-line Python loop as a 2D numpy broadcast gives NO
  meaningful gain (1.0-1.2x, only at tiny sizes) — wofz still runs on the
  same number of points; loop overhead is trivial. Not worth the added
  memory/complexity.
- Windowing (evaluate each line only on its ±cutoff grid slice via
  np.searchsorted) gives a real ~1.5x (~33%) speedup that scales with
  line count (5→300 lines: 1.49x→1.56x). Tradeoff: small tail-truncation
  error (tunable via window width). Deferred: not worth the accuracy
  tradeoff / added complexity.
- Micro-opts (constant bin width -> scalar dx, hoist sqrt(2)/sqrt(2pi))
  are all <5%; skip.
- Only path to a large win is replacing wofz with a pseudo-Voigt
  approximation — faster but changes accuracy; not pursued.

Decision: No code change at this time.
