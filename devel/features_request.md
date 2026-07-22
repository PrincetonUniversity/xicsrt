# XICSRT Feature Requests

## F004 - Exploratory JAX-accelerated tools_jax for the numpy OO engine
Started: 2026-07-21
Status: Abandoned (2026-07-21). Root cause of the hang diagnosed and
confirmed by benchmark; `tools_jax/` and its test removed, and the
`_XicsrtSourceGeneric.py` import reverted to `xicsrt.tools`. See
"Root cause and benchmarks" below before attempting this again.

Root cause and benchmarks (2026-07-21 follow-up session):
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
