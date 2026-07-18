# XICSRT Feature Requests

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
