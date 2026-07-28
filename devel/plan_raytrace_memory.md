# Plan: Raytrace Memory & Multiprocessing Instrumentation (F006)

This file includes AI generated content using Claude (Opus 4.5).

Status: Approved 2026-07-28. Not yet started.
Tracking: devel/features_request.md (F006).
Baseline commit: e89a3bd ("Changed the handling of gridsize_max, now matches
the jax version.")

This document is self-contained: it includes the approved plan, all
measurements, the user decisions behind each choice, and the codebase
research needed to implement in a fresh session without re-exploration.

## Origin of this work

User goal: run a W7-X Ar16+ raytrace with ~1e9 rays generated / ~1e6 rays
detected on the Princeton Stellar cluster (768 GB allocation, 96 cores),
scaling up from a working local run of 5.295e7 generated / 1.272e4 detected
(efficiency 2.403e-04).

Reference notebook (local machine):
`/u/npablant/code/notebooks/npablant-2019/logbook/2026-07-26 - W7-X XICS
Raytracing - SULI 2026 L. Alston, Part 6b - Memory optimization claude.ipynb`

Config of that run: `number_of_runs = 12`, `number_of_iter = 1`,
`keep_history = False`, `time_resolution = 1e-1`,
`wavelength_dist = 'ar16_voigt'`, `max_rays = 1e10`, via
`xicsrt_analysis/w7x_npablant/xicsrt_w7x_npablant.py` (source
`XicsrtPlasmaW7xSimple`, optics `crystal` = `XicsrtOpticSphericalCrystal`,
`detector` = `XicsrtOpticDetector`).

Original user hypothesis: "I think that my issue is inefficiency of the
`combine_raytrace` function, rather than the ability to handle the number of
rays." Reported symptom: during the run total memory is 5-7 GB per process
and CPU is healthy; then CPU usage drops away and the run spends a long time
with low CPU while memory usage steps up.

**The original hypothesis was investigated and disproved.** See Finding 1
and Finding 2 below. The work that remains is worth doing on its own merits
(one real correctness bug, one real memory win, two speedups), but it is not
the cause of the reported symptom.

## Findings (all measured, not assumed)

### Finding 1: `combine_raytrace` is not the bottleneck

At the 1e6-detected-rays target the combined history is ~195 MB (~400 MB
transient while inputs and output are both live). That is negligible against
a 768 GB allocation.

Furthermore, the reference run sets `keep_history = False`, and both of the
suspected functions are guarded:

- `_sort_raytrace`: `if len(input['history']) > 0:` (xicsrt_raytrace.py:254)
- `combine_raytrace`: `if len(input_list[0]['found']['history']) > 0:`
  (xicsrt_raytrace.py:359)

With `keep_history = False` **neither history block executes at all**. The
observed tail cannot have been caused by either function in that run.

### Finding 2: the observed tail is swap, not an XICSRT bug

12 workers x ~6 GB/worker = 60-84 GB against 32 GB of physical RAM on the
local machine: 2-3x oversubscribed. That produces exactly the reported
signature -- CPU collapses (processes blocked on page-in), wall-clock
explodes, and resident memory climbs in steps as the parent faults worker
pages back in during `pool.join()`. `Pool(None)` uses all cores, so all 12
runs start at once rather than batching.

This will not occur on Stellar at 768 GB. Per user instruction, this
environmental finding is recorded here but deliberately NOT written into the
F006 feature entry, which is kept code-focused.

### Finding 3: memory is not the constraint for the 1e9-ray goal

Measured, by allocating the actual arrays:

| quantity | value |
|---|---|
| bytes/ray for one stored element | 65 B (origin 24 + direction 24 + wavelength 8 + weight 8 + mask 1) |
| history, 3 elements (plasma, crystal, detector) | 195 B/ray |
| sphere-intersect temporaries | ~110 B/ray |
| transient factor in `raytrace_single` | ~2x history (see fix 6) |

Peak ~= `n_workers * rays_per_iter * 500 B`.

Ray dicts carry 5 keys and history is stored via `deepcopy(rays)` at *every*
element (`_Dispatcher.py:162` for the source, `_Dispatcher.py:187` for each
optic). Arrays are never compacted: masked-off rays are still carried at full
width until `_sort_raytrace` runs at the end of the iteration.

At 12 workers x 4.4e6 rays/iter with `keep_history = True`: **~26 GB**.
Against 768 GB that is ~30x headroom.

Capacity table (768 GB, history on, 60% safety margin):

| n_workers | rays/iter/worker | rays in flight |
|---|---|---|
| 12 | 7.7e7 | 9.2e8 |
| 48 | 1.9e7 | 9.2e8 |
| 96 | 9.6e6 | 9.2e8 |

### Finding 4: einsum does NOT use threaded BLAS

Measured on the dev machine (numpy 2.4.6, Accelerate BLAS), 2e6 rays:

| operation | 1 thread | 8 threads |
|---|---|---|
| `einsum 'ij,ij->i'` | 13.0 ms | 13.1 ms |
| `einsum 'ij,ijk->ik'` | 44.0 ms | 44.6 ms |
| `sqrt` | 2.0 ms | 2.1 ms |
| `M @ M` (2000^3 gemm) | 62 ms | 33.5 ms |

XICSRT's hot path is entirely `'ij,ij->i'` / `'ij,ijk->ik'` contractions and
elementwise ufuncs (`_ShapeSphere.py:74`, `_InteractMirror.py:39`,
`_XicsrtSourceGeneric.py:329`). These dispatch to numpy's single-threaded
internal loop and never reach `dgemm`. The only `np.linalg` call in a trace
path is `eigvals` in `xicsrt_quartic.py`, which is torus-only (the W7-X
config uses a sphere).

Consequence: the user's planned Stellar layout of **12 runs x 8 threads uses
~12 of 96 cores**, not 96. The user elected to keep 12 runs anyway; this is
recorded as a documented note, not changed. If wall-clock later matters more
than per-run memory, `number_of_runs = 96` with `OMP_NUM_THREADS=1` would be
~8x faster at ~210 GB peak, still well inside 768 GB.

### Finding 5: `_sort_raytrace` lost-ray sampling is slow and wasteful

Current code (xicsrt_raytrace.py:259-266) materializes all lost indices, then
builds `arange(N_lost)` and runs a full Fisher-Yates `np.random.shuffle` over
millions of elements -- all to retain ~833 lost rays.

Measured at N=4.4e6 rays, 2.4e-4 efficiency, `max_lost = 833`:

| implementation | cost | correct? |
|---|---|---|
| old: `arange` + `shuffle` | 0.0734 s | baseline |
| `Generator.choice(replace=False)` + `flatnonzero` | 0.0035 s (21x) | all sampled rays genuinely lost |
| rejection sampling, no `flatnonzero` | 0.00012 s (610x) | all sampled rays genuinely lost |

Uniformity of the `choice` variant verified by chi-square over 4000 trials:
chi2 = 9961.9, dof = 9970, **p = 0.52** -- indistinguishable from uniform.

**Decision: use `choice` + `flatnonzero`, NOT the rejection sampler.** At
0.0035 s the `choice` version is already 0.013% of a ~26 s iteration; the
remaining 0.0034 s is unmeasurable in practice. The rejection variant needs a
retry loop, `np.unique` de-duplication, a degenerate path for scarce lost
rays, and a subtler uniformity argument -- real complexity and real bug
surface for no user-visible gain. That is the wrong trade against the
readability-first mandate in AGENTS.md. The ~35 MB `flatnonzero` transient is
negligible against a ~26 GB peak.

### Finding 6: the global RNG hazard, and why it is avoidable

`np.random.shuffle` consumes a number of draws from the **global** numpy RNG
stream that depends on `N_lost`. Verified:

    N=  100 next=0.235120407257
    N=  101 next=0.235120407257
    N= 5000 next=0.253069322789

`_sort_raytrace` is called inside the iteration loop
(xicsrt_raytrace.py:157), so any replacement that also draws from the global
stream will consume a different number of draws and desynchronize every
subsequent iteration. A naive 1:1 regression test would then fail on
iteration 2+ despite the physics being untouched -- and, worse, could mask a
real regression.

This is fully avoided by using a dedicated `np.random.Generator`. Verified
that a dedicated Generator leaves the global stream untouched:

    global untouched by Generator: True

Found rays are produced entirely from the global stream *before*
`_sort_raytrace` runs, so with a dedicated Generator they remain
**bit-identical**. Only *which* lost rays are retained changes.

User decision: "I only need bit-identical reproducibility for the found rays,
I don't care if the lost-rays are sampled differently in a new implementation
as they are used only for visualization not for anything meaningful."

### Finding 7: pre-existing defects discovered during review

1. **`combine_raytrace` silently drops the `weight` array** from all
   combined histories. `RayArray.zeros()` (`_RayArray.py:82`) only creates
   origin/direction/mask/wavelength, and the copy loop
   (xicsrt_raytrace.py:383) iterates over the *output* keys, so `weight`
   never gets copied. Documented as a "known benign quirk" at
   `devel/jaxrt_sync.md:164-166`.
2. **The profiler is blind inside workers.** `profiler_results` is a
   per-process module global (`util/profiler.py:23`) and is never returned to
   the parent, so `profiler.report()` in the notebook shows only parent-side
   timings. This is precisely why the reported tail was never diagnosable.
3. **`mp: gathering` measures nothing.** `profiler.start('mp: gathering')`
   (`xicsrt_multiprocessing.py:58`) wraps `.get()` on results that have
   already completed (after `pool.join()`), so it returns instantly. The real
   cost lives inside the untimed `pool.join()`.
4. **`raytrace_single` holds ~2x history.** The previous iteration's `single`
   and the dispatcher's `self.history` stay alive while the next iteration
   allocates its rays.

## Phase 0 -- guidance only, NO code changes

Per user instruction, explain the logic but change nothing.

Total rays = `number_of_runs * number_of_iter * rays_per_iter`, where
`rays_per_iter` is driven by `time_resolution` (via the intensity calculation
in `_XicsrtPlasmaGeneric.create_sources`, line 310). Memory scales with
`n_workers * rays_per_iter` **only**: iterations run sequentially inside a
worker and are reduced to found+lost by `_sort_raytrace` before the next
begins.

**To reach 1e9 rays: raise `number_of_iter`; never raise `time_resolution`.**
Iterations are free in memory; `time_resolution` multiplies the peak.

With 12 runs at the current ~4.4e6 rays/iter, `number_of_iter = 19` gives
~1.0e9 generated / ~1e6 detected at ~26 GB peak (history on).

Side note: `raytrace_mp` accumulates `random_seed += ii` inside the loop
(`xicsrt_multiprocessing.py:48-50`), so seeds go 0, 1, 3, 6, 10, ... --
non-uniform but non-colliding. Leaving `random_seed = None` seeds each worker
from OS entropy.

## Step 1 -- create the F006 entry

Add F006 to `devel/features_request.md` (most recent at top), status Pending,
linking to this plan file. Keep it code-focused: the findings, the
measurement tables, and the einsum/BLAS note. Per user instruction, do NOT
include the local-machine swap guidance from Finding 2.

## Step 2 -- build the regression harness FIRST, against unmodified code

Create `testing/compare_raytrace_regression.py`. Temporary; not intended for
`master`. It must exist and be trusted **before** any source change.

Two tiers, reflecting the user's decision in Finding 6:

**Tier A -- exact (bit-identical):**
- `found` history: every key, every element (plasma, crystal, detector)
- all images
- all meta counts
- assertion: `np.testing.assert_array_equal` (exact, NOT `allclose`)

**Tier B -- statistical (lost rays only):**
- correct retained count
- every retained index corresponds to a genuinely lost ray
- uniformity across repeated trials (chi-square, as in Finding 5)

Harness configuration:
- fixed `random_seed`
- `keep_history = True`
- **`number_of_iter = 3`** -- this is the global-RNG-divergence detector. If a
  change perturbs the global stream, iteration 1 passes and iterations 2-3
  fail. A single-iteration test gives a false pass.
- small `bundle_count` (~200) for fast turnaround
- a second variant with `keep_history = False`, to exercise the guard paths
  the production run actually hits

Baseline capture: use `git worktree add` at commit e89a3bd so both versions
run in separate processes against identical inputs.

**Self-test the harness:** deliberately perturb one ray by 1 ULP and confirm
the harness FAILS. An untested test harness is worthless.

## Step 3 -- instrumentation (must be bit-identical under Tier A)

1. `xicsrt_multiprocessing.raytrace`: replace the inert `mp: gathering` timer
   with real ones -- `mp: pool_join` (where compute and IPC transfer actually
   happen), `mp: result_transfer`, `mp: combine`.
2. Worker-side profiling: add `profiler.merge(results, prefix='worker: ')` to
   `util/profiler.py`; have `raytrace_single` attach its `profiler_results` to
   the output dict when profiling is enabled; have the parent aggregate into
   its global so the user's existing `profiler.report()` call surfaces worker
   timings alongside parent ones. `merge` must correctly sum `timedelta`
   totals and `num_calls`.
   - User decision: merge into the parent global with a `worker:` prefix
     (chosen over a per-worker breakdown or a results-dict-only field).
3. Diagnostic logging: per-run found/lost counts, peak RSS
   (`resource.getrusage`), and estimated history bytes.

All three are purely additive; the harness must show bit-identical output
afterward. This is the first real exercise of the harness.

## Step 4 -- code changes

| # | file / location | change | test bar |
|---|---|---|---|
| 4 | `xicsrt_raytrace.py:259-266` | `_sort_raytrace`: dedicated `np.random.Generator` seeded from the run's `random_seed`; `choice(..., replace=False)` replaces `arange` + `shuffle` | Tier A exact; Tier B statistical |
| 5 | `optics/_TraceObject.py:289` | `make_image`: vectorized scatter-add replaces the per-ray Python loop | Tier A exact |
| 6 | `xicsrt_raytrace.py:153-158` | `raytrace_single`: release full-N history right after sorting (`sources.history.clear()`, `optics.history.clear()`, `del single`) | Tier A exact |
| 7 | `xicsrt_raytrace.py:366-390` | `combine_raytrace`: allocate with `np.empty` keyed off the *input* keys (fixes the dropped `weight`), drop the `zeros()` + `copy()` double-materialization, free each input as it is consumed | Tier A exact, plus `weight` now present |

Notes:
- Fix 4 must seed the Generator deterministically from `random_seed` so
  reruns remain reproducible; they simply will not match the *old*
  implementation's selection.
- Fix 5 is only ~1.6% of runtime (measured: 0.466 s -> 0.052 s per 1e6 rays
  hitting an optic) but is bit-identical, trivially safe, and converges the
  numpy engine with `jaxrt/_images.py`, which already scatter-adds.
- Fix 6 is the single largest memory win: it removes the ~2x history
  transient, which matters most under the high `number_of_iter` recommended
  in Phase 0.
- Fixes 6 and 7 are pure allocation/deallocation changes with zero RNG
  interaction, so they must be bit-identical under Tier A -- a strong and
  easy correctness bar.

## jaxrt synchronization

`jaxrt/_engine.py:43` imports `_sort_raytrace` and `combine_raytrace`
directly from the numpy engine, so fixes 4 and 7 propagate to jaxrt
automatically.

Because `jaxrt/_rays.py:56` already creates `weight`, fix 7 **resolves** the
"known benign quirk" recorded at `devel/jaxrt_sync.md:164-166`. Delete that
entry rather than adding a divergence-log note.

Fix 5 converges `_TraceObject.make_image` with `jaxrt/_images.py`, which
already uses scatter-add (see the translation idiom table at
`jaxrt_sync.md:96`).

No divergence-log entry is expected. Confirm at implementation time.

## Verification

    pytest tests/
    pytest tests/jaxrt/
    python examples/example_00/example_00.py

plus the regression harness after every individual step (not batched at the
end). Add AI disclaimer headers to every modified Python file per AGENTS.md.

Leave `xicsrt/_version.py` alone -- the user will handle the version bump.
Note that fix 7 changes the output dict structure (histories gain a `weight`
key), which per `_version.py` semantics warrants a **minor** bump.

## Explicitly out of scope

- **Per-element history compaction.** Any ray with `mask == False` after
  element k can never be found, so compacting history to survivors (plus a
  reservoir sample of newly-lost rays) after each element would cut history
  from 195 B/ray to ~65 B/ray and raise the in-flight ceiling ~3x. This is a
  real design change; it would need its own feature entry and a jaxrt review.
  Deliberately deferred.
- **Worker/thread layout changes.** Finding 4 shows the 12x8 layout uses ~12
  of 96 cores, but the user elected to keep 12 runs. Document only.

## Suggested prompt to resume in a fresh session

    Read devel/plan_raytrace_memory.md (F006 in devel/features_request.md)
    and implement it. Build the regression harness in Step 2 first and
    self-test it before touching any source file, then proceed through
    Steps 3 and 4 in order, running the harness after each individual
    change rather than batching the verification.
