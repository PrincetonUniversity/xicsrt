# Plan: hybrid GPU raytracing for W7-X training-set generation

Status: design note only — DO NOT implement. Gated on production experience
with the Phase 1 (F010) numpy-path acceleration.
Tracking: F010 Phase 3 in devel/features_request.md.
This file includes AI generated content using Claude (Fable 5).

## Concept

Split each iteration between the two engines at the existing
generation/tracing boundary:

1. **Plasma generation stays in numpy** (post-F010 vectorized path):
   bundle setup, DESC rho mapping, xics_jax line parameters, batched
   direct Voigt sampling. Plasma sources are deliberately out of jaxrt
   scope (sources still in flux; ar16/VMEC must stay out of the public
   codebase — see F010 "out of scope").
2. **Optics tracing moves to the jaxrt jit'd chain on GPU**: the jaxrt
   engine already separates generation from tracing at the
   `_build_trace` boundary in `xicsrt/jaxrt/_engine.py` — `_build_trace`
   returns a pure jit'd function of (rays pytree, key) that runs the
   whole optics chain. A hybrid driver would construct the optics
   portion only and feed it externally generated rays.

## Ray handoff: padded fixed capacity

jaxrt requires fixed array shapes inside jit (jaxrt_sync.md invariant 7).
The numpy generator produces a Poisson-varying ray count N per iteration.
Handoff:

- capacity C = ceil(mean + 10*sigma) of the per-iteration ray count
  (same rule the jaxrt sources already use; see invariant 2).
- rays are written into (C, ...) arrays; entries >= N are masked from
  birth (`mask=False`).
- overflow (N > C) raises RuntimeError — never truncate silently.

Statistics are preserved exactly: capacity+mask is a layout, not an
approximation. Detected counts and all per-element statistics are
identical in distribution to the pure-numpy engine.

## Expected benefit and its ceiling

Post-Phase-1 the worker time split shifts toward tracing (~60% share:
optics tracing plus per-ray direction/origin overheads that would move
with it). Amdahl ceiling for GPU-accelerating tracing only:

    speedup_node <= 1 / (0.4 + 0.6/s_gpu)  ->  ~2.0-2.4x per node
    (for s_gpu in the 5-20x range; beyond that generation dominates)

The Phase-1 numbers must be re-measured on Stellar before trusting this
split (single-process M1 measurement showed 5.1x overall from Phase 1
alone, which changes the balance materially).

## Fleet economics (why this is deferred)

- Stellar GPU jobs are capped at 2 nodes; CPU jobs can use 42+ nodes.
- Even a 2.4x per-node GPU speedup on 2 nodes loses to 42 CPU nodes
  running the Phase-1 path: 2 x 2.4 = 4.8 node-equivalents vs 42.
- run_12 (GPU+MPS hybrid, pre-Phase-1) showed no benefit because
  generation dominated and one A100 was shared by 12 workers. Phase 1
  removes much of the generation cost, so the balance is now different —
  but the fleet-size argument stands unless GPU node limits change.

## Prerequisites before implementing

1. First real A100 benchmark of the jaxrt optics chain via
   `testing/benchmark_jaxrt.py` (all existing numbers are CPU/M1).
2. Stellar re-profile of the Phase-1 numpy path (run_08 layout and
   re-tuned splits 24x4 / 48x2) to fix the true post-F010 tracing share.
3. Production experience from the first Phase-2 training campaign:
   actual core-hours per image and queue behavior decide whether the
   2-node GPU pool is worth engineering effort.

## Sketch of the implementation (when/if gated open)

- New driver `raytrace_hybrid(config)`:
  - host: instantiate numpy plasma source(s) via Dispatcher (unchanged).
  - device: `jaxrt._dispatch.setup_optics` + `_build_trace` for the
    optics chain only; jit once per config.
  - per iteration: numpy `generate_rays()` -> pad/mask to capacity ->
    `jnp.asarray` upload -> jit'd trace -> host download -> reuse the
    existing `_sort_raytrace` / `combine_raytrace` plumbing.
- Multi-worker layout: one process per GPU (not 12 workers sharing one
  A100 — run_12 showed MPS contention); remaining CPU cores run numpy
  generation threads for that process.
- No changes to photon statistics anywhere (capacity+mask only).
