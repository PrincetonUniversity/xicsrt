# Plan F022: normalize W7-X emissivity_scale for a consistent source ray count

Status: approved by user 2026-08-03, not yet started.
Tracking: F022 in devel/features_request.md.

Note: this plan was produced with AI assistance using Claude (Sonnet 5).

## Objective

Make the number of rays generated at the plasma source consistent across
W7-X ML training samples (up to Poisson / bundle-count Monte-Carlo noise),
by normalizing away the two known causes of the current ~700x spread in
generated ray count, rather than predicting or clamping the ray count after
the fact.

All changes are entirely within `xicsrt_analysis`. Do NOT modify core
`xicsrt` or `xicsrt_contrib`.

## Root cause (already diagnosed, not re-investigated here)

Per F021's investigation (`devel/features_request.md`, `devel/devel_ai_log.txt`
2026-08-03 entry): the ~700x spread in per-sample generated ray count is
dominated by two multiplicative factors that scale the true expected photon
count before the Poisson draw:

1. `xicsrt.tools.xicsrt_spline.generate_random_emissivity` normalizes the
   profile SHAPE to a peak of 1.0 by design (this is correct and must not
   change), but the profile's volume integral varies ~157x across samples
   because the randomized zero-crossing radius changes.
2. `XicsrtPlasmaW7x.bundle_generate` (`xicsrt_analysis/w7x_npablant/sources/_XicsrtPlasmaW7x.py:196-209`)
   multiplies the w-line-only emissivity by a Te-dependent `I_tot/I_w` factor
   to recover the full Ar16+ spectrum; this factor varies ~26x over the
   sampled Te range (and is intentionally zeroed below `te_min`).

Genuine Poisson photon-counting noise and bundle-count Monte-Carlo noise are
small by comparison (`sqrt(N)/N` scale, and ~1% at `bundle_count=1e4`
respectively) and are NOT the problem being fixed here.

## Design

Compute a per-sample shape integral that captures both factors above, using
the true DESC flux-surface volume (independent of xicsrt's box/bundle
Monte-Carlo sampling):

```
S = integral from rho=0 to 1 of:
    profile_emissivity(rho) * (I_tot/I_w)_with_te_min_cutoff(Te(rho)) * dV/drho(rho) dρ
```

- `profile_emissivity(rho)` and `Te(rho)` come from the sample's own spline
  hooks (`get_emissivity`, `get_temperature_e`) -- the same physics the
  raytrace itself will use.
- `(I_tot/I_w)_with_te_min_cutoff` is the same Ar16+-model factor already
  computed per-bundle in `bundle_generate`, extracted into a shared method
  and evaluated here on a 1-D rho grid instead of per-bundle.
- `dV/drho` comes from DESC's `V_r(r)` compute on the same equilibrium.

This is the large-`bundle_count` / analytic limit of what
`XicsrtPlasmaGeneric.create_sources` currently estimates by Monte Carlo over
bundle box positions, so it reproduces the real physics rather than
approximating it.

Set `emissivity_scale = target_rate / S`, where `target_rate` is an
arbitrary, fixed reference total emitted photon rate [ph/s] (a placeholder
constant, since only the product `emissivity_scale * time_resolution`
matters for the final ray count -- the absolute value is retuned later via
one calibration raytrace). After this:

```
N_expected = time_resolution * (solid_angle / 4*pi) * target_rate
```

is the same for every sample (`spread`, and therefore `solid_angle`, is
already constant across a campaign). `time_resolution` is then hand-tuned
once per campaign to hit the desired image ray count (e.g. ~1e4), exactly as
described by the user.

Explicitly out of scope:
- Detected/imaged ray count. Bragg/rocking-curve efficiency depends on each
  sample's Ti/velocity (Doppler width relative to the crystal acceptance),
  a separate and smaller effect. This plan only normalizes the SOURCE ray
  count.
- Wiring the calibration call into the Part 2b notebook or
  `xicsrt_train_task.py`. Build and verify the library functions only.
- The xarray/padding output-format redesign (F020) -- revisit once
  source-side count consistency is confirmed in practice.

## Implementation

### 1. `xicsrt_analysis/w7x_npablant/sources/_XicsrtPlasmaW7x.py`

- Add the AI-generated-code file disclaimer if not already present (it is;
  keep the existing one, append tool/version if this session's model differs
  materially -- check current header first).
- `__init__`: add `self._flux_vol_rho = None` and `self._flux_vol_vr = None`.
- Extract the existing inline `cold` / `factor` computation in
  `bundle_generate` (current lines ~196-209) into a new method:

  ```python
  def _full_spectrum_factor(self, temperature, temperature_e):
      """
      Return I_tot/I_w (with the te_min cutoff), evaluated at the given
      ion and electron temperatures.

      Extracted from bundle_generate so the same factor can be evaluated
      either per-bundle (during raytracing) or on a 1-D rho grid (during
      emissivity_scale calibration in shape_integral), without the two
      ever drifting apart.
      """
      location, intensity, sigma, gamma, w_mask = self._eval_line_model(
          temperature, temperature_e)
      intensity_w = np.sum(intensity[:, w_mask], axis=1)
      intensity_tot = np.sum(intensity, axis=1)
      cold = temperature_e <= self.param['te_min']
      factor = np.where(
          cold, 0.0, intensity_tot / np.where(cold, 1.0, intensity_w))
      return factor, (location, intensity, sigma, gamma)
  ```

  Note `_eval_line_model` is also needed by `get_line_parameters` via the
  `self._line_cache`, so `bundle_generate` must still cache
  `(location, intensity, sigma, gamma)` from the per-bundle call. Keep
  `_full_spectrum_factor` returning both the factor and the line-model
  tuple so `bundle_generate` can set `self._line_cache` from its return
  value without a second `_eval_line_model` call. Update `bundle_generate`
  to call `_full_spectrum_factor` and use its outputs in place of the
  inline computation; behavior must be unchanged (verify with a numeric
  diff against the current per-bundle emissivity before/after).

- Add flux-surface-volume table caching, mirroring the existing
  `_init_fsa_tables` pattern:

  ```python
  def _init_flux_volume_table(self, n_rho=200):
      """
      Build the dV/drho table used by shape_integral.

      Cached on the instance; computed once per equilibrium on a
      LinearGrid, like _init_fsa_tables. Unlike _init_fsa_tables (which
      starts at rho=0.01 to avoid a DESC singularity in <|grad(rho)|>),
      this starts at rho=0: V_r(r) is finite at the axis (verified
      numerically), and excluding rho=0 would bias the integral against
      axis-peaked emissivity profiles.
      """
      if self._flux_vol_rho is not None:
          return
      self.initialize_vmec()
      rho = np.linspace(0.0, 1.0, n_rho)
      grid = Grid(...)  # LinearGrid(rho=rho, M=self.eq.M_grid, N=self.eq.N_grid, NFP=self.eq.NFP)
      data = self.eq.compute(['V_r(r)'], grid=grid)
      self._flux_vol_rho = rho
      self._flux_vol_vr = grid.compress(np.asarray(data['V_r(r)']))
  ```

- Add the public integral:

  ```python
  def shape_integral(self, n_rho=200):
      """
      Integrate profile_emissivity(rho) * (I_tot/I_w)(Te(rho)) * dV/drho
      over rho, using the real DESC flux-surface volume.

      This is the analytic (bundle_count -> infinity) limit of the
      Monte-Carlo estimate create_sources makes over bundle positions,
      evaluated with this sample's own profile hooks. Used by
      xicsrt_w7x_npablant.calibrate_emissivity_scale to normalize
      emissivity_scale so that every sample's expected source ray count
      depends only on time_resolution, not on the randomized profile
      shape.

      This method was AI generated using Claude (Sonnet 5).
      """
      self._init_flux_volume_table(n_rho)
      rho = self._flux_vol_rho
      emiss = np.broadcast_to(np.asarray(self.get_emissivity(rho)), rho.shape)
      ti = np.broadcast_to(np.asarray(self.get_temperature(rho)), rho.shape)
      te = np.broadcast_to(np.asarray(self.get_temperature_e(rho)), rho.shape)
      factor, _ = self._full_spectrum_factor(ti, te)
      return np.trapezoid(emiss * factor * self._flux_vol_vr, rho)
  ```

  Note: profile hooks may return NaN outside their spline knot range (they
  should not for rho in [0,1] since `XicsrtPlasmaW7xProfile`'s splines
  cover the full range), but guard is not needed if splines are always
  built with knots spanning [0, 1] -- verify this holds for
  `generate_random_*` defaults before assuming it.

### 2. `xicsrt_analysis/w7x_npablant/xicsrt_w7x_npablant.py`

- Remove the dead duplicate `time_resolution` assignment (currently set to
  `1e-4` at one line then overwritten to `1e-2` a few lines later in
  `get_config()`); keep only the final value with its existing comment.
- Add:

  ```python
  # Arbitrary reference full-spectrum photon rate [ph/s] used to normalize
  # emissivity_scale (see calibrate_emissivity_scale). Only the product
  # emissivity_scale * time_resolution sets the actual ray count, so the
  # absolute value here does not matter; time_resolution is retuned by
  # hand after calibration to hit the desired image ray count.
  TARGET_PHOTON_RATE = 1e18

  # Per-process caches for calibrate_emissivity_scale, keyed by wout_file,
  # so repeated calls across samples in the same process do not repeat the
  # ~1.3s DESC equilibrium load or ~2.4s V_r(rho) compute every time.
  _equilibrium_cache = {}
  _volume_table_cache = {}

  def calibrate_emissivity_scale(config, target_rate=TARGET_PHOTON_RATE, n_rho=200):
      """
      Set config['sources']['plasma']['emissivity_scale'] so that this
      sample's shape_integral() equals target_rate.

      Must be called after the profile_* options are set (e.g. after
      update_config_with_profiles). Instantiates a XicsrtPlasmaW7xProfile
      element via xicsrt.xicsrt_public.get_element to evaluate
      shape_integral with the exact physics used by the real raytrace.
      Reuses a cached DESC equilibrium and flux-volume table across calls
      in the same process (keyed by wout_file) to avoid repeating the
      ~1.3s equilibrium load and ~2.4s V_r(rho) compute per sample.

      This function was AI generated using Claude (Sonnet 5).
      """
      from xicsrt.xicsrt_public import get_element

      wout_file = config['sources']['plasma']['wout_file']
      plasma = get_element(config, 'plasma')

      if wout_file in _equilibrium_cache:
          plasma.eq = _equilibrium_cache[wout_file]
      else:
          plasma.initialize_vmec()
          _equilibrium_cache[wout_file] = plasma.eq

      key = (wout_file, n_rho)
      if key in _volume_table_cache:
          plasma._flux_vol_rho, plasma._flux_vol_vr = _volume_table_cache[key]
      else:
          plasma._init_flux_volume_table(n_rho)
          _volume_table_cache[key] = (plasma._flux_vol_rho, plasma._flux_vol_vr)

      scale = target_rate / plasma.shape_integral(n_rho=n_rho)
      config['sources']['plasma']['emissivity_scale'] = scale
      return config
  ```

- Do NOT call `calibrate_emissivity_scale` from `update_config_with_profiles`,
  `get_config`, the Part 2b notebook, or `xicsrt_train_task.py`. Leave those
  call sites untouched; wiring is left to the user for a later session.

### 3. Verification script

New file `xicsrt_analysis/w7x_npablant/verify_ray_calibration.py` (standalone
script, not pytest -- `xicsrt_analysis` has no test suite/config today).

- Part (a), sanity check: for one fixed seed, build a config via
  `get_config` -> `initialize` -> `update_config_with_profiles` ->
  `calibrate_emissivity_scale`, set `use_poisson=False` (deterministic),
  instantiate the plasma via `get_element`, call `plasma.generate_rays()`
  directly (source only, no optics), and compare `len(rays['origin'])`
  against the analytic prediction:
  `time_resolution * (solid_angle/(4*pi)) * target_rate`.
  Report both numbers and the relative difference; expect a close match
  (only bundle-count Monte-Carlo noise, no Poisson noise in this mode).

- Part (b), batch statistics: loop over ~50-100 random seeds with
  `use_poisson=True`, generate + calibrate each config, call
  `generate_rays()`, record `len(rays['origin'])`, and report
  min/median/max/relative standard deviation. Expect the spread to collapse
  from the current ~700x to roughly bundle-count Monte-Carlo noise combined
  with genuine Poisson noise (order a few percent to tens of percent, not
  orders of magnitude).

- Print a clear pass/fail-style summary (no assert-based pytest), since this
  is exploratory verification, not a regression test.

## Verification checklist

- [ ] `_full_spectrum_factor` extraction does not change `bundle_generate`'s
      numeric output (spot check against pre-change values for one seed).
- [ ] `shape_integral` runs without error for at least 3 different random
      profile seeds.
- [ ] Verification script part (a): analytic vs. actual generated ray count
      agree to within a few percent (`use_poisson=False`).
- [ ] Verification script part (b): batch spread across ~50-100 seeds is at
      most a few tens of percent (down from ~700x), reported explicitly to
      the user as part of the session summary.

## Session close

- Update F022 status in `devel/features_request.md` based on the
  verification results; ask the user before marking Done.
- Append a concise entry to `devel/devel_ai_log.txt` per AGENTS.md.
