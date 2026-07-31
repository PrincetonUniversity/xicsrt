# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Sonnet 5).
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Contains the XicsrtPlasmaBundleSource class.
"""
import numpy as np

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_spread
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._Dispatcher import find_xicsrt_class
from xicsrt.sources._XicsrtPlasmaGeneric import XicsrtPlasmaGeneric


@dochelper
class XicsrtPlasmaBundleSource(XicsrtPlasmaGeneric):
    """
    A plasma object that models every bundle with a dispatched ray source.

    .. Note::
      This class is meant to be used as a worked example, not as a
      production plasma model. `XicsrtPlasmaGeneric` generates every bundle
      with a single vectorized pass and should be preferred whenever its
      built-in behavior (isotropic emission, per-bundle scalar spread) is
      sufficient; it is both faster and simpler to reason about.

    Why this class exists
    ----------------------
    `XicsrtPlasmaGeneric.create_sources` hard-codes the physics of
    `XicsrtSourceFocused` directly into a vectorized implementation for
    performance. That means a user who has written their own ray source
    (for example one with a novel spatial or angular distribution) has no
    way to use it for plasma bundle generation, even though conceptually
    nothing prevents it: a bundle is nothing more than a tiny ray source
    located at one point in the plasma.

    This class restores that coupling. For every active bundle it
    instantiates a fresh ray-source object (selected by the config option
    `bundle_source_class`, resolved by name through the same dispatcher
    mechanism used for `sources`/`optics`/`filters` in the main config) and
    calls its `generate_rays()` method. The rays produced by every bundle
    are then concatenated into a single ray array. Because a new object is
    created for every bundle, this is much slower than
    `XicsrtPlasmaGeneric`, but the implementation is short, linear and easy
    to adapt.

    Writing a compatible ray source
    --------------------------------
    Any class dispatchable by `class_name` (i.e. living in a file
    `_Xicsrt<Name>.py` on the search path, see `general.pathlist`) can be
    used as `bundle_source_class`, as long as it:

    - derives (directly or indirectly) from `XicsrtSourceGeneric`, or at
      least implements a compatible `generate_rays()`;
    - accepts the config options `origin`, `xsize`/`ysize`/`zsize`,
      `zaxis`/`xaxis`, `intensity` and `use_poisson`. These are set for
      every bundle by this class (see `build_bundle_source_config`) and
      control where and how many rays are produced.

    All other plasma options (`spread`, `wavelength`, `temperature`,
    `velocity`, `target`, ...) are forwarded too, but with
    `strict=False`: any option your source class does not define is simply
    dropped rather than raising an error. This is what makes an arbitrary
    ray source "just work" without editing this file, but it also means a
    misspelled option name in `bundle_source_config` will be silently
    ignored instead of raising, since strict checking only ever applies to
    class objects with a known set of options. Double check option names
    if a run isn't behaving as expected.

    Limitations
    -----------
    Only `angular_dist = 'isotropic'` supports a per-bundle spread that
    depends on bundle location (`spread_radius`), since the bundle
    intensity normalization requires a matching per-bundle solid angle
    (see `setup_bundle_spread`). The `multi_voigt` wavelength distribution
    is not supported here since its per-bundle line-parameter hook
    (`XicsrtPlasmaGeneric.get_line_parameters`) has no equivalent in a
    per-bundle source loop; use `XicsrtPlasmaGeneric` (or a subclass) for
    that instead.
    """

    def default_config(self):
        """
        bundle_source_class : string ('XicsrtSourceFocused')
          The `class_name` of the ray source used to generate rays for each
          individual bundle. This class is resolved with the same
          dispatcher search paths used for the `sources`/`optics`/`filters`
          config sections (built-in paths plus `general.pathlist`), so a
          user-defined ray source class can be used here without any
          change to this file.

        bundle_source_config : dict (None)
          Additional config options that will be merged into the config
          used to construct every bundle's ray source, after the plasma
          options have been forwarded (see class docstring). Use this to
          set options specific to `bundle_source_class` that have no
          equivalent plasma-level option (for example
          `XicsrtSourceDirected.direction`).

          .. Note::
            The default is `None` rather than an empty dict so that
            `strict_config_check` does not recurse into (and therefore
            reject unknown keys of) a user-supplied dictionary here; see
            `xicsrt.xicsrt_config.update_config`. This mirrors how
            `aperture` is handled in `TraceObject.default_config`.
        """
        config = super().default_config()

        config['bundle_source_class'] = 'XicsrtSourceFocused'
        config['bundle_source_config'] = None

        # These options only make sense for a vectorized multiline sampler
        # keyed on a bundle_index array (see
        # XicsrtPlasmaGeneric.get_line_parameters) and have no equivalent
        # for a per-bundle source loop.
        del config['line_locations']
        del config['line_intensities']
        del config['line_sigmas']
        del config['line_gammas']

        return config

    def check_param(self):
        super().check_param()
        if str.lower(self.param['wavelength_dist']) == 'multi_voigt':
            raise NotImplementedError(
                "wavelength_dist = 'multi_voigt' requires a per-bundle line "
                "table (see XicsrtPlasmaGeneric.get_line_parameters), which "
                "has no equivalent here. Use XicsrtPlasmaGeneric instead.")

    def initialize(self):
        super().initialize()

        # Resolve the bundle source class once, rather than on every
        # bundle, so that a bad class_name fails immediately instead of
        # after a possibly long bundle-generation loop.
        self.bundle_source_cls = find_xicsrt_class(
            self.param['pathlist'], self.param['bundle_source_class'])

    def setup_bundle_spread(self, bundle_input):
        """
        Calculate the spread and solid angle for each bundle.

        This overrides `XicsrtPlasmaGeneric.setup_bundle_spread`, which
        always assumes an isotropic emission cone (and therefore always
        uses `xicsrt_spread.solid_angle_isotropic`) regardless of
        `angular_dist`. Since this class allows an arbitrary ray source
        with an arbitrary `angular_dist`, the matching solid angle formula
        from `xicsrt_spread.solid_angle` is used instead; this raises a
        clear error for any distribution that has no known solid angle
        formula ('flat', 'flat_xy', 'gaussian') rather than silently
        producing an incorrect absolute intensity.

        Only the 'isotropic' distribution supports a per-bundle spread
        (`spread_radius`, an angular spread chosen so that a fixed spot
        size is projected at the target). The other supported distribution,
        'isotropic_xy', uses a fixed multi-value spread (see
        `xicsrt_spread.vector_dist_isotropic_xy`) that is shared by every
        bundle, so a single shared solid angle is used instead.
        """
        angular_dist = str.lower(self.param['angular_dist'])

        if angular_dist == 'isotropic':
            if self.param['spread_radius'] is not None:
                vector = bundle_input['origin'] - self.param['target']
                dist = np.linalg.norm(vector, axis=1)
                spread = np.arctan(self.param['spread_radius'] / dist)
            else:
                spread = self.param['spread']
            bundle_input['spread'][:] = spread
            bundle_input['solid_angle'][:] = xicsrt_spread.solid_angle(
                spread, name=angular_dist)
        else:
            if self.param['spread_radius'] is not None:
                raise NotImplementedError(
                    "'spread_radius' requires a per-bundle scalar spread and "
                    "is only supported for angular_dist = 'isotropic'.")
            bundle_input['spread'][:] = 0.0
            bundle_input['solid_angle'][:] = xicsrt_spread.solid_angle(
                self.param['spread'], name=angular_dist)

        return bundle_input

    def bundle_generate(self, bundle_input):
        """
        Populate the plasma parameters for every bundle.

        This default implementation gives every bundle the same, constant
        emissivity, temperature and velocity (i.e. a uniform plasma cube,
        equivalent to `XicsrtPlasmaCubic`). Override this method to
        implement a specific plasma geometry, exactly as is done for
        `XicsrtPlasmaGeneric` subclasses such as `XicsrtPlasmaToroidal`.
        """
        bundle_input = super().bundle_generate(bundle_input)
        bundle_input['temperature'][:] = self.param['temperature']
        bundle_input['temperature_e'][:] = self.param['temperature_e']
        bundle_input['emissivity'][:] = self.param['emissivity']
        bundle_input['velocity'][:] = self.param['velocity']
        return bundle_input

    def build_bundle_source_config(self, bundle_input, ii):
        """
        Build the config dictionary used to construct the ray source for
        bundle `ii`.

        The full plasma `param` dictionary is forwarded first (with
        `strict=False`, so options the source class does not define are
        simply dropped), then overwritten with the per-bundle geometry and
        plasma values, and finally overwritten with any user-supplied
        `bundle_source_config`. See the class docstring for the rationale.
        """
        source_config = dict(self.param)

        # These keys either do not belong on the source (they describe the
        # plasma object itself) or are set explicitly below/should not be
        # copied from the plasma.
        for key in ('class_name', 'filters', 'pathlist',
                    'bundle_source_class', 'bundle_source_config'):
            source_config.pop(key, None)

        # Per-bundle geometry: each bundle is a tiny source located at its
        # own origin, with the plasma's orientation and a cubic voxel of
        # side `voxel_size` (zero for 'point' bundles).
        source_config['origin'] = bundle_input['origin'][ii]
        source_config['xsize'] = self.param['voxel_size']
        source_config['ysize'] = self.param['voxel_size']
        source_config['zsize'] = self.param['voxel_size']
        source_config['zaxis'] = self.param['zaxis']
        source_config['xaxis'] = self.param['xaxis']

        # Per-bundle plasma values.
        source_config['temperature'] = bundle_input['temperature'][ii]
        source_config['temperature_e'] = bundle_input['temperature_e'][ii]
        source_config['velocity'] = bundle_input['velocity'][ii]
        source_config['intensity'] = self._bundle_intensity(bundle_input, ii)

        # Only an isotropic bundle has a meaningful per-bundle scalar
        # spread (see setup_bundle_spread); for every other angular_dist
        # the shared array-valued spread already in self.param is used.
        if str.lower(self.param['angular_dist']) == 'isotropic':
            source_config['spread'] = bundle_input['spread'][ii]

        if self.param['bundle_source_config'] is not None:
            source_config.update(self.param['bundle_source_config'])

        return source_config

    def _bundle_intensity(self, bundle_input, ii):
        """
        Calculate the number of photons to launch from a single bundle.

        This uses the identical normalization as
        `XicsrtPlasmaGeneric.create_sources`: bundles represent a plasma
        volume of `volume/bundle_count`, independent of `bundle_volume`
        (see the note there for the full derivation).
        """
        intensity = (bundle_input['emissivity'][ii]
                     * self.param['time_resolution']
                     * self.param['bundle_volume']
                     * bundle_input['solid_angle'][ii] / (4 * np.pi))
        intensity *= (self.param['volume']
                      / (self.param['bundle_count'] * self.param['bundle_volume']))
        return intensity

    def create_sources(self, bundle_input):
        """
        Generate rays for every active bundle using a dispatched ray
        source, then concatenate the results into a single ray array.
        """
        self.log.debug('Starting create_sources')

        bundle_index = np.flatnonzero(bundle_input['mask'])

        # Check if the number of rays generated will exceed max ray limits
        # before doing any work. This is only approximate since poisson
        # statistics may be in use.
        predicted_rays = int(sum(
            self._bundle_intensity(bundle_input, ii) for ii in bundle_index))
        self.log.debug(f'Predicted rays: {predicted_rays:0.2e}')
        if self.param['max_rays']:
            if predicted_rays > self.param['max_rays']:
                raise ValueError(
                    f"Current settings will produce too many rays ({predicted_rays:0.2e}). "
                    f"Please reduce integration time or adjust other parameters.")

        rays_list = []
        for ii in bundle_index:
            profiler.start('Ray Bundle Generation')
            source_config = self.build_bundle_source_config(bundle_input, ii)
            source = self.bundle_source_cls(source_config, strict=False)
            rays_list.append(source.generate_rays())
            profiler.stop('Ray Bundle Generation')

        if len(rays_list) == 0:
            raise ValueError('No rays generated. Check plasma input parameters')

        # Concatenate on the keys produced by the source rather than a fixed
        # list, so that a source emitting additional ray properties is not
        # silently truncated.
        profiler.start('Ray Bundle Collection')
        rays = dict()
        for key in rays_list[0]:
            rays[key] = np.concatenate([r[key] for r in rays_list])
        profiler.stop('Ray Bundle Collection')

        if len(rays['mask']) == 0:
            raise ValueError('No rays generated. Check plasma input parameters')

        counts = [len(r['mask']) for r in rays_list]
        self.log.debug('Bundles Generated:       {:0.4e}'.format(len(rays_list)))
        self.log.debug('Rays per bundle, mean:   {:0.0f}'.format(np.mean(counts)))
        self.log.debug('Rays per bundle, median: {:0.0f}'.format(np.median(counts)))
        self.log.debug('Rays per bundle, max:    {:0d}'.format(np.max(counts)))
        self.log.debug('Rays per bundle, min:    {:0d}'.format(np.min(counts)))

        return rays
