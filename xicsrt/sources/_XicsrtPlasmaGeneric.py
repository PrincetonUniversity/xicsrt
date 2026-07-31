# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Opus 5, Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Contains the XicsrtPlasmaGeneric class.
"""
import logging

import numpy as np
import scipy.constants as const

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_spread
from xicsrt.tools import xicsrt_voigt_multi
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._GeometryObject import GeometryObject

@dochelper
class XicsrtPlasmaGeneric(GeometryObject):
    """
    A generic plasma object.

    Plasma object will generate a set of ray bundles where each ray bundle
    has the properties of the plasma at one particular real-space point.

    Each bundle is modeled by a SourceFocused object.

    .. Note::
      If a `voxel` type bundle is used rays may be generated outside of the
      defined plasma volume (as defined by xsize, ysize and zsize). The bundle
      *centers* are randomly distributed throughout the plasma volume, but this
      means that if a bundle is (randomly) placed near the edges of the plasma
      then the bundle voxel volume may extend past the plasma boundary. This
      behavior is expected. If it is important to have a sharp plasma boundary
      then consider using the 'point' bundle_type instead.

    **Profile hook convention**

    The profile hooks (`get_emissivity`, `get_temperature`,
    `get_temperature_e`, `get_velocity`) receive full-length arrays of
    length `bundle_count` and return full-length arrays (or scalars, which
    are broadcast); `bundle_generate` applies the bundle mask to the
    result. Hooks must never compress their input by the mask: keeping the
    array shapes fixed between iterations avoids expensive retracing in
    jax-backed implementations (e.g. DESC equilibria).

    The scalar hooks take `rho`, the normalized minor radius. `get_velocity`
    takes `point_flx`, the full flux coordinates `[rho, theta, zeta]`,
    because converting flux-surface velocity profiles into Cartesian
    vectors requires the local geometry, not just rho. In both cases NaN in
    rho (column 0 of `point_flx`) marks points outside the last closed flux
    surface; hooks must propagate NaN so those bundles are masked out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_objects = []

    def default_config(self):
        """
        xsize
          The size of this element along the xaxis direction.

        ysize
          The size of this element along the yaxis direction.

        zsize
          The size of this element along the zaxis direction.

        angular_dist : string ('isotropic')
          The type of angular distribution to use for the emitted rays.
          Only the 'isotropic' distribution is supported for plasma sources.

        spread: float (None) [radians]
          The angular spread for the emission cone. The spread defines the
          half-angle of the cone. See 'angular_dist' in :any:`XicsrtSourceGeneric`
          for detailed documentation.

        spread_radius: float (None) [meters]
          If specified, the spread will be calculated for each bundle such that
          the spotsize at the target matches the given radius. This is useful
          when working with very extended plasma sources.
          This options is incompatible with 'spread'.

        use_poisson
          No documentation yet. Please help improve XICSRT!

        wavelength_dist : string ('voigt')
          No documentation yet. Please help improve XICSRT!

        wavelength : float (1.0) [Angstroms]
          No documentation yet. Please help improve XICSRT!

        mass_number : float (1.0) [au]
          No documentation yet. Please help improve XICSRT!

        linewidth : float (0.0) [1/s]
          No documentation yet. Please help improve XICSRT!

        emissivity : float (0.0) [ph/m^3]
          No documentation yet. Please help improve XICSRT!

        temperature : float (0.0) [eV]
          Ion temperature. Will be used to calculate the doppler broadening.

        temperature_e : float (0.0) [eV]
          Electron temperature. May be used by some models to determine line emissivities.

        velocity : float (0.0) [m/s]
          No documentation yet. Please help improve XICSRT!

        time_resolution : float (1e-3) [s]
          No documentation yet. Please help improve XICSRT!

        bundle_type : string ('voxel')
          Define how the origin of rays within the bundle should be distributed.
          Available options are: 'voxel' or 'point'.

        bundle_volume : float (1e-3) [m^3]
          The volume in which the rays within the bundle should distributed.
          if bundle_type is 'point' this will not affect the distribution,
          though it will still affect the number of bundles if bundle_count
          is set to None.

        bundle_count : int (None)
          The number of bundles to generate. If set to `None` then this number
          will be automatically determined by volume/bundle_volume. This default
          means that each bundle represents exactly the given `bundle_volume` in
          the plasma. For high quality raytracing studies this value should
          generally be set to a value much larger than volume/bundle_volume!

        max_rays : int (1e7)
          No documentation yet. Please help improve XICSRT!

        max_bundles : int (1e7)
          No documentation yet. Please help improve XICSRT!

        filters
          No documentation yet. Please help improve XICSRT!

        """
        config = super().default_config()
                
        config['xsize']          = 0.0
        config['ysize']         = 0.0
        config['zsize']          = 0.0

        config['angular_dist']      = 'isotropic'
        config['spread']           = None
        config['spread_radius']    = None
        config['target']           = None
        config['use_poisson']      = False

        config['wavelength_dist']  = 'voigt'
        config['wavelength']       = 1.0
        config['wavelength_range'] = None
        config['mass_number']      = 1.0
        config['linewidth']        = 0.0

        # Only used for wavelength_dist = 'multi_voigt'.
        # See get_line_parameters for how these are consumed; subclasses
        # may override that hook to compute per-bundle line parameters.
        config['line_locations']   = np.array([1.0])
        config['line_intensities'] = np.array([1.0])
        config['line_sigmas']      = np.array([0.0])
        config['line_gammas']      = np.array([0.0])

        config['emissivity']      = 0.0
        config['temperature']     = 0.0
        config['temperature_e']   = 0.0
        config['velocity']        = 0.0

        config['time_resolution'] = 1e-3
        config['bundle_type']     = 'voxel'
        config['bundle_volume']   = 1e-6
        config['bundle_count']    = None
        config['max_rays']        = int(1e7)
        config['max_bundles']     = int(1e7)
        
        config['filters']         = []
        return config

    def initialize(self):
        super().initialize()
        if self.param['max_rays'] is not None:
            self.param['max_rays']     = int(self.param['max_rays'])
        self.param['volume']       = self.config['xsize'] * self.config['ysize'] * self.config['zsize']

        if self.param['bundle_count'] is None:
            self.param['bundle_count'] = self.param['volume']/self.param['bundle_volume']
        self.param['bundle_count'] = int(np.round(self.param['bundle_count']))
        if self.param['bundle_count'] < 1:
            raise Exception(f'Bundle volume is larger than the plasma volume.')
        if self.param['bundle_count'] > self.param['max_bundles']:
            raise ValueError(
                f"Current settings will produce too many bundles ({self.param['bundle_count']:0.2e}). "
                f"Increase the bundle_volume, explicitly set bundle_count or increase max_bundles.")

    def setup_bundles(self):
        self.log.debug('Starting setup_bundles')
        if self.param['bundle_type'] == 'point':
            self.param['voxel_size'] = 0.0
        elif self.param['bundle_type'] == 'voxel':
            self.param['voxel_size'] = self.param['bundle_volume'] ** (1/3)

        # These values should be overwritten in a derived class.
        bundle_input = {}
        bundle_input['origin']         = np.zeros([self.param['bundle_count'], 3], dtype = np.float64)
        bundle_input['temperature']    = np.zeros([self.param['bundle_count']], dtype = np.float64)
        bundle_input['temperature_e']  = np.zeros([self.param['bundle_count']], dtype = np.float64)
        bundle_input['emissivity']     = np.ones([self.param['bundle_count']], dtype = np.float64)
        bundle_input['velocity']       = np.zeros([self.param['bundle_count'], 3], dtype = np.float64)
        bundle_input['mask']           = np.ones([self.param['bundle_count']], dtype = np.bool_)
        bundle_input['spread']         = np.zeros([self.param['bundle_count']], dtype = np.float64)
        bundle_input['solid_angle']    = np.zeros([self.param['bundle_count']], dtype = np.float64)
        
        # randomly spread the bundles around the plasma box
        offset = np.zeros((self.param['bundle_count'], 3))
        offset[:,0] = np.random.uniform(-1 * self.param['xsize'] /2, self.param['xsize'] /2, self.param['bundle_count'])
        offset[:,1] = np.random.uniform(-1 * self.param['ysize']/2, self.param['ysize']/2, self.param['bundle_count'])
        offset[:,2] = np.random.uniform(-1 * self.param['zsize'] /2, self.param['zsize'] /2, self.param['bundle_count'])

        bundle_input['origin'][:] = self.point_to_external(offset)

        # Setup the bundle spread and solid angle.
        bundle_input = self.setup_bundle_spread(bundle_input)

        return bundle_input

    def setup_bundle_spread(self, bundle_input):
        """
        Calculate the spread and solid angle for each bundle.

        If the config option 'spread_radius' is provide the spread will be
        determined for each bundle by a spotsize at the target.

        Note: Even if the idea of a spread radius is added to the generic
              source object we still need to calculate and save the results
              here so that we can correctly calcuate the bundle intensities.
        """
        if self.param['spread_radius'] is not None:
            vector = bundle_input['origin'] - self.param['target']
            dist = np.linalg.norm(vector, axis=1)
            spread = np.arctan(self.param['spread_radius']/dist)
        else:
            spread = self.param['spread']

        bundle_input['spread'][:] = spread
        bundle_input['solid_angle'][:] = xicsrt_spread.solid_angle_isotropic(
            bundle_input['spread'])

        return bundle_input

    def get_emissivity(self, rho):
        """
        Emissivity profile hook. See the class docstring for the hook
        convention (full-length arrays in, full-length arrays out).
        """
        return self.param['emissivity']

    def get_temperature(self, rho):
        """
        Ion temperature profile hook [eV]. See the class docstring for the
        hook convention.
        """
        return self.param['temperature']

    def get_temperature_e(self, rho):
        """
        Electron temperature profile hook [eV]. See the class docstring for
        the hook convention.
        """
        return self.param['temperature_e']

    def get_velocity(self, point_flx):
        """
        Velocity profile hook [m/s].

        Parameters
        ----------
        point_flx : ndarray, shape (bundle_count, 3)
            Flux coordinates [rho, theta, zeta] of the bundle origins.
            Column 0 is rho, with NaN marking points outside the last
            closed flux surface. See the class docstring for the hook
            convention.

        Returns
        -------
        ndarray, shape (bundle_count, 3) or scalar
            Cartesian velocity vectors.
        """
        return self.param['velocity']

    def bundle_generate(self, bundle_input):
        self.log.debug('Starting bundle_generate')
        return bundle_input

    def bundle_filter(self, bundle_input):
        self.log.debug('Starting bundle_filter')
        for filter in self.filter_objects:
            bundle_input = filter.filter(bundle_input)
        return bundle_input
    
    def get_line_parameters(self, bundle_input, m, bundle_index):
        """
        Return per-bundle multiline Voigt parameters for wavelength sampling.

        This hook is used when ``wavelength_dist = 'multi_voigt'``. The base
        implementation broadcasts the static line parameters from the config
        options (`line_locations`, `line_intensities`, `line_sigmas`,
        `line_gammas`) to every bundle. Subclasses may override this to
        compute the spectral line list from the local plasma parameters
        (for example from the per-bundle ion and electron temperatures).

        Parameters
        ----------
        bundle_input : dict
            The bundle input dictionary (full length ``bundle_count``).
        m : ndarray of bool
            The bundle mask. Line parameters are only needed for masked
            (active) bundles.
        bundle_index : ndarray of int, shape (n_rays,)
            For each ray, the index of its bundle within the *masked*
            bundle arrays (i.e. an index into ``bundle_input['origin'][m]``).

        Returns
        -------
        tuple of ndarray
            Arrays ``(locations, intensities, sigmas, gammas)``, each of
            shape (n_masked_bundles, n_lines), consumed by
            :func:`xicsrt.tools.xicsrt_voigt_multi.multi_voigt_random_batched`.

        This method was AI generated using Claude (Fable 5).
        """
        n_bundles = int(np.sum(m))
        locations = np.broadcast_to(
            np.asarray(self.param['line_locations'], dtype=np.float64),
            (n_bundles, len(self.param['line_locations'])))
        intensities = np.broadcast_to(
            np.asarray(self.param['line_intensities'], dtype=np.float64),
            locations.shape)
        sigmas = np.broadcast_to(
            np.asarray(self.param['line_sigmas'], dtype=np.float64),
            locations.shape)
        gammas = np.broadcast_to(
            np.asarray(self.param['line_gammas'], dtype=np.float64),
            locations.shape)
        return locations, intensities, sigmas, gammas

    def create_sources(self, bundle_input):
        """
        Generate rays from the bundle list in a single vectorized pass.

        Every active bundle emits a Poisson-distributed number of rays
        (exact photon statistics), all of which are generated together:
        origins, focused directions, wavelengths and Doppler shifts are
        computed with per-ray array operations using each ray's own
        bundle parameters. This is statistically identical to the previous
        per-bundle XicsrtSourceFocused loop, but not bit-identical for a
        given random seed.

        Parameters
        ----------
        bundle_input : dict
            Dictionary of arrays describing the locations, emissivities,
            temperatures and velocities of all ray bundles to be emitted.

        This method was AI generated using Claude (Fable 5).
        """
        m = bundle_input['mask']

        # Calculate the expected number of photons from each bundle volume.
        #
        # We allow bundle_volume and bundle_count to be independent, which
        # means that a bundle representing a volume in the plasma can be
        # launched from a virtual volume of a different size. In order to
        # allow this while maintaining overall photon statistics from the
        # plasma, we normalize the intensity so that each bundle represents
        # a volume of plasma_volume/bundle_count. In doing so bundle_volume
        # cancels out, but the calculation is left separate for clarity.
        intensity = (bundle_input['emissivity'][m]
                     * self.param['time_resolution']
                     * self.param['bundle_volume']
                     * bundle_input['solid_angle'][m] / (4 * np.pi))
        intensity *= (self.param['volume']
                      / (self.param['bundle_count'] * self.param['bundle_volume']))

        # Check if the number of rays generated will exceed max ray limits.
        # This is only approximate since poisson statistics may be in use.
        predicted_rays = int(np.sum(intensity))
        self.log.debug(f'Predicted rays: {predicted_rays:0.2e}')
        if self.param['max_rays']:
            if predicted_rays > self.param['max_rays']:
                raise ValueError(
                    f"Current settings will produce too many rays ({predicted_rays:0.2e}). "
                    f"Please reduce integration time or adjust other parameters.")

        profiler.start("Ray Bundle Generation")

        # The number of rays from each bundle: exact Poisson statistics.
        if self.param['use_poisson']:
            counts = np.random.poisson(intensity)
        else:
            if np.any(intensity < 1):
                raise ValueError(
                    'intensity of less than one encountered. Turn on poisson statistics.')
            counts = intensity.astype(np.int64)

        # For each ray, the index of its bundle within the masked arrays.
        bundle_index = np.repeat(np.arange(len(counts)), counts)
        total_rays = len(bundle_index)

        rays = dict()
        profiler.start('generate_origin')
        rays['origin'] = self._generate_origins(bundle_input, m, bundle_index)
        profiler.stop('generate_origin')

        profiler.start('generate_direction')
        rays['direction'] = self._generate_directions(
            rays['origin'], bundle_input, m, bundle_index)
        profiler.stop('generate_direction')

        profiler.start('generate_wavelength')
        rays['wavelength'] = self._generate_wavelengths(
            bundle_input, m, bundle_index)

        # Doppler shift from the per-bundle plasma velocity.
        velocity = np.asarray(bundle_input['velocity'][m], dtype=np.float64)
        if np.any(velocity != 0.0):
            c = const.physical_constants['speed of light in vacuum'][0]
            v_per_ray = velocity[bundle_index]
            rays['wavelength'] *= (
                1 - np.einsum('ij,ij->i', v_per_ray, rays['direction']) / c)
        profiler.stop('generate_wavelength')

        rays['weight'] = np.ones(total_rays, dtype=np.float64)
        rays['mask'] = np.ones(total_rays, dtype=np.bool_)

        profiler.stop("Ray Bundle Generation")

        if total_rays == 0:
            raise ValueError('No rays generated. Check plasma input parameters')

        self.log.debug('Bundles Generated:       {:0.4e}'.format(
            len(m[m])))
        self.log.debug('Rays per bundle, mean:   {:0.0f}'.format(
            np.mean(counts)))
        self.log.debug('Rays per bundle, median: {:0.0f}'.format(
            np.median(counts)))
        self.log.debug('Rays per bundle, max:    {:0d}'.format(
            np.max(counts)))
        self.log.debug('Rays per bundle, min:    {:0d}'.format(
            np.min(counts)))

        return rays

    def _generate_origins(self, bundle_input, m, bundle_index):
        """
        Generate per-ray origins from the bundle origins.

        For 'point' bundles all rays start at the bundle origin. For 'voxel'
        bundles a uniform offset within the voxel cube (aligned with the
        plasma axes) is added.

        This method was AI generated using Claude (Fable 5).
        """
        origins = bundle_input['origin'][m][bundle_index]

        voxel_size = self.param['voxel_size']
        if voxel_size > 0.0:
            total_rays = len(bundle_index)
            offset = np.random.uniform(
                -voxel_size/2, voxel_size/2, (total_rays, 3))
            origins = (origins
                       + np.einsum('i,j', offset[:, 0], self.xaxis)
                       + np.einsum('i,j', offset[:, 1], self.yaxis)
                       + np.einsum('i,j', offset[:, 2], self.zaxis))
        return origins

    def _generate_directions(self, origins, bundle_input, m, bundle_index):
        """
        Generate per-ray focused directions.

        Each ray's emission cone is aimed from its origin at the target
        (matching XicsrtSourceFocused) with that ray's own bundle spread.

        This method was AI generated using Claude (Fable 5).
        """
        if str.lower(self.param['angular_dist']) != 'isotropic':
            raise NotImplementedError(
                "Only the 'isotropic' angular_dist is supported for plasma sources.")

        # Per-ray focused normal: from origin towards the target.
        normal = self.param['target'] - origins
        normal = normal / np.sqrt(
            np.einsum('ij,ij->i', normal, normal))[:, np.newaxis]

        spread = bundle_input['spread'][m][bundle_index]
        dir_local = xicsrt_spread.vector_dist_isotropic(spread, len(bundle_index))

        # Generate basis vectors perpendicular to the per-ray normal and
        # project the local directions onto them. This matches the basis
        # construction in XicsrtSourceGeneric.random_direction.
        o_1 = (np.cross(normal, self.param['xaxis'])
               + np.cross(normal, self.param['zaxis']))
        o_1 = xm.normalize(o_1)
        o_2 = xm.normalize(np.cross(normal, o_1))

        direction = (dir_local[:, 0:1] * o_2
                     + dir_local[:, 1:2] * o_1
                     + dir_local[:, 2:3] * normal)
        return direction

    def _generate_wavelengths(self, bundle_input, m, bundle_index):
        """
        Generate per-ray wavelengths using each ray's bundle parameters.

        For the 'voigt' distribution the Gaussian width is computed from the
        per-bundle ion temperature; the Voigt variate is sampled directly as
        Normal + Cauchy (exact, handles zero temperature or linewidth).
        For 'multi_voigt' the per-bundle line parameters are provided by the
        get_line_parameters() hook and sampled with the batched mixture
        sampler.

        This method was AI generated using Claude (Fable 5).
        """
        total_rays = len(bundle_index)
        wtype = str.lower(self.param['wavelength_dist'])

        if wtype == 'monochrome':
            wavelength = np.full(total_rays, self.param['wavelength'],
                                 dtype=np.float64)

        elif wtype == 'uniform':
            wavelength = np.random.uniform(
                self.param['wavelength_range'][0],
                self.param['wavelength_range'][1],
                total_rays)

        elif wtype == 'voigt':
            c = const.physical_constants['speed of light in vacuum'][0]
            amu_kg = const.physical_constants['atomic mass unit-kilogram relationship'][0]
            ev_j = const.physical_constants['electron volt-joule relationship'][0]

            # Natural line width (identical for all bundles).
            gamma = (self.param['linewidth'] * self.param['wavelength']**2
                     / (4 * np.pi * c * 1e10))

            # Doppler broadened line width from the per-bundle temperature.
            temperature = bundle_input['temperature'][m]
            sigma = (np.sqrt(temperature / self.param['mass_number']
                             / amu_kg / c**2 * ev_j)
                     * self.param['wavelength'])

            # Direct Voigt sampling: center + Normal(0, sigma) + Cauchy(0, gamma).
            wavelength = np.full(total_rays, self.param['wavelength'],
                                 dtype=np.float64)
            wavelength += np.random.normal(0.0, 1.0, total_rays) * sigma[bundle_index]
            if gamma != 0.0:
                wavelength += np.random.standard_cauchy(total_rays) * gamma

        elif wtype == 'multi_voigt':
            locations, intensities, sigmas, gammas = self.get_line_parameters(
                bundle_input, m, bundle_index)
            wavelength = xicsrt_voigt_multi.multi_voigt_random_batched(
                locations, intensities, sigmas, gammas, bundle_index)

        else:
            raise Exception(f'Wavelength distribution {wtype} unknown')

        return wavelength

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Apply filters to filter out ray bundles
        bundle_input = self.bundle_filter(bundle_input)  
        ## Populate that list with ray bundle parameters, like emissivity
        bundle_input = self.bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays
