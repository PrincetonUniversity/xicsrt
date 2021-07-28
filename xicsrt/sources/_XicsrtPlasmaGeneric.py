# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Contains the XicsrtPlasmaGeneric class.
"""
import logging

import numpy as np

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_spread
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._GeometryObject import GeometryObject
from xicsrt.sources._XicsrtSourceFocused import XicsrtSourceFocused

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
          Available distributions: 'isotropic', 'isotropic_xy', 'flat',
          'flat_xy', 'gaussian', and 'gaussian_flat'.
          See `XicsrtSourceGeneric` for documentation of each distribution.

          Warning: Only the 'isotropic' distribution is currently supported!

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
          No documentation yet. Please help improve XICSRT!

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

        config['emissivity']      = 0.0
        config['temperature']     = 0.0
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
        bundle_input['origin']       = np.zeros([self.param['bundle_count'], 3], dtype = np.float64)
        bundle_input['temperature']  = np.ones([self.param['bundle_count']], dtype = np.float64)
        bundle_input['emissivity']   = np.ones([self.param['bundle_count']], dtype = np.float64)
        bundle_input['velocity']     = np.zeros([self.param['bundle_count'], 3], dtype = np.float64)
        bundle_input['mask']         = np.ones([self.param['bundle_count']], dtype = np.bool)
        bundle_input['spread']       = np.zeros([self.param['bundle_count']], dtype = np.float64)
        bundle_input['solid_angle']  = np.zeros([self.param['bundle_count']], dtype = np.float64)
        
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

        # For the time being the fuction solid_angle is not vectorized, so a
        # loop is necessary.
        for ii in range(len(bundle_input['spread'])):
            bundle_input['solid_angle'][ii] = xicsrt_spread.solid_angle(bundle_input['spread'][ii])

        return bundle_input

    def get_emissivity(self, rho):
        return self.param['emissivity']

    def get_temperature(self, rho):
        return self.param['temperature']

    def get_velocity(self, rho):
        return self.param['velocity']

    def bundle_generate(self, bundle_input):
        self.log.debug('Starting bundle_generate')
        return bundle_input

    def bundle_filter(self, bundle_input):
        self.log.debug('Starting bundle_filter')
        for filter in self.filter_objects:
            bundle_input = filter.filter(bundle_input)
        return bundle_input
    
    def create_sources(self, bundle_input):
        """
        Generate rays from a list of bundles.

        bundle_input
          a list containing dictionaries containing the locations, emissivities,
          temperatures and velocitities and of all ray bundles to be emitted.
        """

        rays_list = []
        count_rays_in_bundle = []

        m = bundle_input['mask']

        # Check if the number of rays generated will exceed max ray limits.
        # This is only approximate since poisson statistics may be in use.

        predicted_rays = int(np.sum(
            bundle_input['emissivity'][m]
            * self.param['time_resolution']
            * self.param['bundle_volume']
            * bundle_input['solid_angle'][m] / (4 * np.pi)
            * self.param['volume']
            / (self.param['bundle_count'] * self.param['bundle_volume'])))

        self.log.debug(f'Predicted rays: {predicted_rays:0.2e}')
        if predicted_rays > self.param['max_rays']:
            raise ValueError(
                f"Current settings will produce too many rays ({predicted_rays:0.2e}). "
                f"Please reduce integration time or adjust other parameters.")

        # Bundle generation loop
        for ii in range(self.param['bundle_count']):
            if not bundle_input['mask'][ii]:
                continue
            profiler.start("Ray Bundle Generation")
            source_config = dict()
            
            # Specially dependent parameters
            source_config['origin']      = bundle_input['origin'][ii]
            source_config['temperature'] = bundle_input['temperature'][ii]
            source_config['velocity']    = bundle_input['velocity'][ii]
            source_config['spread']      = bundle_input['spread'][ii]

            # Calculate the total number of photons to launch from this bundle
            # volume. Since the source can use poisson statistics, this should
            # be of floating point type.
            intensity = (bundle_input['emissivity'][ii]
                         * self.param['time_resolution']
                         * self.param['bundle_volume']
                         * bundle_input['solid_angle'][ii] / (4 * np.pi))
            
            # Scale the number of photons based on the number of bundles.
            #
            # Ultimately we allow bundle_volume and bundle_count to be
            # independent, which means that a bundle representing a volume in
            # the plasma can be launched from virtual volume of a different
            # size.
            #
            # In order to allow this while maintaining overall photon statistics
            # from the plasma, we normalize the intensity so that each bundle
            # represents a volume of plasma_volume/bundle_count.
            #
            # In doing so bundle_volume cancels out, but I am leaving the
            # calculation separate for clarity.
            intensity *= self.param['volume'] / (self.param['bundle_count'] * self.param['bundle_volume'])
            
            source_config['intensity'] = intensity

            # constants
            source_config['xsize']            = self.param['voxel_size']
            source_config['ysize']            = self.param['voxel_size']
            source_config['zsize']            = self.param['voxel_size']
            source_config['zaxis']            = self.param['zaxis']
            source_config['xaxis']            = self.param['xaxis']
            source_config['target']           = self.param['target']
            source_config['mass_number']      = self.param['mass_number']
            source_config['wavelength_dist']  = self.param['wavelength_dist']
            source_config['wavelength']       = self.param['wavelength']
            source_config['wavelength_range'] = self.param['wavelength_range']
            source_config['linewidth']        = self.param['linewidth']
            source_config['angular_dist']      = self.param['angular_dist']
            source_config['use_poisson']      = self.param['use_poisson']
                
            #create ray bundle sources and generate bundled rays
            source       = XicsrtSourceFocused(source_config)
            bundled_rays = source.generate_rays()

            rays_list.append(bundled_rays)
            count_rays_in_bundle.append(len(bundled_rays['mask']))

            profiler.stop("Ray Bundle Generation")

        profiler.start('Ray Bundle Collection')
        # append bundled rays together to form a single ray dictionary.    
        # create the final ray dictionary
        total_rays = np.int(np.sum(count_rays_in_bundle))
        rays                = dict()
        rays['origin']      = np.zeros((total_rays,3), dtype=np.float64)
        rays['direction']   = np.zeros((total_rays,3), dtype=np.float64)
        rays['wavelength']  = np.zeros((total_rays), dtype=np.float64)
        rays['weight']      = np.zeros((total_rays), dtype=np.float64)
        rays['mask']        = np.ones((total_rays), dtype=np.bool)

        index = 0
        for ii, num_rays in enumerate(count_rays_in_bundle):
            rays['origin'][index:index+num_rays] = rays_list[ii]['origin']
            rays['direction'][index:index+num_rays] = rays_list[ii]['direction']
            rays['wavelength'][index:index+num_rays] = rays_list[ii]['wavelength']
            rays['weight'][index:index+num_rays] = rays_list[ii]['weight']
            rays['mask'][index:index+num_rays] = rays_list[ii]['mask']
            index += num_rays
        profiler.stop('Ray Bundle Collection')
            
        if len(rays['mask']) == 0:
            raise ValueError('No rays generated. Check plasma input parameters')

        self.log.debug('Bundles Generated:       {:0.4e}'.format(
            len(m[m])))
        self.log.debug('Rays per bundle, mean:   {:0.0f}'.format(
            np.mean(count_rays_in_bundle)))
        self.log.debug('Rays per bundle, median: {:0.0f}'.format(
            np.median(count_rays_in_bundle)))
        self.log.debug('Rays per bundle, max:    {:0d}'.format(
            np.max(count_rays_in_bundle)))
        self.log.debug('Rays per bundle, min:    {:0d}'.format(
            np.min(count_rays_in_bundle)))

        return rays

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
