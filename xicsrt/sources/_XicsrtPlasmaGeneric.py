# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir A. Pablant <nablant@pppl.gov>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
This script holds all of the plasma classes. These are separate from ray
sources; rather than emitting rays, plasmas create ray sources. They are
effectively advanced substitutes for regular ray sources.
"""
import logging

import numpy as np   
from collections import OrderedDict

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._GeometryObject import GeometryObject
from xicsrt.sources._XicsrtSourceFocused import XicsrtSourceFocused

@dochelper
class XicsrtPlasmaGeneric(GeometryObject):
    """
    A generic plasma object.

    Plasma object will generate a set of ray bundles where each ray bundle
    has the properties of the plamsa at one particular real-space point.

    Each bundle is modeled by a SourceFocused object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_objects = []

    def default_config(self):
        """
        width
          The width of this element. Aligned with the x-axis.

        height
          The height of this element. Aligned with the y-axis.

        depth:
          The depth of this element. Aligned with the z-axis.

        spread: float (pi) [radians]
          The angular spread for the emission cone. The spread defines the
          half-angle of the cone. A value of `pi` results in fully isotropic
          emission (which is not generally useful in raytracing applications).

        .. warning::

          Config documentation is incomplete.
        """
        config = super().default_config()
                
        config['xsize']          = 0.0
        config['ysize']         = 0.0
        config['zsize']          = 0.0
        
        config['spread']         = 2*np.pi
        config['target']         = None
        
        config['mass_number']     = 1.0
        config['wavelength']      = 1.0
        config['linewidth']       = 0.0
        config['intensity']       = 0.0
        config['temperature']     = 0.0
        config['velocity']        = 0.0
        config['use_poisson']     = False
        config['wavelength_dist'] = 'voigt'
        
        config['emissivity']      = 0.0
        config['max_rays']        = int(1e7)
        config['time_resolution'] = 1e-3
        config['bundle_count']    = int(1e5)
        config['bundle_volume']   = 1e-3
        config['bundle_type']     = 'voxel'
        
        config['filter_list']     = []
        return config
 
    def initialize(self):
        super().initialize()
        self.param['max_rays']     = int(self.param['max_rays'])
        self.param['bundle_type']  = str.lower(self.param['bundle_type'])
        self.param['bundle_count'] = int(self.param['bundle_count'])
        self.param['volume']       = self.config['xsize'] * self.config['ysize'] * self.config['zsize']
        self.param['solid_angle']  = 4 * np.pi * np.sin(self.config['spread']/2)**2
        
        
    def setup_bundles(self):
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
        
        # randomly spread the bundles around the plasma box
        offset = np.zeros((self.param['bundle_count'], 3))
        offset[:,0] = np.random.uniform(-1 * self.param['xsize'] /2, self.param['xsize'] /2, self.param['bundle_count'])
        offset[:,1] = np.random.uniform(-1 * self.param['ysize']/2, self.param['ysize']/2, self.param['bundle_count'])
        offset[:,2] = np.random.uniform(-1 * self.param['zsize'] /2, self.param['zsize'] /2, self.param['bundle_count'])

        bundle_input['origin'][:] = self.point_to_external(offset)
        return bundle_input

    def get_emissivity(self, rho):
        return self.param['emissivity']

    def get_temperature(self, rho):
        return self.param['temperature']

    def get_velocity(self, rho):
        return self.param['velocity']

    def bundle_generate(self, bundle_input):
        return bundle_input

    def bundle_filter(self, bundle_input):
        for filter in self.filter_objects:
            bundle_input = filter.filter(bundle_input)
        return bundle_input
    
    def create_sources(self, bundle_input):
        """
        Generate rays from a list of bundles.

        bundle_input
          a list containing dictionaries containing the locations,
          emissivities, temperatures and velocitities and of all ray bundles to be
          emitted.
        """
        
        #create ray dictionary
        rays_list = []
        count_rays_in_bundle = []

        m = bundle_input['mask']

        # Check if the number of rays generated will exceed max ray limits.
        # This is only approximate since poisson statistics may be in use.
        predicted_rays = int(np.sum(
            bundle_input['emissivity'][m]
            * self.param['time_resolution']
            * self.param['bundle_volume']
            * self.param['solid_angle'] / (4 * np.pi)
            * self.param['volume'] / (self.param['bundle_count'] * self.param['bundle_volume'])))

        if predicted_rays >= self.param['max_rays']:
            raise ValueError('Current settings will produce too many rays. Please reduce integration time.')
        
        #bundle generation loop
        for ii in range(self.param['bundle_count']):
            if not bundle_input['mask'][ii]:
                continue
            profiler.start("Ray Bundle Generation")
            source_config = OrderedDict()
            
            #spacially-dependent parameters
            source_config['origin']      = bundle_input['origin'][ii]
            source_config['temperature'] = bundle_input['temperature'][ii]
            source_config['velocity']    = bundle_input['velocity'][ii]

            # Calculate the total number of photons to launch from this bundle
            # volume. Since the source can use poisson statistics, this should
            # be of floating point type.
            intensity = (bundle_input['emissivity'][ii]
                         * self.param['time_resolution']
                         * self.param['bundle_volume']
                         * self.param['solid_angle'] / (4 * np.pi))
            
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
            source_config['xsize']           = self.param['voxel_size']
            source_config['ysize']          = self.param['voxel_size']
            source_config['zsize']           = self.param['voxel_size']
            source_config['zaxis']           = self.param['zaxis']
            source_config['xaxis']           = self.param['xaxis']
            source_config['target']          = self.param['target']
            source_config['mass_number']     = self.param['mass_number']
            source_config['wavelength']      = self.param['wavelength']
            source_config['linewidth']       = self.param['linewidth']
            source_config['spread']          = self.param['spread']
            source_config['use_poisson']     = self.param['use_poisson']
            source_config['wavelength_dist'] = self.param['wavelength_dist']
                
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

        logging.debug('Bundles Generated:       {:0.4e}'.format(
            len(m[m])))
        logging.debug('Rays per bundle, mean:   {:0.0f}'.format(
            np.mean(count_rays_in_bundle)))
        logging.debug('Rays per bundle, median: {:0.0f}'.format(
            np.median(count_rays_in_bundle)))
        logging.debug('Rays per bundle, max:    {:0d}'.format(
            np.max(count_rays_in_bundle)))
        logging.debug('Rays per bundle, min:    {:0d}'.format(
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
