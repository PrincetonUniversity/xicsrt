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
from xicsrt.xicsrt_objects import TraceObject
from xicsrt.sources._XicsrtSourceFocused import XicsrtSourceFocused

class XicsrtPlasmaGeneric(TraceObject):
    """
    A generic plasma object.

    Plasma object will generate a set of ray bundles where each ray
    bundle has the properties of the plamsa at one particular real-space point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bundle_filters = []

    def get_default_config(self):
        config = super().get_default_config()
                
        config['width']          = 0.0
        config['height']         = 0.0
        config['depth']          = 0.0
        
        config['spread']         = 2*np.pi
        config['target']         = None
        
        config['mass_number']    = 1.0
        config['wavelength']     = 1.0
        config['linewidth']      = 0.0
        config['intensity']      = 0.0
        config['temperature']    = 0.0
        config['velocity']       = 0.0
        config['use_poisson']    = False
        
        config['emissivity']      = 0.0
        config['max_rays']        = int(1e7)
        config['time_resolution'] = 1e-3
        config['bundle_count']    = int(1e5)
        config['bundle_volume']   = 1e-3
        config['bundle_type']     = 'voxel'
        return config
 
    def initialize(self):
        super().initialize()
        self.param['max_rays'] = int(self.param['max_rays'])
        self.param['bundle_type']  = str.lower(self.param['bundle_type'])
        self.param['bundle_count'] = int(self.param['bundle_count'])
        
        self.param['volume']       = self.config['width'] * self.config['height'] * self.config['depth']
        self.param['solid_angle']  = 4 * np.pi * np.sin(self.config['spread'] * np.pi / 360)**2
        
    def setup_bundles(self):
        
        if self.param['bundle_type'] == 'point':
            self.param['voxel_size'] = 0.0
        elif self.param['bundle_type'] == 'voxel':
            self.param['voxel_size'] = self.param['bundle_volume'] ** (1/3)

        # These values should be overwritten in a derived class.
        bundle_input = {}
        bundle_input['origin']       = np.zeros([self.param['bundle_count'], 3], dtype = np.float64)
        bundle_input['temperature']  = np.ones([self.param['bundle_count']], dtype = np.float64)
        bundle_input['emissivity']   = np.ones([self.param['bundle_count']], dtype = np.int)
        bundle_input['velocity']     = np.zeros([self.param['bundle_count'], 3], dtype = np.float64)
        bundle_input['mask']         = np.ones([self.param['bundle_count']], dtype = np.bool)
        
        return bundle_input

    def bundle_generate(self, bundle_input):
        return bundle_input

    def bundle_filter(self, bundle_input):
        for filter in self.bundle_filters:
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
        predicted_rays = int(
            np.sum(bundle_input['emissivity'][m])
            * self.param['time_resolution']
            * self.param['bundle_volume']
            * self.param['solid_angle'] / (4 * np.pi)
            * self.param['volume'] / (self.param['bundle_count'] * self.param['bundle_volume']))

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
            # In order to allow this while maintaning overall photon statistics
            # from the plasma, we normalize the intensity so that each bundle
            # represents a volume of plasma_volume/bundle_count.
            #
            # In doing so bundle_volume cancels out, but I am leaving the
            # calculation separate for clarity.
            intensity *= self.param['volume'] / (self.param['bundle_count'] * self.param['bundle_volume'])

            source_config['intensity'] = intensity
            
            # constants
            source_config['width']       = self.param['voxel_size']
            source_config['height']      = self.param['voxel_size']
            source_config['depth']       = self.param['voxel_size']
            source_config['zaxis']       = self.param['zaxis']
            source_config['xaxis']       = self.param['xaxis']
            source_config['target']      = self.param['target']
            source_config['mass_number'] = self.param['mass_number']
            source_config['wavelength']  = self.param['wavelength']
            source_config['linewidth']   = self.param['linewidth']
            source_config['spread']      = self.param['spread']
            source_config['use_poisson'] = self.param['use_poisson']
            
            #create ray bundle sources and generate bundled rays
            source       = XicsrtSourceFocused(source_config)
            bundled_rays = source.generate_rays()

            rays_list.append(bundled_rays)
            count_rays_in_bundle.append(len(bundled_rays['mask']))

            profiler.stop("Ray Bundle Generation")

            
        profiler.start('Ray Bundle Collection')
        # append bundled rays together to form a single ray dictionary.    
        # create the final ray dictionary
        total_rays = np.sum(count_rays_in_bundle)
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

        logging.info('Rays per bundle, mean:   {:0.0f}'.format(
            np.mean(count_rays_in_bundle)))
        logging.info('Rays per bundle, median: {:0.0f}'.format(
            np.median(count_rays_in_bundle)))
        logging.info('Rays per bundle, max:    {:0d}'.format(
            np.max(count_rays_in_bundle)))
        logging.info('Rays per bundle, min:    {:0d}'.format(
            np.min(count_rays_in_bundle)))

        return rays

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.bundle_generate(bundle_input)
        
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        
        return rays
