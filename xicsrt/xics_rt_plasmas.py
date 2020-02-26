# -*- coding: utf-8 -*-
"""
Authors
-------
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
from xicsrt.xics_rt_math    import cart2cyl, cart2toro
from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_objects import TraceObject

class GenericPlasma(TraceObject):
    """
    A generic plasma object.

    Plasma object will generate a set of ray bundles where each ray
    bundle has the properties of the plamsa at one particular real-space point.
    """
    def __init__(self, config):
        super().__init__(
            config['position']
            ,config['normal']
            ,config['orientation'])

        self.config   = config
        self.max_rays       = config['max_rays']
        self.position       = config['position']
        self.normal         = config['normal']
        self.xorientation   = config['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.target         = config['target']
        self.width          = config['width']
        self.height         = config['height']
        self.depth          = config['depth']
        self.volume         = self.width * self.height * self.depth

        # Bundle parameters.
        # Voxels are 3D pixels, Chronons are time pixels
        self.spread         = config['spread']
        self.solid_angle    = 4 * np.pi * np.sin(config['spread'] * np.pi / 360)**2
        self.chronon_size   = config['time_resolution']
        self.bundle_count   = config['bundle_count']
        self.bundle_volume  = config['bundle_volume']
        self.bundle_type    = config['bundle_type']

        # Plasma parameters.
        self.mass_number    = config['mass']
        self.wavelength     = config['wavelength']
        self.linewidth      = config['linewidth']
        self.emissivity     = config['emissivity']
        self.temperature    = config['temperature']
        self.velocity       = config['velocity']
        self.use_poisson    = config['use_poisson']

        self.check_inputs()

    def check_inputs(self):
        self.bundle_type = str.upper(self.bundle_type)
        
    def setup_bundles(self):
        
        if self.bundle_type == 'POINT':
            self.voxel_size = 0.0
        elif self.bundle_type == 'VOXEL':
            self.voxel_size = self.bundle_volume ** (1/3)

        # These values should be overwritten in a derived class.
        bundle_input = {}
        bundle_input['position']     = np.zeros([self.bundle_count, 3], dtype = np.float64)
        bundle_input['temperature']  = np.ones([self.bundle_count], dtype = np.float64)
        bundle_input['emissivity']   = np.ones([self.bundle_count], dtype = np.int)
        bundle_input['velocity']     = np.zeros([self.bundle_count, 3], dtype = np.float64)
        bundle_input['mask']         = np.ones([self.bundle_count], dtype = np.bool)
        
        return bundle_input

    def bundle_generate(self, bundle_input):
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
        rays                = dict()
        rays['origin']      = np.zeros([0,3], dtype = np.float64)
        rays['direction']   = np.ones( [0,3], dtype = np.float64)
        rays['wavelength']  = np.ones( [0], dtype=np.float64)
        rays['weight']      = np.ones( [0], dtype=np.float64)
        rays['mask']        = np.ones( [0], dtype=np.bool)

        count_rays_in_bundle = []

        m = bundle_input['mask']
        
        # Check if the number of rays generated will exceed max ray limits.
        # This is only approximate since poisson statistics may be in use.
        predicted_rays = int(
            np.sum(bundle_input['emissivity'][m])
            * self.chronon_size
            * self.bundle_volume
            * self.solid_angle / (4 * np.pi)
            * self.volume / (self.bundle_count * self.bundle_volume))

        if predicted_rays >= self.max_rays:
            raise ValueError('Current settings will produce too many rays. Please reduce integration time.')
        
        #bundle generation loop
        for ii in range(self.bundle_count):
            
            if not bundle_input['mask'][ii]:
                continue

            profiler.start("Ray Bundle Generation")
            source_input = OrderedDict()
            
            #spacially-dependent parameters
            source_input['position']    = bundle_input['position'][ii]
            source_input['temperature'] = bundle_input['temperature'][ii]
            source_input['velocity'] = bundle_input['velocity'][ii]

            # Calculate the total number of photons to launch from this bundle
            # volume. Since the source can use poisson statistics, this should
            # be of floating point type.
            intensity = (bundle_input['emissivity'][ii]
                         * self.chronon_size
                         * self.bundle_volume
                         * self.solid_angle / (4 * np.pi))

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
            intensity *= self.volume / (self.bundle_count * self.bundle_volume)

            source_input['intensity'] = intensity

            # constants
            source_input['width']       = self.voxel_size
            source_input['height']      = self.voxel_size
            source_input['depth']       = self.voxel_size
            source_input['normal']      = self.normal
            source_input['orientation'] = self.xorientation
            source_input['target']      = self.target
            source_input['mass']        = self.mass_number
            source_input['wavelength']  = self.wavelength
            source_input['linewidth']   = self.linewidth
            source_input['spread']      = self.config['spread']
            source_input['use_poisson'] = self.use_poisson
            
            #create ray bundle sources and generate bundled rays
            source       = FocusedExtendedSource(source_input)
            bundled_rays = source.generate_rays()

            count_rays_in_bundle.append(len(bundled_rays['mask']))

            # append bundled rays together to form a single ray dictionary.
            #
            # It would (probably) be faster to first put these into a normal
            # pythorn dictionary, then do the collection at the end. This wolud
            # take more memory though.
            profiler.start('Ray Bundle Collection')
            rays['origin'] = np.append(rays['origin'], bundled_rays['origin'], axis=0)
            rays['direction'] = np.append(rays['direction'], bundled_rays['direction'], axis=0)
            rays['wavelength'] = np.append(rays['wavelength'], bundled_rays['wavelength'])
            rays['weight'] = np.append(rays['weight'], bundled_rays['weight'])
            rays['mask'] = np.append(rays['mask'], bundled_rays['mask'])
            profiler.start('Ray Bundle Collection')

            profiler.stop("Ray Bundle Generation")
            
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
    
class CubicPlasma(GenericPlasma):
    """
    A cubic plasma.
    """
                
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2,  self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2,  self.depth/2,  self.bundle_count)        
                
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', x_offset, self.xorientation)
                  + np.einsum('i,j', y_offset, self.yorientation)
                  + np.einsum('i,j', z_offset, self.normal))
        
        #evaluate temperature at each point
        #plasma cube has consistent temperature throughout
        bundle_input['temperature'][:] = self.temperature
        
        #evaluate emissivity at each point
        #plasma cube has a constant emissivity througout.
        bundle_input['emissivity'][:] = self.emissivity
            
        return bundle_input


class CylindricalPlasma(GenericPlasma):
    """
    A cylindrical plasma ordiented along the Y axis.

    This class is meant only to be used as an exmple for generating 
    more complecated classes for specific plasmas.

    plasma normal           = absolute X
    plasma x orientation    = absolute Z
    plasma y orientation    = absolute Y
    """
                
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2,  self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2,  self.depth/2,  self.bundle_count)        
        
        bundle_input['position'][:] = (
            self.position
            + np.einsum('i,j', x_offset, self.xorientation)
            + np.einsum('i,j', y_offset, self.yorientation)
            + np.einsum('i,j', z_offset, self.normal))
        
        #convert from cartesian coordinates to cylindrical coordinates [radius, azimuth, height]
        radius, azimuth, height = cart2cyl(z_offset, x_offset, y_offset)

        
        # Let plasma temperature and emissivity fall off as a function of
        # radius.
        bundle_input['emissivity'][step_test]  = self.emissivity / radius
        bundle_input['temperature'][step_test] = self.temperature / radius
        bundle_input['velocity'][step_test]  = self.velocity
        
        return bundle_input


class ToroidalPlasma(GenericPlasma):
    """
    A cylindrical plasma ordiented along the Y axis.

    This class is meant only to be used as an exmple for generating 
    more complecated classes for specific plasmas.
    """
    def __init__(self, config):
        super().__init__(config)

        self.major_radius   = config['major_radius']
        self.minor_radius   = config['minor_radius']
        
    def bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2 , self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2 , self.depth/2,  self.bundle_count)        
        
        #unlike the other plasmas, the toroidal plasma has fixed orientation to
        #prevent confusion
        bundle_input['position'][:] = (
            self.position
            + np.einsum('i,j', x_offset, np.array([1, 0, 0]))
            + np.einsum('i,j', y_offset, np.array([0, 1, 0]))
            + np.einsum('i,j', z_offset, np.array([0, 0, 1])))
        
        #convert from cartesian coordinates to toroidal coordinates [sigma, tau, phi]
        #torus is oriented along the Z axis
        rad, pol, tor = cart2toro(x_offset, y_offset, z_offset, self.major_radius)
        
        step_test = (rad <= self.minor_radius)

        # Let plasma temperature and emissivity fall off as a function of
        # radius.
        bundle_input['emissivity'][step_test]  = self.emissivity / radius
        bundle_input['temperature'][step_test] = self.temperature / radius
        bundle_input['velocity'][step_test]  = self.velocity
        
        return bundle_input
