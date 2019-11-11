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
        self.position       = config['position']
        self.normal         = config['normal']
        self.xorientation   = config['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.target         = config['target']
        # Voxels are 3D pixels, Chronons are time pixels
        self.width          = config['width']
        self.height         = config['height']
        self.depth          = config['depth']
        self.volume         = self.width * self.height * self.depth
        self.solid_angle    = config['spread'] * (np.pi ** 2) / 180
        self.chronon_size   = config['time_resolution']
        self.bundle_count   = config['bundle_count']
        self.bundle_volume  = config['bundle_volume']
        self.bundle_type    = config['bundle_type']
        
        self.mass_number    = config['mass']
        self.temp           = config['temp']
        self.wavelength     = config['wavelength']
        self.linewidth      = config['linewidth']
        
    def setup_bundles(self):
            
        bundle_input = dict()
        bundle_input['position']     = np.zeros([self.bundle_count, 3], dtype = np.float64)
        bundle_input['temp']         = np.ones([self.bundle_count], dtype = np.float64)
        bundle_input['emissivity']   = np.ones([self.bundle_count], dtype = np.int)
        
        return bundle_input
        
    def create_sources(self, bundle_input):
        ## Bundle_input is a list containing dictionaries containing the locations,
        ## temperatures, and emissivities of all ray bundles to be emitted
        
        #create ray dictionary
        rays                = dict()
        rays['origin']      = np.zeros([0,3], dtype = np.float64)
        rays['direction']   = np.ones( [0,3], dtype = np.float64)
        rays['wavelength']  = np.ones( [0], dtype=np.float64)
        rays['weight']      = np.ones( [0], dtype=np.float64)
        rays['mask']        = np.ones( [0], dtype=np.bool)

        flag_warning_issued_intensity = False

        #bundle generation loop
        for ii in range(self.bundle_count):
            profiler.start("Ray Bundle Generation")
            source_input = OrderedDict()
            #spacially-dependent parameters
            source_input['position']    = bundle_input['position'][ii]
            source_input['temp']        = bundle_input['temp'][ii]

            # Calculate the total number of photons to launch from this bundle volume.
            intensity = (bundle_input['emissivity'][ii]
                         * self.chronon_size
                         * self.bundle_volume
                         * self.solid_angle)

            # Scale the number of photons based on the number of bundles.
            #
            # bundle_volume cancels out here, each bundle represents an area of
            # volume/bundle_count.  I am leaving the calculation as is for now
            # for clarity in case a different approach is needed in the future.
            volume_factor = self.volume/(self.bundle_count*self.bundle_volume)
            intensity *= volume_factor

            # Ignore bundles with no intensity (quietly except for one warning).
            if intensity < 1:
                if not flag_warning_issued_intensity:
                    flag_warning_issued_intensity = True
                    logging.warning('Bundle intensity of less than one encountered.')
                continue

            source_input['intensity'] = int(intensity)
            
            #constants
            if self.bundle_type == 'voxel':
                voxel_size = np.power(self.bundle_volume, 1/3)
                source_input['width'] = voxel_size
                source_input['height'] = voxel_size
                source_input['depth'] = voxel_size
            if self.bundle_type == 'point':
                source_input['width'] = 0
                source_input['height'] = 0
                source_input['depth'] = 0

            source_input['normal']      = self.normal
            source_input['orientation'] = self.xorientation
            source_input['target']      = self.target
            source_input['mass']        = self.mass_number
            source_input['wavelength']  = self.wavelength
            source_input['linewidth']   = self.linewidth
            source_input['spread']      = self.config['spread']
            
            #create ray bundle sources and generate bundled rays
            source       = FocusedExtendedSource(source_input)
            bundled_rays = source.generate_rays()
            
            #append bundled rays together to form a single ray dictionary
            if len(rays['mask']) >= 1e7:
                print('Ray-Bundle Generation Halted: Too Many Rays')
                break
            else:
                rays['origin']      = np.append(rays['origin'],      bundled_rays['origin'], axis = 0)
                rays['direction']   = np.append(rays['direction'],   bundled_rays['direction'], axis = 0)
                rays['wavelength']  = np.append(rays['wavelength'],  bundled_rays['wavelength'])
                rays['weight']      = np.append(rays['weight'],      bundled_rays['weight'])
                rays['mask']        = np.append(rays['mask'],        bundled_rays['mask'])
            profiler.stop("Ray Bundle Generation")

        if len(rays['mask']) == 0:
            raise Exception('No rays generated. Check plasma input parameters')

        return rays
        
class CubicPlasma(GenericPlasma):
    """
    A cubic plasma.
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.position       = config['position']
        self.normal         = config['normal']
        self.xorientation   = config['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        
        self.width          = config['width']
        self.height         = config['height']
        self.depth          = config['depth']
        self.volume         = self.width * self.height * self.depth
        self.bundle_count   = config['bundle_count']
        self.bundle_volume  = config['bundle_volume']
        self.bundle_type    = config['bundle_type']
                
    def cubic_bundle_generate(self, bundle_input):
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
        bundle_input['temp'][:]         = self.temp
        
        #evaluate emissivity at each point
        #plasma cube has a constant emissivity througout.
        bundle_input['emissivity'][:]   = 1e12
            
        return bundle_input
    
    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.cubic_bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays


class CylindricalPlasma(GenericPlasma):
    """
    A cylindrical plasma ordiented along the Y axis

    plasma normal           = absolute X
    plasma x orientation    = absolute Z
    plasma y orientation    = absolute Y
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.position       = config['position']
        self.normal         = config['normal']
        self.xorientation   = config['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        
        self.width          = config['width']
        self.height         = config['height']
        self.depth          = config['depth']
        self.volume         = self.width * self.height * self.depth
        self.bundle_count   = config['bundle_count']
                
    def cylindrical_bundle_generate(self, bundle_input):
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
        
        #evaluate temperature at each point
        #plasma cylinder temperature falls off as a function of radius
        bundle_input['temp']         = self.temp / radius
        
        #evaluate emissivity at each point
        #plasma cylinder emissivity falls off as a function of radius
        bundle_input['emissivity']   = 1e10 / radius
        
        return bundle_input

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.cylindrical_bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays


class ToroidalPlasma(GenericPlasma):
    def __init__(self, config):
        super().__init__(config)

        self.major_radius   = config['major_radius']
        self.minor_radius   = config['minor_radius']
        
    def toroidal_bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2 , self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2 , self.depth/2,  self.bundle_count)        
        
        #unlike the other plasmas, the toroidal plasma has fixed orientation to
        #prevent confusion
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', x_offset, np.array([1, 0, 0]))
                  + np.einsum('i,j', y_offset, np.array([0, 1, 0]))
                  + np.einsum('i,j', z_offset, np.array([0, 0, 1])))
        
        #convert from cartesian coordinates to toroidal coordinates [sigma, tau, phi]
        #torus is oriented along the Z axis
        rad, pol, tor = cart2toro(x_offset, y_offset, z_offset, self.major_radius)
        
        step_test    = np.zeros(self.bundle_count, dtype = np.bool)
        step_test[:] = (rad <= self.minor_radius)

        #evaluate temperature at each point
        #plasma torus temperature falls off as a function of radius
        bundle_input['temp'][step_test] = self.temp
        
        #evaluate emissivity at each point
        #plasma torus emissivity falls off as a function of radius
        bundle_input['emissivity'][step_test]   = 1e11
  
        return bundle_input

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.toroidal_bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays
