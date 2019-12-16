# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:50:18 2019

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
    def __init__(self, plasma_input):
        super(GenericPlasma, self).__init__(
            plasma_input['position']
            ,plasma_input['normal']
            ,plasma_input['orientation'])
        
        self.plasma_input   = plasma_input
        #spacial information
        self.position       = plasma_input['position']
        self.normal         = plasma_input['normal']
        self.xorientation   = plasma_input['orientation']
        self.yorientation   = np.cross(self.normal, self.xorientation )
        self.yorientation  /= np.linalg.norm(self.yorientation)
        self.target         = plasma_input['target']
        self.width          = plasma_input['width']
        self.height         = plasma_input['height']
        self.depth          = plasma_input['depth']
        self.volume         = self.width * self.height * self.depth
        #bundle information
        self.bundle_count   = plasma_input['bundle_count']
        self.bundle_type    = str.lower(plasma_input['bundle_type'])
        self.max_rays       = plasma_input['max_rays']
        self.solid_angle    = 4 * np.pi * np.sin(plasma_input['spread'] * np.pi / 360) ** 2
        self.voxel_size     = plasma_input['space_resolution']
        self.chronon_size   = plasma_input['time_resolution']
        self.bundle_volume  = self.voxel_size ** 3
        #profile information
        self.use_profiles   = plasma_input['use_profiles']
        self.temp_data      = plasma_input['temperature_data']
        self.emis_data      = plasma_input['emissivity_data']
        self.temperature    = plasma_input['temperature']
        self.emissivity     = plasma_input['emissivity']
        self.mass_number    = plasma_input['mass']
        self.wavelength     = plasma_input['wavelength']
        self.linewidth      = plasma_input['linewidth']
        
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
                         * self.solid_angle / (4 * np.pi))

            # Scale the number of photons based on the number of bundles.
            #
            # bundle_volume cancels out here, each bundle represents an area of
            # volume/bundle_count.  I am leaving the calculation as is for now
            # for clarity in case a different approach is needed in the future.
            
            volume_factor = self.volume / (self.bundle_count * self.bundle_volume)
            intensity *= volume_factor
            """
            if intensity < 1:
                logging.warning('Bundle intensity is less than one. ')
                continue
            """
            source_input['intensity'] = int(intensity)
            
            #constants
            if self.bundle_type == 'voxel':
                source_input['width']   = self.voxel_size
                source_input['height']  = self.voxel_size
                source_input['depth']   = self.voxel_size
            if self.bundle_type == 'point':
                source_input['width']   = 0
                source_input['height']  = 0
                source_input['depth']   = 0

            source_input['normal']      = self.normal
            source_input['orientation'] = self.xorientation
            source_input['target']      = self.target
            source_input['mass']        = self.mass_number
            source_input['wavelength']  = self.wavelength
            source_input['linewidth']   = self.linewidth
            source_input['spread']      = self.plasma_input['spread']
            
            #create ray bundle sources and generate bundled rays
            source       = FocusedExtendedSource(source_input)
            bundled_rays = source.generate_rays()
            
            #append bundled rays together to form a single ray dictionary
            if len(rays['mask']) >= self.max_rays:
                print('Ray-Bundle Generation Halted: Too Many Rays')
                break
            else:
                rays['origin']      = np.append(rays['origin'],      bundled_rays['origin'], axis = 0)
                rays['direction']   = np.append(rays['direction'],   bundled_rays['direction'], axis = 0)
                rays['wavelength']  = np.append(rays['wavelength'],  bundled_rays['wavelength'])
                rays['weight']      = np.append(rays['weight'],      bundled_rays['weight'])
                rays['mask']        = np.append(rays['mask'],        bundled_rays['mask'])
            profiler.stop("Ray Bundle Generation")
        print(' Bundles Generated: {:6.4e}'.format(ii + 1))        
        return rays
        
class CubicPlasma(GenericPlasma):
    def __init__(self, plasma_input):
        super().__init__(plasma_input)
        
        self.position       = plasma_input['position']
        self.normal         = plasma_input['normal']
        self.xorientation   = plasma_input['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        
        self.width          = plasma_input['width']
        self.height         = plasma_input['height']
        self.depth          = plasma_input['depth']
        self.volume         = self.width * self.height * self.depth
        self.bundle_count   = plasma_input['bundle_count']
        self.bundle_volume  = plasma_input['bundle_volume']
        self.bundle_type    = plasma_input['bundle_type']
                
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
        bundle_input['temp'][:]         = self.temperature
        
        #evaluate emissivity at each point
        #plasma cube has a constant emissivity througout.
        bundle_input['emissivity'][:]   = self.emissivity
            
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
    def __init__(self, plasma_input):
        super().__init__(plasma_input)
        
        self.position       = plasma_input['position']
        self.normal         = plasma_input['normal']
        self.xorientation   = plasma_input['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        
        self.width          = plasma_input['width']
        self.height         = plasma_input['height']
        self.depth          = plasma_input['depth']
        self.volume         = self.width * self.height * self.depth
        self.bundle_count   = plasma_input['bundle_count']
                
    def cylindrical_bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2,  self.width/2,  self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2,  self.depth/2,  self.bundle_count)        
        
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', x_offset, self.xorientation)
                  + np.einsum('i,j', y_offset, self.yorientation)
                  + np.einsum('i,j', z_offset, self.normal))
        
        #convert from cartesian coordinates to cylindrical coordinates [radius, azimuth, height]
        #cylinder is ordiented along the Y axis
        """
        plasma normal           = absolute X
        plasma x orientation    = absolute Z
        plasma y orientation    = absolute Y
        """
        radius, azimuth, height = cart2cyl(z_offset, x_offset, y_offset)
        
        #evaluate temperature at each point
        #plasma cylinder temperature falls off as a function of radius
        bundle_input['temp']         = self.temperature / radius
        
        #evaluate emissivity at each point
        #plasma cylinder emissivity falls off as a function of radius
        bundle_input['emissivity']   = self.emissivity / radius
        
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
    def __init__(self, plasma_input):
        super(ToroidalPlasma, self).__init__(plasma_input)
        self.position       = plasma_input['position']
        
        self.width          = plasma_input['width']
        self.height         = plasma_input['height']
        self.depth          = plasma_input['depth']
        self.volume         = self.width * self.height * self.depth
        self.major_radius   = plasma_input['major_radius']
        self.minor_radius   = plasma_input['minor_radius']
        self.bundle_count   = plasma_input['bundle_count']
        
    def toroidal_bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2 , self.width/2 , self.bundle_count) + self.major_radius
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        z_offset = np.random.uniform(-1 * self.depth/2 , self.depth/2 , self.bundle_count)        
        
        #unlike the other plasmas, the toroidal plasma has fixed orientation to
        #prevent confusion
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', x_offset, np.array([1, 0, 0]))
                  + np.einsum('i,j', y_offset, np.array([0, 1, 0]))
                  + np.einsum('i,j', z_offset, np.array([0, 0, 1])))
        
        #convert from cartesian coordinates to toroidal coordinates [rad, pol, tor]
        #torus is oriented along the Z axis
        rad, pol, tor = cart2toro(x_offset, y_offset, z_offset, self.major_radius)
        
        #evaluate temperature and emissivity at each point
        if self.use_profiles is False:
            #step function profile
            step_test    = np.zeros(self.bundle_count, dtype = np.bool)
            step_test[:] = (rad <= self.minor_radius)
            bundle_input['temp'][step_test]         = self.temperature
            bundle_input['emissivity'][step_test]   = self.emissivity
            
        if self.use_profiles is True:
            #read and interpolate profile from data file
            temp_data  = np.loadtxt(self.temp_data, dtype = np.float64)
            emis_data  = np.loadtxt(self.emis_data, dtype = np.float64)
            nrad       = rad / self.minor_radius
            
            bundle_input['temp']        = np.interp(nrad, temp_data[:,0], temp_data[:,1],
                                                    left = 1.0, right = 1.0)
            bundle_input['emissivity']  = np.interp(nrad, emis_data[:,0], emis_data[:,1],
                                                    left = 1.0, right = 1.0)
        
        return bundle_input

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.toroidal_bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays