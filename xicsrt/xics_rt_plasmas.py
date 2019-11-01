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

import numpy as np   
from collections import OrderedDict

from xicsrt.util import profiler
from xicsrt.xics_rt_math    import cart2cyl#, cart2pol, cart2flux
from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_objects import TraceObject

class GenericPlasma(TraceObject):
    def __init__(self, plasma_input):
        super().__init__(
            plasma_input['position']
            ,plasma_input['normal']
            ,plasma_input['orientation'])
        
        self.plasma_input   = plasma_input
        self.position       = plasma_input['position']
        self.normal         = plasma_input['normal']
        self.xorientation   = plasma_input['orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.target         = plasma_input['target']
        # Voxels are 3D pixels, Chronons are time pixels
        self.width          = plasma_input['width']
        self.height         = plasma_input['height']
        self.depth          = plasma_input['depth']
        self.volume         = self.width * self.height * self.depth
        self.solid_angle    = plasma_input['spread'] * (np.pi ** 2) / 180 
        self.voxel_size     = plasma_input['space_resolution']
        self.chronon_size   = plasma_input['time_resolution']
        self.bundle_count   = plasma_input['bundle_count']
        
        self.mass_number    = plasma_input['mass']   
        self.temp           = plasma_input['temp']
        self.wavelength     = plasma_input['wavelength']
        self.linewidth      = plasma_input['linewidth']
        
        self.partitioning   = plasma_input['volume_partitioning']
        
    def setup_bundles(self):
        ## Decide how many bundles there will be
        if self.partitioning is True:
            self.bundle_count * int(self.volume / self.voxel_size)
            
        bundle_input = dict()
        bundle_input['position']     = np.zeros([self.bundle_count, 3], dtype = np.float64)
        bundle_input['temp']         = np.ones( [self.bundle_count], dtype = np.float64)
        bundle_input['emissivity']   = np.ones( [self.bundle_count], dtype = np.int)
        
        return bundle_input
        
    def create_sources(self, bundle_input):
        ## Bundle_input is a list containing dictionaries containing the locations,
        ## temperatures, and emissivities of all ray bundles to be emitted
        
        #create ray dictionary
        rays                = dict()
        rays['origin']      = np.zeros([1,3], dtype = np.float64)
        rays['direction']   = np.ones( [1,3], dtype = np.float64)
        rays['wavelength']  = np.ones( [1], dtype=np.float64)
        rays['weight']      = np.ones( [1], dtype=np.float64)
        rays['mask']        = np.ones( [1], dtype=np.bool)
        
        #bundle generation loop
        for ii in range(self.bundle_count):
            profiler.start("Ray Bundle Generation")
            source_input = OrderedDict()
            #spacially-dependent parameters
            source_input['position']    = bundle_input['position'][ii]
            source_input['temp']        = bundle_input['temp'][ii]
            source_input['intensity']   = int(bundle_input['emissivity'][ii]
                * self.chronon_size * self.voxel_size * self.solid_angle)
            
            #constants
            source_input['width'], source_input['height'], source_input['depth'] = [np.power(self.voxel_size, 1/3)] * 3
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
                
    def cubic_bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        w_offset = np.random.uniform(-1 * self.width/2, self.width/2, self.bundle_count)
        h_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        d_offset = np.random.uniform(-1 * self.depth/2, self.depth/2, self.bundle_count)        
                
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', w_offset, self.xorientation)
                  + np.einsum('i,j', h_offset, self.yorientation)
                  + np.einsum('i,j', d_offset, self.normal))
        
        #evaluate temperature at each point
        #plasma cube has consistent temperature throughout
        bundle_input['temp'][:]         = self.temp
        
        #evaluate emissivity at each point
        #plasma cube has uniformly random emissivity throughout
        bundle_input['emissivity'][:]   = np.random.uniform(0, 2e13, self.bundle_count)
            
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
        w_offset = np.random.uniform(-1 * self.width/2, self.width/2, self.bundle_count)
        h_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count)
        d_offset = np.random.uniform(-1 * self.depth/2, self.depth/2, self.bundle_count)        
                
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', w_offset, self.xorientation)
                  + np.einsum('i,j', h_offset, self.yorientation)
                  + np.einsum('i,j', d_offset, self.normal))
        
        #convert from cartesian coordinates to cylindrical coordinates [radius, azimuth, height]
        #NOTE: cylinder is rotated on its side such that height = w_offset || xorientation
        radius, azimuth, height = cart2cyl(d_offset, h_offset, w_offset)
        
        #evaluate temperature at each point
        #plasma cylinder temperature falls off as a function of radius
        bundle_input['temp']         = self.temp / radius
        
        #evaluate emissivity at each point
        #plasma cylinder emissivity falls off as a function of radius
        bundle_input['emissivity']   = 1e12 / radius
        
        return bundle_input

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.cylindrical_bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays