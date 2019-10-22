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
from xicsrt.xics_rt_sources import FocusedExtendedSource
from xicsrt.xics_rt_objects import TraceObject

class GenericPlasma(TraceObject):
    def __init__(self, plasma_input, general_input):
        super().__init__(
            plasma_input['position']
            ,plasma_input['normal']
            ,plasma_input['orientation'])
        
        self.plasma_input   = plasma_input
        self.general_input  = general_input
        
        self.position       = plasma_input['position']
        self.normal         = plasma_input['normal']
        self.orientation    = plasma_input['orientation']
        self.dimensions     = plasma_input['dimensions']
        self.target         = plasma_input['target']
        # Voxels are 3D pixels, Chronons are time pixels
        self.volume         = plasma_input['volume']
        self.solid_angle    = plasma_input['spread'] * (np.pi ** 2) / 180 
        self.voxel_size     = plasma_input['space_resolution']
        self.chronon_size   = plasma_input['time_resolution']
        self.bundle_count   = plasma_input['bundle_count']
        
        self.mass_number    = plasma_input['mass']   
        self.temp           = plasma_input['temp']
        self.wavelength     = plasma_input['wavelength']
        self.linewidth      = plasma_input['linewidth']
        
        np.random.seed(general_input['random_seed'])
        self.partitioning   = plasma_input['volume_partitioning']
        
    def setup_bundles(self):
        ## Decide how many bundles there will be
        #set up variables
        bundle_input = dict()
        
        if self.partitioning is True:
            self.bundle_count * int(self.volume / self.voxel_size)
        
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
        rays['direction']   = np.zeros([1,3], dtype = np.float64)
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
            source_input['orientation'] = self.orientation
            source_input['target']      = self.target
            source_input['mass']        = self.mass_number
            source_input['wavelength']  = self.wavelength
            source_input['linewidth']   = self.linewidth
            source_input['spread']      = self.plasma_input['spread']
            
            #create ray bundle sources and generate bundled rays
            source       = FocusedExtendedSource(source_input, self.general_input)
            bundled_rays = source.generate_rays()
            
            #append bundled rays together to form a single ray dictionary
            rays['origin'] = np.append(rays['origin'], bundled_rays['origin'], axis = 0)
            rays['direction'] = np.append(rays['direction'], bundled_rays['direction'], axis = 0)
            rays['wavelength'] = np.append(rays['wavelength'], bundled_rays['wavelength'])
            rays['weight'] = np.append(rays['weight'], bundled_rays['weight'])
            rays['mask'] = np.append(rays['mask'], bundled_rays['mask'])
            """
            for key in rays:
                rays[key] = np.append(rays[key], bundled_rays[key], axis = 0)
            """
            profiler.stop("Ray Bundle Generation")
                
        return rays
        
class CubicPlasma(GenericPlasma):
    def __init__(self, plasma_input, general_input):
        super().__init__(plasma_input, general_input)
        self.volume = np.prod(self.dimensions)
        
    def cubic_bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        bundle_input['position']        = np.random.uniform(
                low  = (self.position[:] - self.dimensions[:]),
                high = (self.position[:] + self.dimensions[:]),
                size = [self.bundle_count, 3])
        
        #evaluate temperature at each point
        #plasma cube has consistent temperature throughout
        bundle_input['temp'][:]         = self.temp
        
        #evaluate emissivity at each point
        #plasma cube has constant emissivity throughout
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
