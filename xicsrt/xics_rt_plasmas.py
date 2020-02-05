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
        super().__init__(
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
        self.bundle_type    = str.upper(plasma_input['bundle_type'])
        self.bundle_factor  = plasma_input['bundle_factor']
        self.bundle_volume  = plasma_input['bundle_volume']
        self.solid_angle    = 4 * np.pi * np.sin(plasma_input['spread'] * np.pi / 360) ** 2
        self.voxel_size     = self.bundle_volume ** (1/3)
        self.chronon_size   = plasma_input['time_resolution']
        self.max_rays       = plasma_input['max_rays']
        #profile information
        self.use_profiles   = plasma_input['use_profiles']
        self.temp_data      = plasma_input['temperature_data']
        self.emis_data      = plasma_input['emissivity_data']
        self.velo_data      = plasma_input['velocity_data']
        self.temperature    = plasma_input['temperature']
        self.emissivity     = plasma_input['emissivity']
        self.velocity       = plasma_input['velocity']
        self.mass_number    = plasma_input['mass']
        self.wavelength     = plasma_input['wavelength']
        self.linewidth      = plasma_input['linewidth']
        
    def setup_bundles(self):
        #plasma volume, bundle volume, and bundle count are linked
        if self.bundle_type == 'POINT':
            self.bundle_count = int(self.volume / (self.bundle_volume * self.bundle_factor))
        elif self.bundle_type == 'VOXEL':
            self.bundle_volume = self.volume / (self.bundle_count * self.bundle_factor)
            self.voxel_size   = self.bundle_volume ** (1/3)
        
        if self.bundle_count >= self.max_rays:
            raise ValueError('Plasma generated too many bundles. Please increase bundle factor.')
        
        bundle_input = {}
        bundle_input['position']     = np.zeros([self.bundle_count, 3], dtype = np.float64)
        bundle_input['temperature']  = np.ones([self.bundle_count], dtype = np.float64)
        bundle_input['emissivity']   = np.ones([self.bundle_count], dtype = np.int)
        bundle_input['velocity']     = np.zeros([self.bundle_count, 3], dtype = np.float64)
        bundle_input['sightline']    = np.ones([self.bundle_count], dtype = np.bool)
        
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
        
        #determine which bundles are near the sightline; if there is no 
        #sightline, the plasma defaults to generating all bundles
        m = bundle_input['sightline']
        print(' Bundles Generated: {:6.4e}'.format(len(m[m])))
        
        #test to see if the number of rays generated will exceed max ray limits
        predicted_rays = int(np.sum(bundle_input['emissivity'][m])
                         * self.chronon_size
                         * self.bundle_volume
                         * self.solid_angle / (4 * np.pi)
                         * self.volume / (len(m[m]) * self.bundle_volume))

        if predicted_rays >= self.max_rays:
            raise ValueError('Plasma generated too many rays. Please reduce integration time.')
        
        #bundle generation loop
        for ii in range(len(m[m])):
            profiler.start("Ray Bundle Generation")
            source_input = OrderedDict()
            #spacially-dependent parameters
            
            source_input['position']    = bundle_input['position'][m][ii]
            source_input['temp']        = bundle_input['temperature'][m][ii]
            source_input['velocity']    = bundle_input['velocity'][m][ii]

            # Calculate the total number of photons to launch from this bundle volume.
            intensity = (bundle_input['emissivity'][m][ii]
                         * self.chronon_size
                         * self.bundle_volume
                         * self.solid_angle / (4 * np.pi))
            
            # Scale the number of photons based on the number of bundles
            #
            # bundle_volume cancels out here, each bundle represents an area of
            # volume/bundle_count.  I am leaving the calculation as is for now
            # for clarity in case a different approach is needed in the future.
            intensity *= self.volume / (self.bundle_volume * len(m[m]))

            source_input['intensity'] = intensity
            #constants
            source_input['width']       = self.voxel_size
            source_input['height']      = self.voxel_size
            source_input['depth']       = self.voxel_size
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
        bundle_input['temperature'][:]         = self.temperature
        
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
        bundle_input['temperature']  = self.temperature / radius
        
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
        super().__init__(plasma_input)
        self.position       = plasma_input['position']
        
        self.width          = plasma_input['width']
        self.height         = plasma_input['height']
        self.depth          = plasma_input['depth']
        self.volume         = self.width * self.height * self.depth
        self.major_radius   = plasma_input['major_radius']
        self.minor_radius   = plasma_input['minor_radius']
        self.bundle_count   = plasma_input['bundle_count']
        
        self.sight_position = plasma_input['sight_position']
        self.sight_direction= plasma_input['sight_direction']
        self.sight_thickness= plasma_input['sight_thickness']
        
    def toroidal_bundle_generate(self, bundle_input):
        #create a long list containing random points within the cube's dimensions
        x_offset = np.random.uniform(-1 * self.width/2 , self.width/2 , self.bundle_count)
        y_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.bundle_count) + self.major_radius
        z_offset = np.random.uniform(-1 * self.depth/2 , self.depth/2 , self.bundle_count)        
        
        #unlike the other plasmas, the toroidal plasma has fixed orientation to
        #prevent confusion
        bundle_input['position'][:] = (self.position
                  + np.einsum('i,j', x_offset, np.array([1, 0, 0]))
                  + np.einsum('i,j', y_offset, np.array([0, 1, 0]))
                  + np.einsum('i,j', z_offset, np.array([0, 0, 1])))
        
        #calculate whether the ray bundles are within range of the sightline
        #vector from sightline origin to bundle position
        l_0 = self.sight_position - bundle_input['position']
        #projection of l_0 onto the sightline
        proj = np.einsum('j,ij->i',self.sight_direction, l_0)[np.newaxis]
        l_1  = np.dot(np.transpose(self.sight_direction[np.newaxis]), proj)
        l_1  = np.transpose(l_1)
        #component of l_0 perpendicular to the sightline
        l_2 = l_0 - l_1
        #sightline distance is the length of l_2
        distance = np.einsum('ij,ij->i', l_2, l_2) ** .5
        #check to see if the bundle is close enough to the sightline
        bundle_input['sightline'][:] = (self.sight_thickness >= distance)
        
        #convert from cartesian coordinates to toroidal coordinates [rad, pol, tor]
        #torus is oriented along the Z axis
        rad, pol, tor = cart2toro(x_offset, y_offset, z_offset, self.major_radius)
        
        #evaluate temperature and emissivity at each point
        if self.use_profiles is False:
            #step function profile
            step_test    = np.zeros(self.bundle_count, dtype = np.bool)
            step_test[:] = (rad <= self.minor_radius)
            bundle_input['temperature'][step_test]  = self.temperature
            bundle_input['emissivity'][step_test]   = self.emissivity
            bundle_input['velocity'][step_test]     = self.velocity
            
        if self.use_profiles is True:
            #read and interpolate profile from data file
            temp_data  = np.loadtxt(self.temp_data, dtype = np.float64)
            emis_data  = np.loadtxt(self.emis_data, dtype = np.float64)
            #velo_vata  = np.loadtxt(self.velo_data, dtype = np.float64)
            nrad       = rad / self.minor_radius
            
            bundle_input['temperature'] = np.interp(nrad, temp_data[:,0], temp_data[:,1],
                                                    left = 1.0, right = 1.0)
            bundle_input['emissivity']  = np.interp(nrad, emis_data[:,0], emis_data[:,1],
                                                    left = 1.0, right = 1.0)
            """
            bundle_input['velocity'][0] = np.interp(nrad, emis_data[:,0], emis_data[:,1],
                                        left = 1.0, right = 1.0)
            bundle_input['velocity'][1] = np.interp(nrad, emis_data[:,0], emis_data[:,2],
                                        left = 1.0, right = 1.0)      
            bundle_input['velocity'][2] = np.interp(nrad, emis_data[:,0], emis_data[:,3],
                                        left = 1.0, right = 1.0)
            """
        return bundle_input

    def generate_rays(self):
        ## Create an empty list of ray bundles
        bundle_input = self.setup_bundles()
        ## Populate that list with ray bundle parameters, like locations
        bundle_input = self.toroidal_bundle_generate(bundle_input)
        ## Use the list to generate ray sources
        rays = self.create_sources(bundle_input)
        return rays