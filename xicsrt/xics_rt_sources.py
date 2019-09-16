# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:12:15 2017
Edited on Fri Sep 06 11:46:40 2019

@author: James Kring
@editor: Eugene
"""

import numpy as np   
from scipy.stats import cauchy        
import scipy.constants as const

from xicsrt.util import profiler
from xicsrt.math import voigt


class GenericSource:
    """
    Source class to hold basic functions for each source type
    """
    def __init__(self, source_input, general_input):
        self.position       = source_input['source_position']
        self.normal         = source_input['source_direction']
        self.xorientation   = source_input['source_orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.width          = source_input['source_width']
        self.height         = source_input['source_height']
        self.depth          = source_input['source_depth']
        self.spread         = source_input['source_spread']
        self.intensity      = source_input['source_intensity']
        self.mass_number    = source_input['source_mass']   
        self.temp           = source_input['source_temp']
        self.wavelength     = source_input['wavelength']
        self.linewidth      = source_input['linewidth']
        np.random.seed(general_input['random_seed'])

    def generate_rays(self):
        # Definitions:
        #   O: origin of ray
        #   D: direction of ray
        #   W: wavelength of ray
        #   w: relative weight of ray (UNUSED)
        #   m: mask applied to each ray
        
        profiler.start('generate_origin')
        O = self.generate_origin()
        profiler.stop('generate_origin')

        profiler.start('generate_direction')
        D = self.generate_direction(O)
        profiler.stop('generate_direction')

        profiler.start('generate_wavelength')
        W = self.generate_wavelength()
        profiler.stop('generate_wavelength')

        profiler.start('generate_weight')
        w = self.generate_weight()
        profiler.stop('generate_weight')
        
        profiler.start('generate_mask')
        m = self.generate_mask()
        profiler.stop('generate_mask')
        
        rays = {'origin': O, 'direction': D, 'wavelength': W, 'weight': w, 'mask': m}      
        return rays
     
    
    def generate_origin(self):
        # generic origin for isotropic rays
        w_offset = np.random.uniform(-1 * self.width/2, self.width/2, self.intensity)
        h_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.intensity)
        d_offset = np.random.uniform(-1 * self.depth/2, self.depth/2, self.intensity)        
                
        origin = (self.position
                  + np.einsum('i,j', w_offset, self.xorientation)
                  + np.einsum('i,j', h_offset, self.yorientation)
                  + np.einsum('i,j', d_offset, self.normal))
        return origin

    def generate_direction(self, origin):
        normal = self.make_normal()
        D = self.random_direction(origin, normal)
        return D

    def make_normal(self):
        array = np.empty((self.intensity, 3))
        array[:] = self.normal
        normal = array / np.linalg.norm(array, axis=1)[:, np.newaxis]
        return normal
        
    def random_direction(self, origin, normal):
        # Pulled from Novi's FocusedExtendedSource
        def f(theta, number):
            output = np.empty((number, 3))
            
            z   = np.random.uniform(np.cos(theta),1, number)
            phi = np.random.uniform(0, np.pi * 2, number)
            
            output[:,0]   = np.sqrt(1-z**2) * np.cos(phi)
            output[:,1]   = np.sqrt(1-z**2) * np.sin(phi)
            output[:,2]   = z
            return output
        
        direction = np.empty(origin.shape)
        rad_spread = np.radians(self.spread)
        dir_local = f(rad_spread, self.intensity)
        
        o_1 = np.cross(normal, [0,0,1])
        o_1 /=  np.linalg.norm(o_1, axis=1)[:, np.newaxis]
        o_2 = np.cross(normal, o_1)
        o_2 /=  np.linalg.norm(o_2, axis=1)[:, np.newaxis]
        
        R = np.empty((self.intensity, 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal
        
        direction = np.einsum('ij,ijk->ik', dir_local, R)
        return direction

    def generate_wavelength(self):
        #random_wavelength = self.random_wavelength_normal
        #random_wavelength = self.random_wavelength_cauchy
        random_wavelength = self.random_wavelength_voigt
        wavelength = random_wavelength(self.intensity)
        return wavelength

    def random_wavelength_voigt(self, size=None):
        """
        Units:
          wavelength: angstroms
          natural_linewith: 1/s
          temperature: eV
        """
        # Check for the trivial case.
        if (self.linewidth  == 0.0 and self.temp == 0.0):
            return np.ones(size)*self.wavelength
        # Check for the Lorentzian case.
        if (self.temp == 0.0):
            # I need to update the cauchy routine first.
            pass
        # Check for the Gaussian case.
        if (self.linewidth  == 0.0):
            return self.random_wavelength_normal(size)

        c = const.physical_constants['speed of light in vacuum'][0]
        amu_kg = const.physical_constants['atomic mass unit-kilogram relationship'][0]
        ev_J = const.physical_constants['electron volt-joule relationship'][0]
        
        # Natural line width.
        gamma = ( self.linewidth * self.wavelength**2
                  / (4 * np.pi * c * 1e10) )

        # Doppler broadened line width.
        sigma = ( np.sqrt(self.temp / self.mass_number / amu_kg / c**2 * ev_J)
                  * self.wavelength )

        rand_wave = voigt.voigt_random(gamma, sigma, size)
        rand_wave += self.wavelength
        return rand_wave

    def random_wavelength_normal(self, size=None):
        """
        Units:
          wavelength: angstroms
          temperature: eV
        """
        c = const.physical_constants['speed of light in vacuum'][0]
        amu_kg = const.physical_constants['atomic mass unit-kilogram relationship'][0]
        ev_J = const.physical_constants['electron volt-joule relationship'][0]
        
        # Doppler broadened line width.
        sigma = ( np.sqrt(self.temp / self.mass_number / amu_kg / c**2 * ev_J)
                  * self.wavelength )

        rand_wave = np.random.normal(self.wavelength, sigma, size)
        return rand_wave
    
    def random_wavelength_cauchy(self, size=None):
        # This function needs to be updated to use the same definitions
        # as random_wavelength_voigt.
        #
        # As currently writen natual_linewidth is not used in a way
        # consistent with physical units.
        #
        # It also may make sense to add some sort of cutoff here.
        # the extreme tails of the distribution are not really useful
        # for ray tracing.
        fwhm = self.linewidth
        rand_wave = cauchy.rvs(loc=self.wavelength, scale=fwhm, size=size)
        return rand_wave
    
    #weight is not yet implemented
    def generate_weight(self):
        intensity = self.intensity
        w = np.ones((intensity,1), dtype=np.float64)
        return w
    
    def generate_mask(self):
        intensity = self.intensity
        m = np.ones((intensity), dtype=np.bool)
        return m

class FocusedExtendedSource(GenericSource):
    def __init__(self, source_input, general_input):
        #super().__init__()
        self.position       = source_input['source_position']
        self.normal         = source_input['source_direction']
        self.xorientation   = source_input['source_orientation']
        self.yorientation   = (np.cross(self.normal, self.xorientation) / 
                               np.linalg.norm(np.cross(self.normal, self.xorientation)))
        self.width          = source_input['source_width']
        self.height         = source_input['source_height']
        self.depth          = source_input['source_depth']
        self.spread         = source_input['source_spread']
        self.intensity      = source_input['source_intensity']
        self.temp           = source_input['source_temp']
        self.mass_number    = source_input['source_mass']
        self.wavelength     = source_input['source_wavelength']                                         
        self.linewidth      = source_input['source_linewidth']
        self.focus = source_input['source_target'] 
        np.random.seed(general_input['random_seed'])

    def generate_rays(self):
        # Definition:
        #   O: origin of ray
        #   D: direction of ray
        #   W: wavelength of ray
        #   w: weight of ray (UNUSED)
        
        profiler.start('generate_origin')
        O = super().generate_origin()
        profiler.stop('generate_origin')

        profiler.start('generate_direction')
        D = self.generate_direction(O)
        profiler.stop('generate_direction')

        profiler.start('generate_wavelength')
        W = super().generate_wavelength()
        profiler.stop('generate_wavelength')
        
        profiler.start('generate_weight')
        w = super().generate_weight()
        profiler.stop('generate_weight')
        
        profiler.start('generate_mask')
        m = super().generate_mask()
        profiler.stop('generate_mask')
        
        rays = {'origin': O, 'direction': D, 'wavelength': W, 'weight': w, 'mask': m}      
        return rays
    
    def generate_direction(self, origin):
        normal = self.make_normal_focused(origin)
        D = super().random_direction(origin, normal)
        return D
    
    def make_normal_focused(self, origin):
        # Generate ray from the origin to the focus.
        normal = self.focus - origin
        normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
        return normal

class ExtendedSource(GenericSource):
    def __init__(
            self
            ,position=None
            ,normal=None
            ,orientation=None
            ,width =None
            ,height=None
            ,depth=None
            ,spread=None
            ,intensity=None
            ,wavelength=None
            ,temperature=None
            ,mass_number=None
            ,linewidth=None
            ):
        
        super().__init__(
            position=position
            ,normal=normal
            ,orientation=orientation
            ,width =width 
            ,height=height
            ,depth=depth
            ,spread=spread
            ,intensity=intensity
            ,wavelength=wavelength
            ,temperature=temperature
            ,mass_number=mass_number
            ,linewidth=linewidth
            )

class DirectedSource(GenericSource):
    def __init__(
            self
            ,position=None
            ,normal=None
            ,spread=None
            ,intensity=None
            ,wavelength=None
            ,temperature=None
            ,mass_number=None
            ,linewidth=None
            ):
        
        super().__init__(
            position=position
            ,normal=normal
            ,orientation=(np.cross(normal, position)/ 
                             np.linalg.norm(np.cross(normal, position)))
            ,width =0
            ,height=0
            ,depth=0
            ,spread=spread
            ,intensity=intensity
            ,wavelength=wavelength
            ,temperature=temperature
            ,mass_number=mass_number
            ,linewidth=linewidth
            )    
     
    def get_orientation(self):
        self.orientation = (np.cross(self.normal, self.position)/ 
                             np.linalg.norm(np.cross(self.normal, self.position)))       