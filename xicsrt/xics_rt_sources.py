# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:12:15 2017

@author: James Kring
"""
import numpy as np   
from scipy.stats import cauchy        
import scipy.constants as constants

from xicsrt.util import profiler
from xicsrt.math import voigt

class PointSource(object):
    
    """Create a point source object that emits light.
    Emits in four pi steradians.
    Wavelength in angstroms.
    """
    
    def __init__(self, position, intensity, wavelength, temperature, 
                 mass_number, linewidth):
        self.position           = position
        self.wavelength         = wavelength
        self.intensity          = intensity #photons per second
        self.temp               = temperature
        self.natural_linewidth  = linewidth
        self.mass_number        = mass_number
        
        
    def random_direction(self):
        directions = []
        
        append = directions.append
        
        i = 0
        for i in range(0, self.intensity):
            direction = np.random.uniform(-1, 1, 3)
            direction = direction/np.linalg.norm(direction)
            append(direction)
            i += 1        
        
        D = np.array(directions)
        return D  

    
    def random_wavelength_normal(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
        
        
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
    
        
        return rand_wave


    def random_wavelength_cauchy(self):

        fwhm = self.natural_linewidth

        rand_wave = cauchy.rvs(loc = self.wavelength, scale = fwhm, size = 1)
        
        return rand_wave
    

    def generate_origin(self):
        origins = []
        
        append = origins.append
        
        i = 0
        for i in range(0, self.intensity):
            origin = [self.position[0], self.position[1], self.position[2]]
            append(origin)
            i += 1        
        
        O = np.array(origins)
        return O
        
    
    def generate_direction(self):
        D = self.random_direction()
        
        return D


    def generate_wavelength(self):
        wavelengths = []
        
        intensity = self.intensity
        append = wavelengths.append
        #random_wavelength = self.random_wavelength_normal
        random_wavelength = self.random_wavelength_cauchy
        
        i = 0
        for i in range(0, intensity):
            wavelength = random_wavelength()
            append(wavelength)
            i += 1
            
        W = np.array(wavelengths)
        
        return W   
      

    def generate_weight(self):
        intensity = self.intensity

        w = np.ones(intensity,dtype=np.float64)
        
        return w
    
    
    def generate_rays(self):

        O = self.generate_origin()    
        D = self.generate_direction()
        W = self.generate_wavelength()
        w = self.generate_weight()


        return O, D, W, w



        
class DirectedSource(object):
    
    """Create a directed source object that emitts light in a defined cone. 
    Wavelength in angstroms.
    """
    
    def __init__(self, position, direction, spread, intensity, wavelength, 
                 temperature, mass_number, linewidth):
        self.position = position
        self.direction = direction
        self.spread = spread             # in degrees
        self.wavelength = wavelength     # in angstroms
        self.intensity = intensity       # photons per second

        self.xorientation = (np.cross(self.direction, self.position)/ 
                             np.linalg.norm(np.cross(self.direction, self.position)))
        self.yorientation = (np.cross(self.xorientation, self.direction) / 
                             np.linalg.norm(np.cross(self.xorientation,
                                                     self.direction)))    
        self.temp = temperature          # in eV
        self.mass_number = mass_number   # mass number of source material
        self.natural_linewidth = linewidth
        
          
    def random_direction(self):  
        from scipy.linalg import sqrtm, inv


        def sym(w):
            return w.dot(inv(sqrtm(w.T.dot(w))))

        
        def f(theta):
            z   = np.random.uniform(np.cos(theta),1)
            phi = np.random.uniform(0, np.pi * 2)
            x   = np.sqrt(1-z**2) * np.cos(phi)
            y   = np.sqrt(1-z**2) * np.sin(phi)
            z   = z
            
            return [x, y, z]  
        
        
        R = ([self.xorientation, self.yorientation, self.direction])
        R = np.transpose(R) 
        R = sym(R)      
        
        directions = []
        
        rad_spread = self.spread/180 * np.pi
        intensity = self.intensity
        
        append = directions.append
        flatten = np.ndarray.flatten
        dot = np.dot

        i = 0
        for i in range(0, intensity):
            direction = f(rad_spread)
            direction = dot(R, direction) 
            direction = flatten(direction)
            append(direction)
            i += 1        
        
        D = np.array(directions)
        return D  
    
    
    def random_wavelength_normal(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
                
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave

    
    def random_wavelength_cauchy(self):
        fwhm = self.natural_linewidth
        rand_wave = cauchy.rvs(loc = self.wavelength, scale = fwhm, size = 1)
    
        return rand_wave

        
    def generate_origin(self):
        origin = [[self.position[0], self.position[1], self.position[2]]] * self.intensity
        O = np.array(origin)
        return O
        
    
    def generate_direction(self):
        D = self.random_direction()
        
        return D


    def generate_wavelength(self):
        wavelengths = []
        
        intensity = self.intensity
        append = wavelengths.append
        #random_wavelength = self.random_wavelength_normal
        random_wavelength = self.random_wavelength_cauchy
        
        i = 0
        for i in range(0, intensity):
            wavelength = random_wavelength()
            append(wavelength)
            i += 1
            
        W = np.array(wavelengths)
        
        return W   
      

    def generate_weight(self):
        intensity = self.intensity

        w = np.ones(intensity,dtype=np.float64)
        
        return w
    
                
    def generate_rays(self):

        O = self.generate_origin()    
        D = self.generate_direction()
        W = self.generate_wavelength()
        w = self.generate_weight()

        return O, D, W, w
        
        
        
    
class UniformAnalyticSource(object):
    
    """Point source that will generate rays that uniformly illuminate the
    crystal. Points are spread on a grid on the crystal. 
    Directions are then measured from the source location to every point on 
    the grid on the crystal.
    """
    
    def __init__(self, position, direction, spread, intensity, wavelength, 
                 temperature, mass_number, linewidth, crystal):
        self.position = position
        self.direction = direction
        self.spread = spread             # in degrees
        self.wavelength = wavelength     # in angstroms
        self.intensity =  len(crystal.create_center_array_new())

        self.xorientation = (np.cross(self.direction, self.position)/ 
                             np.linalg.norm(np.cross(self.direction, self.position)))
        self.yorientation = (np.cross(self.xorientation, self.direction) / 
                             np.linalg.norm(np.cross(self.xorientation,
                                                     self.direction)))    
        self.temp = temperature          # in eV
        self.mass_number = mass_number   # mass number of source material
        self.crystal = crystal
        self.natural_linewidth = linewidth
        
        
    def theta(self, direction):
        angle = np.arccos(direction[2]/np.linalg.norm(direction))
        return angle    
        
    
    def uniform_direction_new(self):      
        
        v_origin = self.position
        crystal = self.crystal
        duration = 1
        points_c = crystal.create_center_array_new()
        v_direc = []
        
        for i in range(0, duration):
            for point in points_c:
                direction = point - v_origin
                direction = direction/np.linalg.norm(direction)
                direction = direction.tolist()
                v_direc.append(direction)
        
        return np.array(v_direc)

        
    def random_wavelength_normal(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
                
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave

    
    def random_wavelength_cauchy(self):
        fwhm = self.natural_linewidth
        rand_wave = cauchy.rvs(loc = self.wavelength, scale = fwhm, size = 1)
    
        return rand_wave
        

    def generate_origin(self):
        origin = [[self.position[0], self.position[1], self.position[2]]] * self.intensity
        O = np.array(origin)
        return O
        
    
    def generate_direction(self):
        D = self.uniform_direction_new()
        
        return D


    def generate_wavelength(self):
        wavelengths = []
        
        intensity = self.intensity
        append = wavelengths.append
        #random_wavelength = self.random_wavelength_normal
        random_wavelength = self.random_wavelength_cauchy
        
        i = 0
        for i in range(0, intensity):
            wavelength = random_wavelength()
            append(wavelength)
            i += 1
            
        W = np.array(wavelengths)
        
        return W   
      

    def generate_weight(self):
        intensity = self.intensity
        w = np.ones(intensity,dtype=np.float64)
        
        return w   
    
 
    def generate_rays(self):
        profiler.start('generate_origin')
        O = self.generate_origin()
        profiler.stop('generate_origin')

        profiler.start('generate_direction')
        D = self.generate_direction()
        profiler.stop('generate_direction')

        profiler.start('generate_wavelength')
        W = self.generate_wavelength()
        profiler.stop('generate_wavelength')

        profiler.start('generate_weight')
        w = self.generate_weight()
        profiler.stop('generate_weight')

        return O, D, W, w
    
    
    
    
class ExtendedSource:
    """
    Planar Source that emits in cone.
    """
    
        
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
        
        self.position = position
        self.xorientation = orientation
        self.yorientation = (np.cross(normal, orientation) / 
                             np.linalg.norm(np.cross(normal, orientation)))
        self.normal = normal
        self.width = width
        self.height = height
        self.depth = depth
        self.spread = spread
        self.intensity = intensity
        self.wavelength = wavelength
        self.temp = temperature
        self.mass_number = mass_number
        self.natural_linewidth = linewidth
    

    def random_direction(self):  
        from scipy.linalg import sqrtm, inv


        def sym(w):
            return w.dot(inv(sqrtm(w.T.dot(w))))

        
        def f(theta):
            z   = np.random.uniform(np.cos(theta),1)
            phi = np.random.uniform(0, np.pi * 2)
            x   = np.sqrt(1-z**2) * np.cos(phi)
            y   = np.sqrt(1-z**2) * np.sin(phi)
            z   = z
            
            return [x, y, z]  
        
        
        R = ([self.xorientation, self.yorientation, self.normal])
        R = np.transpose(R) 
        R = sym(R)      
        
        directions = []
        
        rad_spread = np.radians(self.spread)
        intensity = self.intensity
        
        append = directions.append
        flatten = np.ndarray.flatten
        dot = np.dot

        i = 0
        for i in range(0, intensity):
            direction = f(rad_spread)
            direction = dot(R, direction) 
            direction = flatten(direction)
            append(direction)
            i += 1        
        
        D = np.array(directions)
        return D   
        

    def random_wavelength_voigt(self, size=None):
        """
        Units:
          wavelength: angstroms
          natural_linewith: 1/s
          temperature: eV
        """

        c = const.physical_constants['speed of light in vacuum'][0]
        amu_kg = const.physical_constants['atomic mass unit-kilogram relationship'][0]
        ev_j = const.physical_constants['electron volt-joule relationship'][0]
        
        # Natural line width.
        gamma = ( self.natural_linewidth * self.wavelength**2
                  / (4 * np.pi * c * 1e10) )

        # Doppler broadened line width.
        sigma = ( np.sqrt(self.temp / self.mass_number / amu_kg / c**2 * ev_J * 1e3)
                  * self.wavelength )

        rand_wave = voigt_random(gamma, sigma, size)
        rand_wave += self.wavelength
        
        return rand_wave

    
    def random_wavelength_normal(self, size=None):
        
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
                
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq
        rand_freq = np.random.normal(mean_freq, sigma, size)
        rand_wave = c / rand_freq
        
        return rand_wave

    
    def random_wavelength_cauchy(self, size=None):
        fwhm = self.natural_linewidth
        rand_wave = cauchy.rvs(loc = self.wavelength, scale = fwhm, size = size)
    
        return rand_wave


    def generate_direction(self):
        D = self.random_direction()
        
        return D    
    
    def random_origin(self):
        half_width = self.width/2
        half_height = self.height/2
        half_depth = self.depth/2
        
        
        w_offset = np.random.uniform(-1 * half_width, half_width, 1)[0]
        h_offset = np.random.uniform(-1 * half_height, half_height,1)[0]
        d_offset = np.random.uniform(-1 * half_depth, half_depth, 1)[0]
        
        position = (self.position + w_offset * self.xorientation
                                  + h_offset * self.yorientation
                                  + d_offset * self.normal)        
        
        return position
        
        
    def generate_origin(self):

        w_offset = np.random.uniform(-1 * self.width/2, self.width/2, self.intensity)
        h_offset = np.random.uniform(-1 * self.height/2, self.height/2, self.intensity)
        d_offset = np.random.uniform(-1 * self.depth/2, self.depth/2, self.intensity)        

                
        origin = (self.position
                  + np.einsum('i,j', w_offset, self.xorientation)
                  + np.einsum('i,j', h_offset, self.yorientation)
                  + np.einsum('i,j', d_offset, self.normal))
        
        return origin


    def generate_wavelength(self):
        #random_wavelength = self.random_wavelength_normal
        #random_wavelength = self.random_wavelength_cauchy
        random_wavelength = self.random_wavelength_normal
        
        wavelength = random_wavelength(self.intensity)
        wavelength = wavelength[:, np.newaxis]
        
        return wavelength

    
    def generate_weight(self):
        intensity = self.intensity
        w = np.ones((intensity,1), dtype=np.float64)
        
        return w
    
        
    def generate_rays(self):

        O = self.generate_origin()    
        D = self.generate_direction()
        W = self.generate_wavelength()
        w = self.generate_weight()

        return O, D, W, w       


class FocusedExtendedSource(ExtendedSource):

    
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
            ,focus=None
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

        self.focus = focus    

        
    def generate_rays(self):

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
        
        return O, D, W, w

    

    def generate_direction(self, origin):
        D = self.random_direction(origin)
        
        return D

    
    def random_direction(self, origin):  
        
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

        
        #for ii in range(self.intensity):
        #    normal = self.focus - origin[ii,:]
        #    normal /= np.linalg.norm(normal)

        #    # Here it doesn't matter which direction the orientation vectors are
        #    # as long as they are orthoganal.
        #    orientation_1 = np.cross(normal, [0,0,1])
        #    orientation_1 /= np.linalg.norm(orientation_1)

        #    orientation_2 = np.cross(normal, orientation_1)
        #    orientation_2 /= np.linalg.norm(orientation_2)

        #    R = ([orientation_1, orientation_2, normal])
        
        #    direction[ii, :] = np.dot(dir_local[ii,:], R) 


        normal = self.focus - origin
        normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]

        o_1 = np.cross(normal, [0,0,1])
        o_1 = o_1 /  np.linalg.norm(o_1, axis=1)[:, np.newaxis]
        
        o_2 = np.cross(normal, o_1)
        o_2 = o_2 /  np.linalg.norm(o_2, axis=1)[:, np.newaxis]
        
        R = np.empty((self.intensity, 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal

        direction = np.einsum('ij,ijk->ik',dir_local, R)
        
        return direction
