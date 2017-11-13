# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:12:15 2017

@author: James
"""
import numpy as np   
        

class PointSource(object):
    
    """Create a point source object that emitts light of a certain wavelength.
    Emits in four pi steradians.
    Wavelength in angstroms.
    """
    
    def __init__(self, position, intensity, wavelength, temperature):
        self.position = position
        self.wavelength = wavelength
        self.intensity = intensity #photons per second
        self.temp = temperature
        
    def random_direction(self):
        direction = np.random.uniform(-1, 1, 3)
        direction = direction/np.linalg.norm(direction)
        return direction.tolist()
        
        
    def random_wavelength(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
        
        
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave
        
    
    def generate_rays(self,duration):

        number_of_rays = self.intensity * duration 
        origin = [[self.position[0], self.position[1], self.position[2]]] * number_of_rays
        O = np.array(origin) 
        
        ray_list = None
        ray_list = []

        wavelength_list = None
        wavelength_list = []

        i = 0
        for i in range(0, number_of_rays):
            ray = self.random_direction()
            ray_list.append(ray)
            
            wavelength = self.random_wavelength()
            wavelength_list.append(wavelength)
            
            i += 1
            
        D = np.array(ray_list)
        W = np.array(wavelength_list)
        
        return O, D, W




        
class DirectedSource(object):
    
    """Create a directed source object that emitts light of a certain 
    wavelength in a defined cone. Wavelength in angstroms.
    """
    
    def __init__(self, position, direction, spread, intensity, wavelength, 
                 temperature, mass_number):
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
        
        
    def theta(self, direction):
        angle = np.arccos(direction[2]/np.linalg.norm(direction))
        #print(angle)
        return angle    
        
    
     
    def random_direction(self):      
        
        directions = []
        
        i = 0
        for i in range(0, self.intensity):
            direction = [np.random.uniform(-.7, .7, 1)[0],
                         np.random.uniform(-.7, .7, 1)[0],
                         np.random.uniform(0, 1, 1)[0]]
            direction = direction/np.linalg.norm(direction)
            spread_rad = self.spread / 180. * np.pi
        
            
            while (self.theta(direction) > spread_rad):
                direction = [np.random.uniform(-.7, .7, 1)[0],
                             np.random.uniform(-.7, .7, 1)[0],
                             np.random.uniform(0, 1, 1)[0]]
            

            R = ([self.xorientation, self.yorientation, self.direction])
            R = np.transpose(R)

            direction = np.dot(R, direction)
            direction = np.ndarray.flatten(direction)      
            direction = direction/np.linalg.norm(direction)
    
            directions.append(direction.tolist())
            i += 1         
        
        D = np.array(directions)
        return D
           
    
    def random_direction_fortran(self):  
        from third import f
        directions = []
        
        rad_spread = self.spread/180 * np.pi
        intensity = self.intensity
        
        append = directions.append
        norm = np.linalg.norm
        flatten = np.ndarray.flatten
        dot = np.dot
        
        
        
        def normalize(direction):
            return direction/norm(direction)
        
        R = ([self.xorientation, self.yorientation, self.direction])
        R = np.transpose(R)    
        
        i = 0
        for i in range(0, intensity):
            direction = f(1, rad_spread)
            direction = flatten(direction)
            direction = dot(R, direction) 
            direction = flatten(direction)
            direction = normalize(direction)
            append(direction)
            i += 1        
        
        D = np.array(directions)
        return D  
    
    
    def random_wavelength(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
        
        
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave
    
        
    def generate_origin(self):
        origin = [[self.position[0], self.position[1], self.position[2]]] * self.intensity
        O = np.array(origin)
        return O
        
    
    def generate_direction(self):
        #Ordered Oldest to Newest
        #D = self.random_direction()
        D = self.random_direction_fortran()
        
        return D


    def generate_wavelength(self):
        wavelengths = []
        
        intensity = self.intensity
        append = wavelengths.append
        random_wavelength = self.random_wavelength
        
        i = 0
        for i in range(0, intensity):
            wavelength = random_wavelength()
            append(wavelength)
            i += 1
            
        W = np.array(wavelengths)
        
        return W
    
        
    def generate_rays(self):

        O = self.generate_origin()    
        D = self.generate_direction()
        W = self.generate_wavelength()

        return O, D, W
        
        
        
    
class UniformAnalyticSource(object):
    
    """Point source that will generate rays that uniformly illuminate the
    crystal. Points are spread on a grid on the crystal. 
    Directions are then measured from the source location to every point on 
    the grid on the crystal.
    """
    
    def __init__(self, position, direction, spread, intensity, wavelength, 
                 temperature, mass_number, crystal):
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
        self.crystal = crystal
        
        
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
        

        return v_direc 

        
    def random_wavelength(self):
        c = 3.00e18                         # angstroms per sec
        conv = 931.4940954e6                # eV per atomic u
        mc2 = self.mass_number * conv       # mass energy in eV     
        
        
        mean_wave = self.wavelength
        mean_freq =  c / mean_wave
        
        sigma = np.sqrt(self.temp / mc2) * mean_freq

        rand_freq = np.random.normal(mean_freq, sigma, 1)
        rand_wave = c / rand_freq
        
        return rand_wave    
        
        
    def generate_rays(self):        
        ray_list = None
        ray_list = self.uniform_direction_new()
        
        number_of_rays = len(ray_list) 
        origin = [[self.position[0], self.position[1], self.position[2]]] * number_of_rays
        O = np.array(origin) 
        
        wavelength_list = None
        wavelength_list = []
        
        i = 0
        for i in range(0, number_of_rays):

            wavelength = self.random_wavelength()
            wavelength_list.append(wavelength)
            
            i += 1
            
        D = np.array(ray_list)
        W = np.array(wavelength_list)
        
        return O, D, W