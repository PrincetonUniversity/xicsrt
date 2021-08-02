# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
"""

import numpy as np   
from scipy.stats import cauchy        
import scipy.constants as const

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_voigt
from xicsrt.tools import xicsrt_spread
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._RayArray import RayArray
from xicsrt.objects._GeometryObject import GeometryObject

@dochelper
class XicsrtSourceGeneric(GeometryObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_objects = []
            
    def default_config(self):
        """
        xsize
          The size of this element along the xaxis direction.

        ysize
          The size of this element along the yaxis direction.

        zsize
          The size of this element along the zaxis direction.

        angular_dist : string ('isotropic')
          | The type of angular distribution to use for the emitted rays.
          | Available distributions (default is 'isotropic'):
          |
          | isotropic
          |   Isotropic emission (uniform spherical) emitted in a cone (circular
          |   cross-section) with a half-angle of 'spread'. The axis of the
          |   emission cone is aligned along the z-axis. 'spread' must be a
          |   single value (scalar or 1-element array).
          | isotropic_xy
          |   Isotropic emission (uniform spherical) emitted in a truncated-cone
          |   (rectangular cross-section) with different x and y half-angles.
          |   'spread' can contain either 1, 2 or 4 values:
          |   s or [s]
          |     A single value that will be used for both the x and y directions.
          |   [x, y]
          |     Two values values that will be used for the x and y directions.
          |   [xmin, xmax, ymin, ymax]
          |     For values that define the asymmetric exent in x and y directions.
          |     Example: [-0.1, 0.1, -0.5, 0.5]
          | flat
          |   Flat emission (uniform planar) emitted in a cone (circular cross-
          |   section) with a half-angle of 'spread'.
          | flat_xy
          |   Flat emission (uniform planar) emitted in a truncated-cone
          |   (rectangular cross-section) with different x and y half-angles.
          |   'spread' can contain either 1, 2 or 4 values, see above.
          | gaussian
          |   Emission with angles away from the z-axis having a Gaussian
          |   distribution (circular cross-section). The 'spread' defines the
          |   Half-Width-at-Half-Max (HWHM) of the distribution. 'spread' must
          |   be a single value (scalar or 1-element array).
          | gaussian_flat
          |   !! Not implemented !!
          |   Cross-section of emission (intersection with constant-z plane) will
          |   have a Gaussian distribution.

        spread : float or array (np.pi) [radians]
          The angular spread for the emission cone. The spread defines the
          half-angle of the emission cone. See 'angular_dist' for detailed
          documentation.

        intensity : int or float
          The number of rays for this source to emit. This should be an
          integer value unless `use_poisson = True`.

          Note: If filters are attached, this will be the number of rays
          emitted before filtering.

        use_poisson : bool (False)
          If `True` the `intenisty` will be treated as the expected value for
          a Poisson distribution and the number of rays will be randomly
          picked from a Poisson distribution. This is setting is typically
          only used internally for Plasma sources.

        wavelength_dist : str ('voigt')
          The type of wavelength distribution for this source.
          Possible values are: 'voigt', 'uniform', 'monochrome'.

          Note: A monochrome distribution can also be achieved by using a
          'voigt' distribution with zero linewidth and temperature.

        wavelength : float (1.0) [angstroms]
          Only used if `wavelength_dist = "monochrome" or "voigt"`
          Central wavelength of the distribution, in Angstroms.

        wavelength_range: tuple [angstroms]
          Only used if `wavelength_dist = "uniform"`
          The wavelength range of the distribution, in Angstroms.
          Must be a 2 element tuple, list or array: (min, max).

        linewidth : float (0.0) [1/s]
          Only used if `wavelength_dist = "voigt"`
          The natural width of the emission line.
          This will control the Lorentzian contribution to the the overall Voigt
          profile. If linewidth == 0, the resulting wavelength distribution will
          be gaussian.

          To convert from a fwhm in [eV]:
          linewidth = 2*pi*e/h*fwhm_ev

          To translate from linewidth to gamma in the Voigt equation:
          gamma = linewidth * wavelength**2 / (4*pi*c*1e10)

        mass_number : float (1.0) [au]
          Only used if `wavelength_dist = "voigt"`
          The mass of the emitting atom in atomic units (au). This mass in used
          to convert temperature into line width. See temperature option.

        temperature : float (0.0) [eV]
          Only used if `wavelength_dist = "voigt"`
          The temperature of the emission line.
          This will control the Gaussian contribution to the overall Voigt
          profile. If temperature == 0, the resulting wavelength distribution
          will be Lorentzian.

          To translate from temperature to sigma in the Voigt equation:
          sigma = np.sqrt(temperature/mass_number/amu_kg/c**2*ev_J)*wavelength

        velocity
          No documentation yet. Please help improve XICSRT!

        filters
          No documentation yet. Please help improve XICSRT!

        """
        config = super().default_config()

        config['xsize'] = 0.0
        config['ysize'] = 0.0
        config['zsize'] = 0.0

        config['intensity']        = 0.0
        config['use_poisson']      = False
        config['angular_dist']      = 'isotropic'
        config['spread']           = np.pi

        # Possible values: 'monochrome', 'voigt', 'uniform
        config['wavelength_dist']  = 'voigt'

        # Only used for wavelength_dist = 'voigt' or 'monochrome'
        config['wavelength']       = 1.0

        # Only used for wavelength_dist = 'voigt'
        config['mass_number']      = 1.0
        config['linewidth']        = 0.0
        config['temperature']      = 0.0
        config['velocity']         = np.array([0.0, 0.0, 0.0])

        # Only used for wavelength_dist = 'uniform'
        config['wavelength_range'] = np.array([0.0, 0.0])
        
        config['filters']          = []

        return config

    def initialize(self):
        super().initialize()
        
        if self.param['use_poisson']:
            self.param['intensity'] = np.random.poisson(self.param['intensity'])
        else:
            if self.param['intensity'] < 1:
                raise ValueError('intensity of less than one encountered. Turn on poisson statistics.')
        self.param['intensity'] = int(self.param['intensity'])
        
    def generate_rays(self):
        rays = RayArray()
        profiler.start('generate_rays')
        
        profiler.start('generate_origin')
        rays['origin'] = self.generate_origin()
        profiler.stop('generate_origin')

        profiler.start('generate_direction')
        rays['direction'] = self.generate_direction(rays['origin'])
        profiler.stop('generate_direction')

        profiler.start('generate_wavelength')
        rays['wavelength'] = self.generate_wavelength(rays['direction'])
        profiler.stop('generate_wavelength')

        profiler.start('generate_weight')
        rays['weight'] = self.generate_weight()
        profiler.stop('generate_weight')
        
        profiler.start('generate_mask')
        rays['mask'] = self.generate_mask()
        profiler.stop('generate_mask')
        
        profiler.start('filter_rays')
        rays = self.ray_filter(rays)
        profiler.stop('filter_rays')        
        
        profiler.stop('generate_rays')
        return rays
     
    def generate_origin(self):
        # generic origin for isotropic rays
        x_offset = np.random.uniform(-1 * self.param['xsize']/2 ,  self.param['xsize']/2, self.param['intensity'])
        y_offset = np.random.uniform(-1 * self.param['ysize']/2, self.param['ysize']/2, self.param['intensity'])
        z_offset = np.random.uniform(-1 * self.param['zsize']/2 ,  self.param['zsize']/2, self.param['intensity'])
        
        origin = (self.origin
                  + np.einsum('i,j', x_offset, self.xaxis)
                  + np.einsum('i,j', y_offset, self.yaxis)
                  + np.einsum('i,j', z_offset, self.zaxis))
        return origin

    def generate_direction(self, origin):
        normal = self.make_normal()
        D = self.random_direction(normal)
        return D

    def make_normal(self):
        array = np.empty((self.param['intensity'], 3))
        array[:] = self.param['zaxis']
        normal = array / np.linalg.norm(array, axis=1)[:, np.newaxis]
        return normal

    def random_direction(self, normal):

        spread = self.param['spread']
        dir_local  = xicsrt_spread.vector_distribution(
            spread,
            self.param['intensity'],
            name=self.param['angular_dist'],
            )

        # Generate some basis vectors that are perpendicular
        # to the normal. The orientation does not matter here.
        o_1  = np.cross(normal, np.array([0,0,1])) + np.cross(normal, np.array([0,1,0]))
        o_1 /=  np.linalg.norm(o_1, axis=1)[:, np.newaxis]
        o_2  = np.cross(normal, o_1)
        o_2 /=  np.linalg.norm(o_2, axis=1)[:, np.newaxis]

        R        = np.empty((self.param['intensity'], 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal
        
        direction = np.einsum('ij,ijk->ik', dir_local, R)
        return direction

    def generate_wavelength(self, direction):
        wtype = str.lower(self.param['wavelength_dist'])
        if wtype == 'monochrome':
            wavelength  = np.ones(self.param['intensity'], dtype = np.float64)
            wavelength *= self.param['wavelength']
        elif wtype == 'uniform':
            wavelength = np.random.uniform(
                self.param['wavelength_range'][0]
                ,self.param['wavelength_range'][1]
                ,self.param['intensity']
                )
        elif wtype == 'voigt':
            #random_wavelength = self.random_wavelength_normal
            #random_wavelength = self.random_wavelength_cauchy
            random_wavelength = self.random_wavelength_voigt
            wavelength = random_wavelength(self.param['intensity'])
            
            #doppler shift
            c = const.physical_constants['speed of light in vacuum'][0]
            wavelength *= 1 - (np.einsum('j,ij->i', self.param['velocity'], direction) / c)
        else:
            raise Exception(f'Wavelength distribution {wtype} unknown')
        
        return wavelength

    def random_wavelength_voigt(self, size=None):
        #Units: wavelength (angstroms), natural_linewith (1/s), temperature (eV)
        
        # Check for the trivial case.
        if (self.param['linewidth']  == 0.0 and self.param['temperature'] == 0.0):
            return np.ones(size)*self.param['wavelength']
        # Check for the Lorentzian case.
        if (self.param['temperature'] == 0.0):
            # I need to update the cauchy routine first.
            #raise NotImplementedError('Random Lorentzian distribution not implemented.')

            # TEMPORARY:
            # The raytracer cannot currently accept a zero temperature, so just add 1eV for now.
            self.param['temperature'] += 1.0
             
        # Check for the Gaussian case.
        if (self.param['linewidth']  == 0.0):
            return self.random_wavelength_normal(size)

        c = const.physical_constants['speed of light in vacuum'][0]
        amu_kg = const.physical_constants['atomic mass unit-kilogram relationship'][0]
        ev_J = const.physical_constants['electron volt-joule relationship'][0]
        
        # Natural line width.
        gamma = (self.param['linewidth'] * self.param['wavelength']**2 / (4 * np.pi * c * 1e10))

        # Doppler broadened line width.
        sigma = (np.sqrt(self.param['temperature'] / self.param['mass_number'] / amu_kg / c**2 * ev_J)
                  * self.param['wavelength'] )

        rand_wave  = xicsrt_voigt.voigt_random(gamma, sigma, size)
        rand_wave += self.param['wavelength']
        return rand_wave

    def random_wavelength_normal(self, size=None):
        #Units: wavelength (angstroms), temperature (eV)
        c       = const.physical_constants['speed of light in vacuum'][0]
        amu_kg  = const.physical_constants['atomic mass unit-kilogram relationship'][0]
        ev_J    = const.physical_constants['electron volt-joule relationship'][0]
        
        # Doppler broadened line width.
        sigma = ( np.sqrt(self.param['temperature'] / self.param['mass_number'] / amu_kg / c**2 * ev_J)
                  * self.param['wavelength'] )

        rand_wave = np.random.normal(self.param['wavelength'], sigma, size)
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
        fwhm = self.param['linewidth']
        rand_wave = cauchy.rvs(loc=self.param['wavelength'], scale=fwhm, size=size)
        return rand_wave
    
    def generate_weight(self):
        # Weight is not currently used within XICSRT but might be useful
        # in the future.
        w = np.ones((self.param['intensity']), dtype=np.float64)
        return w
    
    def generate_mask(self):
        m = np.ones((self.param['intensity']), dtype=np.bool)
        return m
    
    def ray_filter(self, rays):
        for filter in self.filter_objects:
            rays = filter.filter(rays)
        return rays
