# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import numpy as np

from xicsrt.tools.xicsrt_doc import _dochelper_config
from xicsrt.optics._XicsrtOpticMesh import XicsrtOpticMesh

@_dochelper_config
class XicsrtOpticCrystal(XicsrtOpticMesh):

    def default_config(self):

        config = super().default_config()
        
        # xray optical information and polarization information
        config['crystal_spacing'] = 0.0
        config['reflectivity']    = 1.0

        config['do_bragg_check']     = True
        config['rocking_type']       = 'gaussian'
        config['rocking_fwhm']       = None
        config['rocking_sigma_file'] = None
        config['rocking_pi_file']    = None
        config['rocking_mix']        = 0.5 

        return config
    
    def initialize(self):
        super().initialize()
        self.param['rocking_type'] = str.lower(self.param['rocking_type'])
        
    def rocking_curve_filter(self, incident_angle, bragg_angle):

        # Generate or load a probability curve.
        if "step" in self.param['rocking_type']:
            # Step Function
            p = np.where(abs(incident_angle - bragg_angle) <= self.param['rocking_fwhm'] / 2,
                         self.param['reflectivity'], 0)
            
        elif "gauss" in self.param['rocking_type']:
            # Convert from FWHM to sigma.
            sigma = self.param['rocking_fwhm'] / (2 * np.sqrt(2 * np.log(2)))
            
            # Gaussian Distribution
            p = np.exp(-np.power(incident_angle - bragg_angle, 2.) / (2 * sigma**2))  
            p *= self.param['reflectivity']

        elif "file" in self.param['rocking_type']:
            # read datafiles and extract points
            sigma_data  = np.loadtxt(self.param['rocking_sigma_file'], dtype = np.float64)
            pi_data     = np.loadtxt(self.param['rocking_pi_file'], dtype = np.float64)
            
            # convert data from arcsec to rad
            sigma_data[:,0] *= np.pi / (180 * 3600)
            pi_data[:,0]    *= np.pi / (180 * 3600)

            # Convert from microrad to rad
            #sigma_data[:, 0] *= 1e-6
            #pi_data[:, 0] *= 1e-6

            # evaluate rocking curve reflectivity value for each incident ray
            sigma_curve = np.zeros(len(incident_angle), dtype = np.float64)
            pi_curve    = np.zeros(len(incident_angle), dtype = np.float64)
            
            sigma_curve = np.interp((incident_angle - bragg_angle), 
                                    sigma_data[:,0], sigma_data[:,1], 
                                    left = 0.0, right = 0.0)
            
            pi_curve    = np.interp((incident_angle - bragg_angle), 
                                    pi_data[:,0], pi_data[:,1], 
                                    left = 0.0, right = 0.0)
            
            p = self.param['rocking_mix'] * sigma_curve + (1 - self.param['rocking_mix']) * pi_curve
            
        else:
            raise Exception('Rocking curve type not understood: {}'.format(self.param['rocking_type']))

        # DEBUGGING:
        #   Plot the rocking curve.
        #
        # from mirutil.plot import mirplot
        # diff = (incident_angle - bragg_angle)
        # idx = np.argsort(diff)
        # plotlist = mirplot.PlotList()
        # plotlist.append({
        #     'x': diff[idx] * 1e6
        #     , 'y': p[idx]
        #     , 'xbound': [-5000, 5000]
        # })
        # plotlist.plotToScreen()

        # Rreate a random number for each ray between 0 and 1.
        test = np.random.uniform(0.0, 1.0, len(incident_angle))

        # compare that random number with the reflectivity distribution
        # and filter out the rays that do not survive the random
        # rocking curve filter.
        mask = (p >= test)
        
        return mask
    
    def angle_check(self, rays, normals, mask=None):
        if mask is None:
            mask = rays['mask']
        D = rays['direction']
        W = rays['wavelength']
        m = mask
        
        bragg_angle = np.zeros(m.shape, dtype=np.float64)
        dot = np.zeros(m.shape, dtype=np.float64)
        incident_angle = np.zeros(m.shape, dtype=np.float64)
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the optic
        bragg_angle[m] = np.arcsin( W[m] / (2 * self.param['crystal_spacing']))
        dot[m] = np.abs(np.einsum('ij,ij->i',D[m], -1 * normals[m]))
        incident_angle[m] = (np.pi / 2) - np.arccos(dot[m] / self.norm(D[m]))

        #check which rays satisfy bragg, update mask to remove those that don't
        if self.param['do_bragg_check'] is True:
            m[m] &= self.rocking_curve_filter(bragg_angle[m], incident_angle[m])

        return rays, normals
    
    def reflect_vectors(self, X, rays, normals, mask=None):
        if mask is None:
            mask = rays['mask']
        O = rays['origin']
        D = rays['direction']
        m = mask
        
        # Check which vectors meet the Bragg condition (with rocking curve)
        rays, normals = self.angle_check(rays, normals, m)
        
        # Perform reflection around normal vector, creating new rays with new
        # origin O = X and new direction D
        O[:]  = X[:]
        D[m] -= 2 * (np.einsum('ij,ij->i', D[m], normals[m])[:, np.newaxis]
                     * normals[m])
        
        return rays
