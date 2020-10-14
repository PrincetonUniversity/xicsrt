# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import numpy as np

from xicsrt.tools import bragg_reader
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticMesh import XicsrtOpticMesh

@dochelper
class XicsrtOpticCrystal(XicsrtOpticMesh):
    """
    .. SourceNote
        See documentation or use help(XicsrtOpticCrystal) for a complete list
        of config options.
    """

    def default_config(self):
        """
        crystal_spacing
          The spacing between crystal planes.

          .. Note::

                This is the nominal 'd' crystal spacing, not the '2d' spacing
                often used in the literature.

        reflectivity: float (1.0)
          A reflectivity factor for this optic. The reflectivity will modify
          the probability that a ray will reflect from this optic.

        do_bragg_check: bool (True)
          Switch between x-ray Bragg reflections and optical reflections for
          this optic. If True, a rocking curve will be used to determine the
          probability of reflection for rays based on their incident angle.

        rocking_type: str ('gaussian')
          The type of shape to use for the crystal rocking curve.
          Allowed types are 'step', 'gaussian' and 'file'.

        rocking_fwhm: float [rad]
          The width of the rocking curve, in radians.
          This option only used when rocking_type is 'step' or 'gaussian'.

        rocking_file: str or list
          A filename from which to read rocking curve data. A list may be used
          if sigma and pi data are in separate files.

        rocking_filetype: str
          The type of rocking curve file to be loaded.
          The following formats are currently supported: 'xop', 'x0h', 'simple'.

          .. Note::

                Actually at this point only 'xop' is supported. np 2020-10-13

        rocking_mix: float
          A mixing factor to combine the sigma and pi reflectivities.
          This value will be interpreted as sigma/pi and will mix the reflection
          probabilities linearly. ref = sigma*mix + pi*(1-mix)

        """

        config = super().default_config()
        
        # xray optical information and polarization information
        config['crystal_spacing'] = 0.0
        config['reflectivity']    = 1.0

        config['do_bragg_check']     = True
        config['rocking_type']       = 'gaussian'
        config['rocking_fwhm']       = None
        config['rocking_file']       = None
        config['rocking_filetype']   = None
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
                         1.0, 0.0)
            
        elif "gauss" in self.param['rocking_type']:
            # Convert from FWHM to sigma.
            sigma = self.param['rocking_fwhm'] / (2 * np.sqrt(2 * np.log(2)))
            
            # Gaussian Distribution
            p = np.exp(-np.power(incident_angle - bragg_angle, 2.) / (2 * sigma**2))

        elif "file" in self.param['rocking_type']:

            data = bragg_reader.read(self.param['rocking_file'], self.param['rocking_filetype'])

            units = data['units']['dtheta_in']
            if  units == 'urad':
                scale_theta = 1e-6
            elif units == 'arcset':
                scale_theta = np.pi / (180 * 3600)
            elif units =='rad':
                scale_theta = 1.0
            else:
                raise Exception(f'Units in rocking curve data not understood: {units}')

            # evaluate rocking curve reflectivity value for each incident ray
            # This calculation is done in units of 'radians'.
            dtheta = (incident_angle - bragg_angle)
            sigma = np.interp(
                dtheta
                ,data['value']['dtheta_in']*scale_theta
                ,data['value']['reflect_s']
                ,left = 0.0
                ,right = 0.0)
            pi = np.interp(
                dtheta
                ,data['value']['dtheta_in']*scale_theta
                ,data['value']['reflect_p']
                ,left = 0.0
                ,right = 0.0)
            
            p = self.param['rocking_mix'] * sigma + (1 - self.param['rocking_mix']) * pi
            
        else:
            raise Exception('Rocking curve type not understood: {}'.format(self.param['rocking_type']))

        # DEBUGGING:
        #   Plot the rocking curve.
        #
        #from mirutil.plot import mirplot
        #diff = (incident_angle - bragg_angle)
        #idx = np.argsort(diff)
        #plotlist = mirplot.PlotList()
        #plotlist.append({
        #     'x': diff[idx] * 1e6
        #     , 'y': p[idx]
        # })
        #plotlist.plotToScreen()

        p *= self.param['reflectivity']

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
        bragg_angle[m] = np.arcsin(W[m] / (2 * self.param['crystal_spacing']))
        dot[m] = np.abs(np.einsum('ij,ij->i',D[m], -1 * normals[m]))
        incident_angle[m] = (np.pi / 2) - np.arccos(dot[m] / self.norm(D[m]))

        #check which rays satisfy bragg, update mask to remove those that don't
        if self.param['do_bragg_check'] is True:
            m[m] &= self.rocking_curve_filter(incident_angle[m], bragg_angle[m])

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
