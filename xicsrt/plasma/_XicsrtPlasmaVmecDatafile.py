# -*- coding: utf-8 -*-
"""
Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
A plasma source based on a VMEC equilibrium.
"""

import numpy as np

from xicsrt.plasma._XicsrtPlasmaVmec import XicsrtPlasmaVmec
from xicsrt.util import profiler

import stelltools

class XicsrtPlasmaVmecDatafile(XicsrtPlasmaVmec):
    
    def __init__(self, config):
        super().__init__(config)

        self.use_profiles   = config['use_profiles']

        self.temperature_file = config['temperature_file']
        self.emissivity_file  = config['emissivity_file']
        self.velocity_file    = config['velocity_file']
    
    def get_emissivity(self, rho):
        output = np.zeros(len(rho))
        if self.use_profiles is True:
            # Read and interpolate profile from data file
            data  = np.loadtxt(self.emissivity_file, dtype = np.float64)           
            output[:]  = np.interp(rho, data[:,0], data[:,1], left=0.0, right=0.0)
        else:
            output[:] = self.emissivity

        return output

    def get_temperature(self, rho):
        output = np.zeros(len(rho))
        if self.use_profiles is True:
            # Read and interpolate profile from data file
            data  = np.loadtxt(self.temperature_file, dtype = np.float64)           
            output[:]  = np.interp(rho, data[:,0], data[:,1], left=0.0, right=0.0)
        else:
            output[:] = self.temperature

        return output

    def get_velocity(self, rho):
        output = np.zeros((len(rho),3))
        if self.use_profiles is True:
            # Read and interpolate profile from data file
            data  = np.loadtxt(self.velocity_file, dtype = np.float64)           
            output[:, 0]  = np.interp(rho, data[:,0], data[:,1], left=0.0, right=0.0)           
            output[:, 1]  = np.interp(rho, data[:,0], data[:,2], left=0.0, right=0.0)           
            output[:, 2]  = np.interp(rho, data[:,0], data[:,3], left=0.0, right=0.0)
        else:
            output[:] = self.velocity

        return output
    
