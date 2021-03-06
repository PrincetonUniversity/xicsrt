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

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.sources._XicsrtPlasmaVmec import XicsrtPlasmaVmec
from xicsrt.util import profiler

@dochelper
class XicsrtPlasmaVmecDatafile(XicsrtPlasmaVmec):

    def default_config(self):
        config = super().default_config()
        config['emissivity_file']  = None
        config['temperature_file'] = None
        config['velocity_file']    = None
        return config
        
    def get_emissivity(self, rho):
        output = np.zeros(len(rho))
        # Read and interpolate profile from data file
        data  = np.loadtxt(self.param['emissivity_file'], dtype = np.float64)           
        output[:]  = np.interp(rho, data[:,0], data[:,1], left=0.0, right=0.0)
        
        return output

    def get_temperature(self, rho):
        output = np.zeros(len(rho))
        
        # Read and interpolate profile from data file
        data  = np.loadtxt(self.param['temperature_file'], dtype = np.float64)           
        output[:]  = np.interp(rho, data[:,0], data[:,1], left=0.0, right=0.0)
        
        return output

    # def get_velocity(self, rho):
    #     output = np.zeros((len(rho),3))
    #
    #     # Read and interpolate profile from data file
    #     data  = np.loadtxt(self.param['velocity_file'], dtype = np.float64)
    #     output[:, 0]  = np.interp(rho, data[:,0], data[:,1], left=0.0, right=0.0)
    #     output[:, 1]  = np.interp(rho, data[:,0], data[:,2], left=0.0, right=0.0)
    #     output[:, 2]  = np.interp(rho, data[:,0], data[:,3], left=0.0, right=0.0)
    #
    #     return output

