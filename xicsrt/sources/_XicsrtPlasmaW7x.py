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
import logging

from xicsrt.sources.xics_rt_vmec import FluxSurfacePlasma

from mirutil import hdf5

from mirutil.classes import MirSignalObject
from mirfusion.xics.analysis.config._XcSystemPaths import XcSystemPaths


class W7xPlasma(FluxSurfacePlasma):
    def __init__(self, config):
        super().__init__(config)

    def getValue(self, rho_req, name):
        ii_time = np.argmin(np.abs(self.options['time'] - self.data['dim']['time']))

        rho = self.data['dim']['rho']
        value = self.data['value'][name][ii_time,:]

        value_out = np.interp(rho_req, rho, value)
        return value_out

    def getEmissivity(self, rho_req):
        return self.getValue(rho_req, 'w_emissivity')

    def getTemperature(self, rho_req):
        return self.getValue(rho_req, 'T_ion_Ar')

    def loadPlasmaProfiles(self):

        # For now just hard code all the xics options.
        options = {}
        options['system'] = 'w7x_ar16'
        options['shot'] = 171207006
        options['setting'] = 'xics_100ms_3cm'
        options['run_xics'] = 6
        options['time'] = 2.2

        self.options = options
        self.data = self.getXicsAnalysis(options)

    def getXicsAnalysis(self, options):
        """
        Get XICS data from a savefile.
        This should eventually use a XICS or stelltran data object, but for now
        this is the fastest way to get this running.
        """
        obj_path = XcSystemPaths()
        filepath = obj_path.getFilepathXicsInversion(
            options['system']
            , options['shot']
            , options['setting']
            , options['run_xics'])
        logging.info('Reading data from: {}'.format(filepath))
        data_raw = hdf5.hdf5ToDict(filepath)
        data = MirSignalObject(data_raw)

        data['coord']['time'] = np.expand_dims(data['dim']['time'], 1)
        data['coord']['rho'] = np.expand_dims(data['dim']['rho'], 0)

        return data

    def generate_rays(self):
        self.loadPlasmaProfiles()
        rays = super().generate_rays()
        return rays
