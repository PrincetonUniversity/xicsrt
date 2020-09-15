# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
"""

import numpy as np
import logging
from copy import deepcopy

from scipy import interpolate
from scipy import constants as const

from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import sinusoidal_spiral

from xicsrt.objects._XicsrtDispatcher import XicsrtDispatcher

def wave_from_energy(energy):
    wave = const.h * const.c / energy / const.e * 1e10
    return wave


def _ray_intersect(ray1, ray2):
    dx = ray2['point'][0] - ray1['point'][0]
    dz = ray2['point'][2] - ray1['point'][2]
    det = ray2['vector'][0] * ray1['vector'][2] - ray2['vector'][2] * ray1['vector'][0]
    u = (dz * ray2['vector'][0] - dx * ray2['vector'][2]) / det
    p = ray1['point'] + u * ray1['vector']
    return p


def _setup_scenario(config):
    lambdaC = wave_from_energy(config['scenario']['energyC'])
    thetaC = xm.bragg_angle(lambdaC, config['optics']['crystal']['crystal_spacing'])
    logging.debug(f"central lambda: {lambdaC:0.4f} Å")
    logging.debug(f"central bragg_angle: {np.degrees(thetaC):0.3f}°")

    config['scenario']['lambdaC'] = lambdaC
    config['scenario']['thetaC'] = thetaC


def _setup_source(config, inp, func):
    out = func(0.0, 0.0, inp, extra=True)

    config['sources']['source']['origin'] = out['S']
    SC = config['optics']['crystal']['origin'] - config['sources']['source']['origin']
    S_zaxis = SC / np.linalg.norm(SC)
    S_xaxis = np.cross(S_zaxis, np.array([0, 1, 0]))
    S_xaxis /= np.linalg.norm(S_xaxis)
    config['sources']['source']['zaxis'] = S_zaxis
    config['sources']['source']['xaxis'] = S_xaxis
    config['sources']['source']['target'] = config['optics']['crystal']['origin']

    lambda_source = wave_from_energy(config['scenario']['energy'])
    config['sources']['source']['wavelength'] = lambda_source


def _setup_detector(config, inp, func):
    """
    Setup the Detector location and dispersion.
    """
    num_aa = 11
    aa = np.linspace(
        config['optics']['crystal']['range_a'][0]
        , config['optics']['crystal']['range_a'][1]
        , num_aa)
    out_list = []
    for a in aa:
        out_list.append(func(a, 0.0, inp, extra=True))

    # Calculate the detector location and orientation.
    #
    # A better solution might be found by actually fitting
    # the points to a line.
    ii_c = (num_aa - 1) // 2
    # D_origin   = (out_list[-1]['D'] + out_list[0]['D'])/2
    D_origin = out_list[ii_c]['D']
    D_xaxis = out_list[-1]['D'] - out_list[0]['D']
    D_xaxis /= np.linalg.norm(D_xaxis)
    D_yaxis = np.array([0, 1, 0])
    D_zaxis = np.cross(D_yaxis, D_xaxis)

    config['optics']['detector']['origin'] = D_origin
    config['optics']['detector']['zaxis'] = D_zaxis
    config['optics']['detector']['xaxis'] = D_xaxis

def _setup_dispersion(config, inp, func):
    """
    Setup interpolators for the dispersion.
    """

    num_aa = 11
    aa = np.linspace(
        config['optics']['crystal']['range_a'][0]
        , config['optics']['crystal']['range_a'][1]
        , num_aa)
    out_list = []
    for a in aa:
        out_list.append(func(a, 0.0, inp, extra=True))

    # Get a VariableRadiiToroid object.
    obj_optics = XicsrtDispatcher(config, 'optics')
    obj_optics.instantiate('detector')
    obj_detector = obj_optics.get_object('detector')
    obj_detector.setup()
    obj_detector.initialize()

    # For clarity only, copy to local variables.
    D_origin = config['optics']['detector']['origin']
    D_xaxis = config['optics']['detector']['xaxis']

    # Now calculate the dispersion on the detector.
    x_list = []
    energy_list = []
    wave_list = []
    for out in out_list:
        DC = out['C'] - out['D']
        SC = out['C'] - out['S']
        bragg = (np.pi - xm.vector_angle(DC, SC)) / 2
        wavelength = 2 * config['optics']['crystal']['crystal_spacing'] * np.sin(bragg)
        energy = const.h * const.c / wavelength / const.e * 1e10

        # Find the intersection of the central ray with the actual
        # detector position.  We cannot use out['D'] here because
        # the points 'D' fall on a curve, while the detector is a
        # a flat plane.
        ray1 = {'point': D_origin, 'vector': D_xaxis}
        ray2 = {'point': out['C'], 'vector': DC}
        D_int = _ray_intersect(ray1, ray2)
        D_int = obj_detector.point_to_local(D_int)

        x_list.append(D_int[0])
        energy_list.append(energy)
        wave_list.append(wavelength)

        logging.debug(f"energy: {energy / 1e3:7.4f} keV,  wave: {wavelength:0.4f} A,  x: {D_int[0] * 1e3:8.4f} mm")

    energy_w = energy_list[-1] - energy_list[1]
    detector_w = x_list[-1] - x_list[0]
    disp = energy_w/detector_w
    logging.debug(f"Detector width: {detector_w*1e3:0.0f} mm, Dispersion: {disp/1e3:0.2f} eV/mm")

    interp = {}
    interp['energy'] = interpolate.interp1d(x_list, energy_list, fill_value='extrapolate')
    interp['wavelength'] = interpolate.interp1d(x_list, wave_list, fill_value='extrapolate')

    return interp


def _get_info(config, inp, func, info=None):
    out = func(0.0, 0.0, inp, extra=True)

    if info is None:
        info = {}

    if 'O' in out:
        # Save the major radius
        CO = out['O'] - out['C']
        CO_dist = np.linalg.norm(CO)
        logging.info(f'major_radius CO: {CO_dist}')
        info['CO_dist'] = CO_dist

    if 'Q' in out:
        # Save the minor radius
        CQ = out['Q'] - out['C']
        CQ_dist = np.linalg.norm(CQ)
        info['CQ_dist'] = CQ_dist

    if 'P' in out:
        # Calculate the distance CP, this is useful in sizing the crystal.
        CP = out['P'] - out['C']
        CP_dist = np.linalg.norm(CP)
        logging.info(f'minor_radius CQ: {CQ_dist}, CP: {CP_dist}')
        info['CP_dist'] = CP_dist


    return info


def setup_vrt(config):
    output = {}

    config['optics']['crystal']['class_name'] = 'XicsrtOpticVariableRadiiToroid'

    _setup_scenario(config)

    config['optics']['crystal']['lambda0'] = config['scenario']['lambdaC']

    # Get a object for this crystal.
    obj_optics = XicsrtDispatcher(config, 'optics')
    obj_optics.instantiate('crystal')
    obj_crystal = obj_optics.get_object('crystal')

    func = obj_crystal.vr_toroid
    inp = obj_crystal.get_vrt_param()

    # Setup the Source location.
    _setup_source(config, inp, func)

    # Setup the Detector location.
    _setup_detector(config, inp, func)

    # Setup dispersion.
    interp = _setup_dispersion(config, inp, func)

    # Setup some extras.
    info = _get_info(config, inp, func)
    config['optics']['crystal']['minor_radius'] = info['CQ_dist']

    output['config'] = config
    output['interp'] = interp
    output['info'] = info

    return output


def setup_torus(config):
    output = {}

    config['optics']['crystal']['class_name'] = 'XicsrtOpticTorus'

    # Setup the basic configuration
    config['optics']['crystal']['minor_radius'] = config['scenario']['minor_radius']

    _setup_scenario(config)

    config['optics']['crystal']['lambda0'] = config['scenario']['lambdaC']

    # Get a object for this crystal.
    obj_optics = XicsrtDispatcher(config, 'optics')
    obj_optics.instantiate('crystal')
    obj_crystal = obj_optics.get_object('crystal')

    func = obj_crystal.torus
    inp = obj_crystal.get_t_param()

    # Setup the Source location.
    _setup_source(config, inp, func)

    # Setup the Detector location.
    _setup_detector(config, inp, func)

    # Setup dispersion.
    interp = _setup_dispersion(config, inp, func)

    # Setup some extras.
    info = _get_info(config, inp, func)

    output['config'] = config
    output['interp'] = interp
    output['info'] = info

    return output

def _get_spiral_crystal_default(config):
    out = {}
    out['r0'] = 0.2655462577585708
    out['b'] = 0.28154129802482347
    out['sC'] = 0.3
    out['phiC'] = 0.0
    out['range_a'] = [-0.023807150903836036, 0.06528972861198715]
    out['range_b'] = [-0.0792259496576749, 0.0792259496576749]
    out['theta0'] = config['scenario']['thetaC']
    out['thetaC'] = config['scenario']['thetaC']
    return out

def setup_spiral(config):

    output = {}
    config['optics']['crystal']['class_name'] = 'XicsrtOpticVariableRadiiSpiral'

    _setup_scenario(config)

    # Set defaults for the crystal parameters.
    c_default = _get_spiral_crystal_default(config)
    for key in c_default:
        if not key in config['optics']['crystal'] or config['optics']['crystal'][key] is None:
            config['optics']['crystal'][key] = c_default[key]

    # Get a object for this crystal.
    obj_optics = XicsrtDispatcher(config, 'optics')
    obj_optics.instantiate('crystal')
    obj_crystal = obj_optics.get_object('crystal')
    obj_crystal.setup_geometry()

    func = obj_crystal.spiral_centered
    inp = obj_crystal.get_spiral_inp()


    # ----------
    # Find the optimal value of 'b'.
    inp = sinusoidal_spiral.find_symmetry_b(inp)
    config['optics']['crystal']['b'] = inp['b']

    # Get a new Spiral object with the new value of b.
    obj_optics = XicsrtDispatcher(config, 'optics')
    obj_optics.instantiate('crystal')
    obj_crystal = obj_optics.get_object('crystal')
    obj_crystal.setup_geometry()

    inp = obj_crystal.get_spiral_inp()
    # ----------


    # Setup the Source location.
    _setup_source(config, inp, func)

    # Setup the Detector location.
    _setup_detector(config, inp, func)

    # Setup dispersion.
    interp = _setup_dispersion(config, inp, func)

    # Setup some extras.
    info = _get_info(config, inp, func)
    config['optics']['crystal']['minor_radius'] = info['CQ_dist']

    output['config'] = config
    output['interp'] = interp
    output['info'] = info

    return output