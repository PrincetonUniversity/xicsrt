# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
"""
import numpy as np
from scipy.spatial import Delaunay

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_math
from xicsrt.tools.xicsrt_math import *
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

import jax
import jax.numpy as jnp

class XicsrtOpticVariableRadiiToroid(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size'] = 0.001
        config['mesh_coarse_size'] = 0.005
        config['range_alpha'] = [-0.01, 0.01]
        config['range_beta']  = [-0.04, 0.04]
        config['major_radius'] = 1.0
        config['minor_radius'] = None

        # Temporary for development.
        config['normal_method'] = 'jax'

        return config

    def setup(self):
        super().setup()

        self.log.debug('Yo mama is here.')

        # Generate the fine mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_crystal_mesh(self.param['mesh_size'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_crystal_mesh(self.param['mesh_coarse_size'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points
        self.param['mesh_coarse_normals'] = mesh_normals
        self.log.debug(f'Coarse mesh points: {mesh_points.shape[0]}')

        # Calculate height at width.
        self.param['width'] = np.max(
            self.param['mesh_points'][:,0])-np.min(self.param['mesh_points'][:,0])
        self.param['height'] = np.max(
            self.param['mesh_points'][:,1])-np.min(self.param['mesh_points'][:,1])

    #def _mesh_precalc(self, points, faces):
    #    output = super()._mesh_precalc(points, faces)

    def vr_toroid_xyz_jax(self, alpha, beta, input):
        #maj_r = input['maj_r']
        #C0 = input['C0']
        #C0_xaxis = input['C0_xaxis']
        #C0_zaxis = input['C0_zaxis']
        #S = input['S']
        C0 = input['crystal_location']
        C0_zaxis  = input['crystal_zaxis']
        C0_xaxis  = input['crystal_xaxis']
        S        = input['source_location']
        maj_r    = input['major_radius']

        # The origin vectors could be pre-calculated.
        O_yaxis = jnp.cross(-1 * C0_zaxis, C0_xaxis)
        Ov1 = -1 * C0_zaxis
        Ov2 = C0_xaxis
        O = C0 - maj_r * Ov1

        Cv1 = Ov1 * jnp.cos(alpha) + Ov2 * jnp.sin(alpha)
        Cr2 = jnp.cross(O_yaxis, Cv1)
        C = O + maj_r * Cv1

        CS = C - S
        CS_dist = jnp.linalg.norm(CS)
        CS_hat = CS / CS_dist

        bragg = jnp.arccos(jnp.dot(CS_hat, -1 * Cr2))
        # sin_bragg = jnp.linalg.norm(jnp.cross(CS_hat, -1*Cr2))
        # cos_bragg = jnp.dot(CS_hat, -1*Cr2)

        CD_dist = maj_r * jnp.sin(bragg)
        CD_hat = CS_hat + 2 * Cr2 * jnp.cos(bragg)
        D = C - CD_dist * CD_hat

        SD = S - D
        SD_dist = jnp.linalg.norm(SD)
        SD_hat = SD / SD_dist

        sin_b = jnp.linalg.norm(jnp.cross(CS_hat, SD_hat))
        CP_dist = CS_dist * sin_b
        CP_hat = jnp.cross(O_yaxis, SD_hat)
        P = C - CP_dist * CP_hat

        Pv1 = CP_hat
        Pv2 = O_yaxis
        PX_hat = Pv1 * jnp.cos(beta) + Pv2 * jnp.sin(beta)
        X = P + CP_dist * PX_hat

        return X

    def vr_toroid_jax(self, alpha, beta, input):
        xyz = self.vr_toroid_xyz_jax(alpha, beta, input)
        profiler.start('vr_toroid_jax, normal')
        # This code can be vectorized.
        dfda, dfdb = jax.jacfwd(self.vr_toroid_xyz_jax, (0, 1))(alpha, beta, input)
        norm_ad = np.cross(dfda, dfdb)
        norm_ad /= np.linalg.norm(norm_ad)
        profiler.stop('vr_toroid_jax, normal')
        return xyz, norm_ad

    def vr_toroid_calculate(self, a, b, input):
        C_center = input['crystal_center']
        C_zaxis  = input['crystal_zaxis']
        C_yaxis  = input['crystal_yaxis']
        S        = input['source_location']
        maj_r    = input['major_radius']

        C_hat = vector_rotate(C_zaxis, C_yaxis, a)
        C = C_center - maj_r * C_hat

        SC = C - S
        SC_dist = np.linalg.norm(SC)
        SC_hat = SC / np.linalg.norm(SC)

        bragg = np.abs(np.pi / 2 - vector_angle(SC_hat, C_hat))

        D_dist = maj_r * np.cos(bragg)
        D = C_center + D_dist * vector_rotate(-1 * C_hat, C_yaxis, bragg)

        SD = D - S
        SD_hat = SD / np.linalg.norm(SD)

        PC_hat = np.cross(SD, C_yaxis)
        PC_hat /= np.linalg.norm(PC_hat)

        eta = vector_angle(SD_hat, SC_hat)
        PC_dist = SC_dist * np.sin(eta)

        P = C + (-1) * PC_hat * PC_dist
        PC_hat = vector_rotate(PC_hat, SD_hat, b)

        xyz = P + PC_hat * PC_dist
        norm = vector_rotate(C_hat, SD_hat, b)

        return xyz, norm

    def vr_toroid_calculate_fd(self, a, b, input, delta=None):
        profiler.start('finite difference')
        if delta is None: delta = 1e-4

        xyz, norm = self.vr_toroid_calculate(a, b, input)
        xyz1, norm1 = self.vr_toroid_calculate(a + delta, b, input)
        xyz2, norm2 = self.vr_toroid_calculate(a, b + delta, input)

        vec1 = xyz1 - xyz
        vec2 = xyz2 - xyz
        norm_fd = np.cross(vec1, vec2)
        norm_fd /= np.linalg.norm(norm_fd)
        profiler.stop('finite difference')
        return xyz, norm_fd

    def generate_crystal_mesh(self, mesh_size=None):
        """
        This method creates the meshgrid for the crystal
        """
        if mesh_size is None: mesh_size = self.param['mesh_size']

        crystal_name = 'Ge (400)'

        major_radius = self.param['major_radius']
        minor_radius = self.param['minor_radius']

        crystal_spacing = 2.82868 / 2
        crystal_location = np.array([0.0, 0.0, 0.0])
        crystal_zaxis = np.array([0.0, 0.0, -1.0])
        crystal_xaxis = np.array([1.0, 0.0, 0.0])
        crystal_yaxis = np.cross(crystal_xaxis, crystal_zaxis)

        # Corresponds to an energy of 9.75 keV
        wavelength = 1.2716
        source_distance = 0.3

        # Start calculations.
        crystal_center = crystal_location + crystal_zaxis * major_radius

        # Calculate the source location for Manfred's setup.
        bragg_angle = xicsrt_math.bragg_angle(wavelength, crystal_spacing)
        self.log.debug(f'crystal    : {crystal_name}')
        self.log.debug(f'wavelength : {wavelength:0.4f} Å')
        self.log.debug(f'bragg_angle: {np.degrees(bragg_angle):0.3f}°')
        source_location = (
            crystal_location
            + source_distance * vector_rotate(
                crystal_zaxis
                ,crystal_yaxis
                ,(np.pi / 2 - bragg_angle))
            )

        input = {}
        input['crystal_location'] = crystal_location
        input['crystal_center'] = crystal_center
        input['crystal_xaxis'] = crystal_xaxis
        input['crystal_yaxis'] = crystal_yaxis
        input['crystal_zaxis'] = crystal_zaxis
        input['source_location'] = source_location
        input['major_radius'] = major_radius

        # --------------------------------
        # Setup the basic grid parameters.

        a_range = self.param['range_alpha']
        b_range = self.param['range_beta']
        a_span = a_range[1] - a_range[0]
        b_span = b_range[1] - b_range[0]

        # For now assume the mesh_size is the angular density
        # in the a direction.
        num_a = a_span/mesh_size
        num_b = num_a / (major_radius * a_span) * (minor_radius * b_span)
        num_a = np.ceil(num_a).astype(int)
        num_b = np.ceil(num_b).astype(int)


        self.log.debug(f'mesh_size: {mesh_size} rad, total: {num_a*num_b}')
        self.log.debug(f'num_a, num_b: {num_a}, {num_b}')

        alpha = np.linspace(a_range[0], a_range[1], num_a)
        beta = np.linspace(b_range[0], b_range[1], num_b)

        aa, bb = np.meshgrid(alpha, beta, indexing='ij')
        xx = np.empty((num_a, num_b))
        yy = np.empty((num_a, num_b))
        zz = np.empty((num_a, num_b))

        normal_xx = np.empty((num_a, num_b))
        normal_yy = np.empty((num_a, num_b))
        normal_zz = np.empty((num_a, num_b))

        # ------------------------------------------------
        # Now calculate the xyz values at each grid point.
        #
        # This is my first attempt at the geometry; I have made no attempt
        # at all to do this efficently, and I am probably caculating a lot
        # of things that are not needed.
        for ii_a in range(num_a):
            for ii_b in range(num_b):

                a = aa[ii_a, ii_b]
                b = bb[ii_a, ii_b]

                # Temporary for development.
                if self.param['normal_method'] == 'jax':
                    xyz, norm = self.vr_toroid_jax(a, b, input)
                elif self.param['normal_method'] == 'fd':
                    xyz, norm = self.vr_toroid_calculate_fd(a, b, input)
                elif self.param['normal_method'] == 'ideal_np':
                    xyz, norm = self.vr_toroid_calculate(a, b, input)
                else:
                    raise Exception(f"normal_method {self.param['normal_method']} unknown.")

                xx[ii_a, ii_b] = xyz[0]
                yy[ii_a, ii_b] = xyz[1]
                zz[ii_a, ii_b] = xyz[2]
                normal_xx[ii_a, ii_b] = norm[0]
                normal_yy[ii_a, ii_b] = norm[1]
                normal_zz[ii_a, ii_b] = norm[2]

        # Combine x & y arrays, add z dimension
        # points_2d = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        # tri = Delaunay(points_2d)
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T
        tri = Delaunay(angles_2d)


        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        normals = np.stack((normal_xx.flatten(), normal_yy.flatten(), normal_zz.flatten())).T
        faces = tri.simplices

        return points, normals, faces

