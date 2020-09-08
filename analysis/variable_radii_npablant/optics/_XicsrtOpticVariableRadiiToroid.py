# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
"""
import numpy as np
from scipy.spatial import Delaunay

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


class XicsrtOpticVariableRadiiToroid(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size_a'] = 0.005
        config['mesh_size_b'] = 0.01
        config['mesh_coarse_size_a'] = 0.01
        config['mesh_coarse_size_b'] = 0.02
        config['range_a'] = [-0.01, 0.01]
        config['range_b'] = [-0.04, 0.04]

        # Parameters needed to define geometry.
        config['major_radius'] = 1.0
        config['minor_radius'] = None
        config['crystal_spacing'] = None
        config['lambda0'] = None
        config['source_distance'] = 0.3

        # Calculation options.
        config['normal_method'] = 'jax'

        return config

    def setup(self):
        super().setup()

        self.log.debug('Yo mama is here.')

        # Generate the fine mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_crystal_mesh(
            self.param['mesh_size_a'], self.param['mesh_size_b'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_crystal_mesh(
            self.param['mesh_coarse_size_a'], self.param['mesh_coarse_size_b'])
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
        self.log.debug(f"WxH: {self.param['width']:0.3f}x{self.param['height']:0.3f}")

    def vr_toroid(self, a, b, param, extra=False):
        """
        Calculate the parameters of the variable-radii toroid.

        This calculation is the same as vr_toroid_1, but uses a method of
        calculation that is easier to read and understand. It may be slightly
        slower.
        """
        C0       = param['crystal_location']
        C0_zaxis = param['crystal_zaxis']
        C0_xaxis = param['crystal_xaxis']
        S        = param['source_location']
        maj_r    = param['major_radius']

        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)
        O = C0 + maj_r * C0_zaxis

        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = O - maj_r * C_norm

        SC = C - S
        SC_dist = np.linalg.norm(SC)
        SC_hat = SC / SC_dist

        bragg = np.abs(np.pi / 2 - xm.vector_angle(SC_hat, C_norm))

        D_dist = maj_r * np.cos(bragg)
        D = O + D_dist * xm.vector_rotate(-1*C_norm, C0_yaxis, bragg)

        SD = D - S
        SD_hat = SD / np.linalg.norm(SD)

        PC_hat = np.cross(SD, C0_yaxis)
        PC_hat /= np.linalg.norm(PC_hat)

        aDSC = xm.vector_angle(SD_hat, SC_hat)
        PC_dist = SC_dist * np.sin(aDSC)
        P = C + (-1) * PC_hat * PC_dist

        PX_hat = xm.vector_rotate(PC_hat, SD_hat, b)
        X = P + PX_hat * PC_dist

        QC_hat = -1 * C_norm
        QCP_angle = xm.vector_angle(QC_hat, PC_hat)
        QC_dist = PC_dist / np.cos(QCP_angle)
        Q = C - QC_hat * QC_dist

        if extra:
            out = {}
            out['C0'] = C0
            out['C'] = C
            out['O'] = O
            out['S'] = S
            out['D'] = D
            out['P'] = P
            out['Q'] = Q
            out['X'] = X
            return out
        else:
            return X

    def vr_toroid_1(self, alpha, beta, param, extra=False):
        """
        Calculate the parameters of the variable-radii toroid.

        This calculation is the same as vr_toroid, but uses a different
        formulation. This formulation should be a little faster. In addition
        this version uses the JAX version of numpy so that JAX functions can
        be used.
        """
        C0       = param['crystal_location']
        C0_zaxis = param['crystal_zaxis']
        C0_xaxis = param['crystal_xaxis']
        S        = param['source_location']
        maj_r    = param['major_radius']

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

        sin_aDSC = jnp.linalg.norm(jnp.cross(CS_hat, SD_hat))
        CP_dist = CS_dist * sin_aDSC
        CP_hat = jnp.cross(O_yaxis, SD_hat)
        P = C - CP_dist * CP_hat

        Pv1 = CP_hat
        Pv2 = O_yaxis
        XP_hat = Pv1 * jnp.cos(beta) + Pv2 * jnp.sin(beta)
        X = P + CP_dist * XP_hat

        if extra:
            Qv1 = XP_hat
            Qv2 = SD_hat
            aQCP = jnp.arccos(np.dot(Cv1, Pv1))
            CQ_dist = CP_dist / jnp.cos(aQCP)
            XQ_hat = Qv1 * jnp.cos(aQCP) + Qv2 * jnp.sin(aQCP)
            Q = X - XQ_hat * CQ_dist

            out = {}
            out['C0'] = C0
            out['C'] = C
            out['O'] = O
            out['S'] = S
            out['D'] = D
            out['P'] = P
            out['Q'] = Q
            out['X'] = X
            return out
        else:
            return X

    def vr_toroid_jax(self, alpha, beta, param):
        xyz = self.vr_toroid_1(alpha, beta, param)
        profiler.start('vr_toroid_jax, normal')
        dfda, dfdb = jax.jacfwd(self.vr_toroid_1, (0, 1))(alpha, beta, param)
        norm_ad = np.cross(dfda, dfdb)
        norm_ad /= np.linalg.norm(norm_ad)
        profiler.stop('vr_toroid_jax, normal')
        return xyz, norm_ad

    def vr_toroid_fd(self, a, b, param, delta=None):
        profiler.start('vr_toroid_fd, finite difference')
        if delta is None: delta = 1e-4

        xyz = self.vr_toroid(a, b, param)
        xyz1 = self.vr_toroid(a + delta, b, param)
        xyz2 = self.vr_toroid(a, b + delta, param)

        vec1 = xyz1 - xyz
        vec2 = xyz2 - xyz
        norm_fd = np.cross(vec1, vec2)
        norm_fd /= np.linalg.norm(norm_fd)
        profiler.stop('vr_toroid_fd, finite difference')
        return xyz, norm_fd

    def vr_toroid_ideal_np(self, a, b, param, delta=None):
        profiler.start('vr_toroid_ideal_np')
        out = self.vr_toroid(a, b, param, extra=True)
        xyz = out['X']
        norm = (out['Q'] - out['X']) / np.linalg.norm(out['Q'] - out['X'])
        profiler.stop('vr_toroid_ideal_np')
        return xyz, norm

    def get_vrt_param(self):
        major_radius = self.param['major_radius']
        minor_radius = self.param['minor_radius']

        crystal_spacing = self.param['crystal_spacing']
        crystal_location = self.param['origin']
        crystal_zaxis = self.param['zaxis']
        crystal_xaxis = self.param['xaxis']
        crystal_yaxis = np.cross(crystal_xaxis, crystal_zaxis)

        lambda0 = self.param['lambda0']
        source_distance = self.param['source_distance']

        # Calculate the source location.
        bragg_angle = xm.bragg_angle(lambda0, crystal_spacing)
        self.log.debug(f'lambda0 : {lambda0:0.4f} Å')
        self.log.debug(f'bragg_angle: {np.degrees(bragg_angle):0.3f}°')
        source_location = (
            crystal_location
            + source_distance
            * xm.vector_rotate(
                crystal_zaxis
                ,crystal_yaxis
                ,(np.pi / 2 - bragg_angle))
            )

        param = {}
        param['crystal_location'] = crystal_location
        param['crystal_xaxis']    = crystal_xaxis
        param['crystal_zaxis']    = crystal_zaxis
        param['source_location']  = source_location
        param['major_radius']     = major_radius
        param['minor_radius']     = minor_radius

        return param

    def generate_crystal_mesh(self, mesh_size_a=None, mesh_size_b=None):
        """
        This method creates the meshgrid for the crystal
        """
        profiler.start('generate_crystal_mesh')
        major_radius = self.param['major_radius']
        minor_radius = self.param['minor_radius']

        vrt_param = self.get_vrt_param()

        # --------------------------------
        # Setup the basic grid parameters.

        a_range = self.param['range_a']
        b_range = self.param['range_b']
        a_span = a_range[1] - a_range[0]
        b_span = b_range[1] - b_range[0]

        # For now assume the mesh_size is in radians.
        num_a = a_span/mesh_size_a
        num_b = b_span/mesh_size_b
        num_a = np.ceil(num_a).astype(int)
        num_b = np.ceil(num_b).astype(int)

        self.log.debug(f'num_a, num_b: {num_a}, {num_b}, total: {num_a*num_b}')

        a = np.linspace(a_range[0], a_range[1], num_a)
        b = np.linspace(b_range[0], b_range[1], num_b)

        aa, bb = np.meshgrid(a, b, indexing='ij')
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
                    xyz, norm = self.vr_toroid_jax(a, b, vrt_param)
                elif self.param['normal_method'] == 'fd':
                    xyz, norm = self.vr_toroid_fd(a, b, vrt_param)
                elif self.param['normal_method'] == 'ideal_np':
                    xyz, norm = self.vr_toroid_ideal_np(a, b, vrt_param)
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

        profiler.stop('generate_crystal_mesh')
        return points, normals, faces

