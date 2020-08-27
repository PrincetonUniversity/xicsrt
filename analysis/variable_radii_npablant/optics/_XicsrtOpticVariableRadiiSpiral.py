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
from xicsrt.objects._GeometryObject import GeometryObject
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

import jax
import jax.numpy as jnp
from xicsrt.tools import xicsrt_math_jax as xmj

class XicsrtOpticVariableRadiiSpiral(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size_a'] = 0.005
        config['mesh_size_b'] = 0.01
        config['mesh_coarse_size_a'] = 0.01
        config['mesh_coarse_size_b'] = 0.02
        config['range_a'] = [-0.01, 0.01]
        config['range_b']  = [-0.04, 0.04]

        config['r0'] = 1.0
        config['b'] = 0.356
        config['theta0'] = np.radians(25)

        # Temporary for development.
        config['normal_method'] = 'jax'

        return config

    def setup_geometry(self):
        """
        Setup a geometry object to allow transformation of the
        spiral geometry to be centered on the crystal with the
        crystal normal as the z-axis. This is what is usually
        used when generating raytracing scenarios.
        """
        inp = {}
        inp['r0'] = self.param['r0']
        inp['b'] = self.param['b']
        inp['theta0'] = self.param['theta0']

        out = self.spiral(0.0, 0.0, inp, dict=True)
        config = {}
        config['origin'] = out['X']
        config['zaxis'] = (out['Q'] - out['X'])/np.linalg.norm(out['Q'] - out['X'])
        config['yaxis'] = np.array([0.0, 0.0, 1.0])
        config['xaxis'] = np.cross(config['yaxis'], config['zaxis'])
        self.geometry = GeometryObject(config, strict=False)

    def setup(self):
        super().setup()

        self.log.debug('Yo mama is here.')

        self.setup_geometry()

        # Generate the fine mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_mesh(
            self.param['mesh_size_a'], self.param['mesh_size_b'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_mesh(
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

    def spiral(self, phi, beta, inp, dict=False):
        b = inp['b']
        r0 = inp['r0']
        theta0 = inp['theta0']
        S = inp.get('S', jnp.array([0.0, 0.0, 0.0]))

        r = xmj.sinusoidal_spiral(phi, b, r0, theta0)
        a = theta0 + b * phi
        t = theta0 + (b - 1) * phi
        C_norm = jnp.array([-1*jnp.sin(a), jnp.cos(a), 0.0])

        C = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), 0.0])
        rho = r / (b * jnp.sin(t))
        O = C + rho * C_norm

        CS = S - C
        CS_dist = jnp.linalg.norm(CS)
        CS_hat = CS / CS_dist

        # When the source is at the origin, bragg will equal theta.
        bragg = jnp.pi/2 - jnp.arccos(jnp.dot(CS_hat, C_norm))
        axis = jnp.array([0.0, 0.0, 1.0])
        CD_hat = xmj.vector_rotate(CS_hat, axis, -2 * (jnp.pi / 2 - bragg))
        CD_dist = rho * jnp.sin(bragg)
        D = C + CD_dist * CD_hat

        SD = D - S
        SD_hat = SD / jnp.linalg.norm(SD)

        CP_hat = -1 * jnp.cross(SD_hat, axis)
        aDSC = xmj.vector_angle(SD_hat, -1 * CS_hat)
        CP_dist = CS_dist * jnp.sin(aDSC)
        P = C + CP_hat * CP_dist

        CQ_hat = C_norm
        aQCP = xmj.vector_angle(CQ_hat, CP_hat)
        CQ_dist = CP_dist / jnp.cos(aQCP)
        Q = C + CQ_hat * CQ_dist

        XP_hat = xmj.vector_rotate(CP_hat, SD_hat, beta)
        X = P - XP_hat * CP_dist

        if dict:
            out = {}
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

    def spiral_centered_dict(self, a, b, inp):
        out = self.spiral(a, b, inp, dict=True)
        for key in out:
            out[key] = self.geometry.point_to_local(out[key])
        return out

    def spiral_jax(self, a, b, inp):
        xyz = self.spiral(a, b, inp)
        dfda, dfdb = jax.jacfwd(self.spiral, (0, 1))(a, b, inp)
        norm_ad = np.cross(dfda, dfdb)
        norm_ad /= np.linalg.norm(norm_ad)
        return xyz, norm_ad

    def spiral_centered_jax(self, a, b, inp):
        xyz_ext, norm_ext = self.spiral_jax(a, b, inp)
        xyz_local = self.geometry.point_to_local(xyz_ext)
        norm_local = self.geometry.vector_to_local(norm_ext)
        return xyz_local, norm_local

    def generate_mesh(self, mesh_size_a=None, mesh_size_b=None):
        """
        This method creates the meshgrid for the crystal
        """

        inp = {}
        inp['r0'] = self.param['r0']
        inp['b'] = self.param['b']
        inp['theta0'] = self.param['theta0']

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
        for ii_a in range(num_a):
            for ii_b in range(num_b):

                a = aa[ii_a, ii_b]
                b = bb[ii_a, ii_b]

                # Temporary for development.
                if self.param['normal_method'] == 'jax':
                    xyz, norm = self.spiral_centered_jax(a, b, inp)
                elif self.param['normal_method'] == 'ideal_np':
                    out = self.spiral_centered_dict(a, b, inp)
                    xyz = out['X']
                    norm = (out['Q'] - out['X'])/np.linalg.norm(out['Q'] - out['X'])
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

