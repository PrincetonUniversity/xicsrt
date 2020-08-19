# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  - Matthew Slominski <mattisaacslominski@yahoo.com>
"""
#import numpy as np
from scipy.spatial import Delaunay

from xicsrt.util import profiler
from xicsrt import xicsrt_math
from xicsrt.xicsrt_math import *
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

import jax
import jax.numpy as jnp


def vector_angle(a, b):
    """
    Find the angle between two vectors. Not vectorized.
    """
    angle = jnp.arccos(jnp.dot(a/jnp.linalg.norm(a), b/jnp.linalg.norm(b)))
    return angle

def vector_rotate(a, b, theta):
    """
    Rotate vector a around vector b by an angle theta (radians)

    Programming Notes:
      u: parallel projection of a on b_hat.
      v: perpendicular projection of a on b_hat.
      w: a vector perpendicular to both a and b.
    """
    b_hat = b / jnp.linalg.norm(b)
    u = b_hat * jnp.dot(a, b_hat)
    v = a - u
    w = jnp.cross(b_hat, v)
    c = u + v * jnp.cos(theta) + w * jnp.sin(theta)
    return c

class XicsrtOpticVariableRadiiCone(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size_a'] = 0.001
        config['mesh_size_b'] = 0.01
        config['mesh_coarse_size_a'] = 0.002
        config['mesh_coarse_size_b'] = 0.008
        config['range_a'] = [-0.01, 0.01]
        config['range_b']  = [-0.04, 0.04]

        # Temporary for development.
        config['normal_method'] = 'jax'

        return config

    def setup(self):
        super().setup()

        self.log.debug('Yo mama is here.')

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

    def multi_cone(self, a, b, inp, return_XQ_hat=False):

        C0 = inp['C0']
        C0_xaxis = inp['C0_xaxis']
        C0_yaxis = inp['C0_yaxis']
        C0_zaxis = inp['C0_zaxis']
        D0 = inp['D0']
        S = inp['S']

        C0_norm = C0_zaxis
        C0S = S - C0
        C0S_dist = jnp.linalg.norm(C0S)
        C0S_hat = C0S / C0S_dist
        bragg0 = jnp.abs(jnp.pi / 2 - vector_angle(C0S_hat, C0_norm))

        C0D0 = D0 - C0
        C0D0_dist = jnp.linalg.norm(C0D0)
        C0E_dist = C0D0_dist * jnp.cos(bragg0)

        C = C0 + a * C0_xaxis
        C_norm = C0_norm
        CS = S - C
        CS_dist = jnp.linalg.norm(CS)
        CS_hat = CS / jnp.linalg.norm(CS)
        bragg = jnp.abs(jnp.pi / 2 - vector_angle(CS_hat, C_norm))

        CE_dist = C0E_dist - a
        CD_dist = CE_dist / jnp.cos(bragg)
        CD_hat = vector_rotate(C_norm, C0_yaxis, -1 * (jnp.pi / 2 - bragg))
        D = C + CD_dist * CD_hat

        SD = D - S
        SD_hat = SD / jnp.linalg.norm(SD)

        CP_hat = -1 * jnp.cross(SD, C0_yaxis)
        CP_hat /= jnp.linalg.norm(CP_hat)
        aDSC = vector_angle(SD_hat, -1 * CS_hat)
        CP_dist = CS_dist * jnp.sin(aDSC)
        P = C + CP_hat * CP_dist

        CQ_hat = C_norm
        #aQCP = vector_angle(CQ_hat, CP_hat)
        #CQ_dist = CP_dist / jnp.cos(aQCP)
        #Q = C + CQ_hat * CQ_dist

        XP_hat = vector_rotate(CP_hat, SD_hat, b)
        X = P - XP_hat * CP_dist

        XQ_hat = vector_rotate(CQ_hat, SD_hat, b)

        if return_XQ_hat:
            return X, XQ_hat
        else:
            return X

    def multi_cone_jax(self, a, b, inp):
        xyz = self.multi_cone(a, b, inp)
        profiler.start('multi_cone_jax, normal')
        dfda, dfdb = jax.jacfwd(self.multi_cone, (0, 1))(a, b, inp)
        norm_ad = np.cross(dfda, dfdb)
        norm_ad /= np.linalg.norm(norm_ad)
        profiler.stop('multi_cone_jax, normal')
        return xyz, norm_ad

    def generate_mesh(self, mesh_size_a=None, mesh_size_b=None):
        """
        This method creates the meshgrid for the crystal
        """

        C_name = 'Ge (400)'
        C_spacing = 2.82868 / 2

        # Corresponds to an energy of 9.75 keV
        wavelength = 1.2716
        C0S_dist = 0.3
        C0D_dist = 1.0

        # Calculate the central bragg angle.
        bragg = xicsrt_math.bragg_angle(wavelength, C_spacing)
        self.log.debug(f'crystal    : {C_name}')
        self.log.debug(f'wavelength : {wavelength:0.4f} Å')
        self.log.debug(f'bragg_angle: {np.degrees(bragg):0.3f}°')

        # Define the Crystal Location
        C0 = np.array([0.0, 0.0, 0.0])
        C0_zaxis = np.array([0.0, 0.0, -1.0])
        C0_xaxis = np.array([1.0, 0.0, 0.0])
        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)

        # Calculate the source location.
        S = C0 + C0S_dist * vector_rotate(C0_zaxis, C0_yaxis, (np.pi / 2 - bragg))

        # Calculate the detector location.
        D0 = C0 + C0D_dist * vector_rotate(C0_zaxis, C0_yaxis, -1 * (np.pi / 2 - bragg))
        D0_zaxis = np.array([-1.0, 0.0, 0.0])
        D0_yaxis = np.array([0.0, 1.0, 0.0])
        D0_xaxis = np.cross(D0_zaxis, D0_yaxis)

        inp = {}
        inp['C0'] = C0
        inp['C0_xaxis'] = C0_xaxis
        inp['C0_yaxis'] = C0_yaxis
        inp['C0_zaxis'] = C0_zaxis
        inp['D0'] = D0
        inp['D0_xaxis'] = D0_xaxis
        inp['D0_yaxis'] = D0_yaxis
        inp['D0_zaxis'] = D0_zaxis
        inp['S'] = S

        # --------------------------------
        # Setup the basic grid parameters.

        a_range = self.param['range_a']
        b_range = self.param['range_b']
        a_span = a_range[1] - a_range[0]
        b_span = b_range[1] - b_range[0]

        # For now assume the mesh_size is the angular density
        # in the a direction.
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
                    xyz, norm = self.multi_cone_jax(a, b, inp)
                elif self.param['normal_method'] == 'ideal_np':
                    xyz, norm = self.multi_cone(a, b, inp, return_XQ_hat=True)
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

