# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  - Matthew Slominski <mattisaacslominski@yahoo.com>
"""
import numpy as np
from scipy.spatial import Delaunay

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_math
from xicsrt.tools.xicsrt_math import *
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

class XicsrtOpticToroid(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size'] = 0.0005
        config['mesh_coarse_size'] = 0.005
        config['range_alpha'] = [-0.01, 0.01]
        config['range_beta']  = [-0.04, 0.04]
        config['major_radius'] = 1.0
        config['minor_radius'] = 0.1

        # Temporary for development.
        config['use_finite_diff'] = False

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

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


    def toroid_calculate(self, a, b, input):
        C_center = input['crystal_center']
        C_zaxis  = input['crystal_zaxis']
        C_yaxis  = input['crystal_yaxis']
        maj_r    = input['major_radius']
        min_r    = input['minor_radius']

        C_norm = vector_rotate(C_zaxis, C_yaxis, a)
        C = C_center - maj_r * C_norm
        Q = C + C_norm * min_r

        axis = np.cross(C_norm, C_yaxis)
        C_norm = xicsrt_math.vector_rotate(C_norm, axis, b)
        xyz = Q - C_norm * min_r
        norm = C_norm

        return xyz, norm

    def toroid_calculate_fd(self, a, b, input, delta=None):
        profiler.start('finite difference')
        if delta is None: delta = 1e-8

        xyz, norm = self.toroid_calculate(a, b, input)
        xyz1, norm1 = self.toroid_calculate(a + delta, b, input)
        xyz2, norm2 = self.toroid_calculate(a, b + delta, input)

        vec1 = xyz1 - xyz
        vec2 = xyz2 - xyz
        norm_fd = np.cross(vec1, vec2)
        norm_fd /= np.linalg.norm(norm_fd)
        profiler.stop('finite difference')
        return xyz, norm_fd

    def generate_crystal_mesh(self, mesh_size):
        """
        This method creates the meshgrid for the crystal
        """
        if mesh_size is None: mesh_size = self.param['mesh_size']

        crystal_name = 'Ge (400)'
        self.log.debug(f'crystal    : {crystal_name}')

        major_radius = self.param['major_radius']
        minor_radius = self.param['minor_radius']

        crystal_spacing = 2.82868 / 2
        crystal_location = np.array([0.0, 0.0, 0.0])
        C_zaxis = np.array([0.0, 0.0, -1.0])
        C_xaxis = np.array([1.0, 0.0, 0.0])
        C_yaxis = np.cross(C_xaxis, C_zaxis)
        C_center = crystal_location + C_zaxis * major_radius

        input = {}
        input['crystal_center'] = C_center
        input['crystal_zaxis'] = C_zaxis
        input['crystal_yaxis'] = C_yaxis
        input['major_radius'] = major_radius
        input['minor_radius'] = minor_radius

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

        for ii_a in range(num_a):
            for ii_b in range(num_b):
                b = bb[ii_a, ii_b]
                a = aa[ii_a, ii_b]
                # Temporary for development.
                if self.param['use_finite_diff']:
                    xyz, norm = self.toroid_calculate_fd(a, b, input)
                else:
                    xyz, norm = self.toroid_calculate(a, b, input)
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

