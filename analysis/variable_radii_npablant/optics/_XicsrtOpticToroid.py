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

from xicsrt import xicsrt_math
from xicsrt.xicsrt_math import *
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

class XicsrtOpticToroid(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size'] = 0.0005
        config['mesh_coarse_size'] = 0.005
        config['major_radius'] = 1.0
        config['minor_radius'] = 0.1

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

        # Generate the fine mesh.
        mesh_points, mesh_faces = self.generate_crystal_mesh(self.param['mesh_size'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_faces = self.generate_crystal_mesh(self.param['mesh_coarse_size'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points
        self.log.debug(f'Coarse mesh points: {mesh_points.shape[0]}')

        # Calculate height at width.
        self.param['width'] = np.max(
            self.param['mesh_faces'][:,0])-np.min(self.param['mesh_faces'][:,0])
        self.param['height'] = np.max(
            self.param['mesh_faces'][:,1])-np.min(self.param['mesh_faces'][:,1])


    def toroid_calculate(self, a, b, input):
        C_center = input['crystal_center']
        C_zaxis  = input['crystal_zaxis']
        C_yaxis  = input['crystal_yaxis']
        maj_r    = input['major_radius']
        min_r    = input['minor_radius']

        C_norm = vector_rotate(C_zaxis, C_yaxis, a)
        C = C_center - maj_r * C_norm
        P = C + C_norm * min_r

        axis = np.cross(C_norm, C_yaxis)
        PC_norm = xicsrt_math.vector_rotate(-1 * C_norm, axis, b)
        xyz = P + PC_norm * min_r
        return xyz


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

        a_range = [-0.01, 0.01]
        b_range = [-0.1, 0.1]
        a_span = a_range[1] - a_range[0]
        b_span = a_range[1] - a_range[0]
        num_a = np.ceil(a_span/mesh_size).astype(int)
        num_b = np.ceil(b_span/mesh_size).astype(int)
        # Temporary until I figure out a better solution.
        num_a *= 4

        self.log.debug(f'mesh_size: {mesh_size} rad, total: {num_a*num_b}')
        self.log.debug(f'num_a, num_b: {num_a}, {num_b}')

        alpha = np.linspace(a_range[0], a_range[1], num_a)
        beta = np.linspace(b_range[0], b_range[1], num_b)

        aa, bb = np.meshgrid(alpha, beta, indexing='ij')
        xx = np.empty((num_a, num_b))
        yy = np.empty((num_a, num_b))
        zz = np.empty((num_a, num_b))

        for ii_a in range(num_a):
            for ii_b in range(num_b):
                b = bb[ii_a, ii_b]
                a = aa[ii_a, ii_b]
                xyz = self.toroid_calculate(a,b, input)
                xx[ii_a, ii_b] = xyz[0]
                yy[ii_a, ii_b] = xyz[1]
                zz[ii_a, ii_b] = xyz[2]

        # Combine x & y arrays, add z dimension
        points_2d = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T
        tri = Delaunay(angles_2d)
        # tri = Delaunay(points_2d)

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        faces = tri.simplices

        return points, faces

