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
        config['mesh_size'] = 0.001

        return config

    def setup(self):
        super().setup()
        mesh_points, mesh_faces = self.generate_crystal_mesh()

        self.param['mesh_points'] = mesh_points
        self.param['mesh_faces'] = mesh_faces

        self.param['width'] = np.max(mesh_points[:,0])-np.min(mesh_points[:,0])
        self.param['height'] = np.max(mesh_points[:,1])-np.min(mesh_points[:,1])

    def generate_crystal_mesh(self):
        """
        This method creates the meshgrid for the crystal
        """

        crystal_name = 'Ge (400)'

        maj_r = 2.0
        min_r = 0.20224

        crystal_spacing = 2.82868 / 2
        crystal_location = np.array([0.0, 0.0, 0.0])
        crystal_zaxis = np.array([0.0, 0.0, -1.0])
        crystal_xaxis = np.array([1.0, 0.0, 0.0])
        C_yaxis = np.cross(crystal_xaxis, crystal_zaxis)
        C_center = crystal_location + crystal_zaxis * maj_r

        # --------------------------------
        # Setup the basic grid parameters.

        a_range = [-0.01, 0.01]
        b_range = [-0.02, 0.02]
        a_span = a_range[1] - a_range[0]
        b_span = a_range[1] - a_range[0]
        num_a = np.ceil(a_span/self.param['mesh_size']).astype(int)
        num_b = np.ceil(b_span/self.param['mesh_size']).astype(int)
        # Temporary until I figure out a better solution.
        num_a *= 4

        print(f'num_a, num_b: {num_a}, {num_b}  total: {num_a*num_b}')

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

                C_tmp = vector_rotate(-1 * crystal_zaxis, C_yaxis, a)
                C = C_center + maj_r * C_tmp

                C_norm = C_center - C
                C_norm = C_norm / np.linalg.norm(C_norm)

                P = C + C_norm * min_r

                axis = np.cross(C_norm, C_yaxis)
                new_vector = xicsrt_math.vector_rotate(-1*C_norm, axis, b)
                xyz = P + new_vector * min_r
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

