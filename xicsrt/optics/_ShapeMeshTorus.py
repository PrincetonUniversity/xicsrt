# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

Define the :class:`ShapeMeshTorus` class.
"""
import numpy as np
from scipy.spatial import Delaunay
from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm

from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper
class ShapeMeshTorus(ShapeMesh):
    """
    A toroidal crystal implemented using a mesh-grid.

    This class meant to be used for two reasons:
    - As an example and template for how to implement a mesh-grid optic.
    - As a verification of the mesh-grid implementation.

    The analytical :class:`ShapeTorus` object should be used for all normal
    raytracing purposes.
    """

    def default_config(self):
        config = super().default_config()
        config['use_meshgrid'] = True
        config['mesh_refine'] = True
        config['mesh_size'] = (21,11)
        config['mesh_coarse_size'] = (11,5)
        config['angle_major'] = [-0.01, 0.01]
        config['angle_minor']  = [-0.05, 0.05]

        # Parameters needed to define geometry.
        config['radius_major'] = 1.0
        config['radius_minor'] = 0.1

        # Calculation options.
        config['normal_method'] = 'analytic'

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

        # Generate the fine mesh.
        mesh_points, mesh_faces, mesh_normals = self.generate_mesh(self.param['mesh_size'])
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals

        # Generate the coarse mesh.
        mesh_points, mesh_faces, mesh_normals = self.generate_mesh(self.param['mesh_coarse_size'])
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points
        self.param['mesh_coarse_normals'] = mesh_normals

        # Calculate width and height of the optic.
        self.param['xsize'] = np.max(
            self.param['mesh_points'][:,0])-np.min(self.param['mesh_points'][:,0])
        self.param['ysize'] = np.max(
            self.param['mesh_points'][:,1])-np.min(self.param['mesh_points'][:,1])
        self.log.debug(f"WxH: {self.param['xsize']:0.3f}x{self.param['ysize']:0.3f}")

    def torus(self, a, b):
        """
        Return a 3D surface coordinate given a set of two angles.
        """
        C0 = self.param['origin']
        C0_zaxis  = self.param['zaxis']
        C0_xaxis  = self.param['xaxis']
        maj_r     = self.param['radius_major']
        min_r     = self.param['radius_minor']

        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)
        O = C0 + maj_r * C0_zaxis

        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = O - maj_r * C_norm
        Q = C + C_norm * min_r

        axis = np.cross(C_norm, C0_yaxis)
        X_norm = xm.vector_rotate(C_norm, axis, b)
        X = Q - X_norm * min_r

        return X, X_norm

    def shape(self, a, b):
        return self.torus(a, b)

    def shape_fd(self, a, b, delta=None):
        profiler.start('finite difference')
        if delta is None: delta = 1e-8

        xyz, _ = self.torus(a, b)
        xyz1, _ = self.torus(a + delta, b)
        xyz2, _ = self.torus(a, b + delta)

        vec1 = xyz1 - xyz
        vec2 = xyz2 - xyz
        norm_fd = np.cross(vec1, vec2)
        norm_fd /= np.linalg.norm(norm_fd)
        profiler.stop('finite difference')
        return xyz, norm_fd

    def shape_jax(self, a, b):
        raise NotImplementedError()

    def calculate_mesh(self, a, b):
        profiler.start('calculate_mesh')

        num_a = len(a)
        num_b = len(b)

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
                if self.param['normal_method'] == 'analytic':
                    xyz, norm = self.shape(a, b)
                elif self.param['normal_method'] == 'fd':
                    xyz, norm = self.shape_fd(a, b)
                elif self.param['normal_method'] == 'jax':
                    xyz, norm = self.shape_jax(a, b)
                else:
                    raise Exception(f"normal_method {self.param['normal_method']} unknown.")

                xx[ii_a, ii_b] = xyz[0]
                yy[ii_a, ii_b] = xyz[1]
                zz[ii_a, ii_b] = xyz[2]
                normal_xx[ii_a, ii_b] = norm[0]
                normal_yy[ii_a, ii_b] = norm[1]
                normal_zz[ii_a, ii_b] = norm[2]

        profiler.stop('calculate_mesh')

        return xx, yy, zz, normal_xx, normal_yy, normal_zz

    def generate_mesh(self, mesh_size=None):
        """
        This method creates the meshgrid for the crystal
        """
        profiler.start('generate_mesh')

        # --------------------------------
        # Setup the basic grid parameters.

        a_range = self.param['angle_major']
        b_range = self.param['angle_minor']

        num_a = mesh_size[0]
        num_b = mesh_size[1]

        self.log.debug(f'num_a, num_b: {num_a}, {num_b}, total: {num_a*num_b}')

        a = np.linspace(a_range[0], a_range[1], num_a)
        b = np.linspace(b_range[0], b_range[1], num_b)

        xx, yy, zz, normal_xx, normal_yy, normal_zz = \
            self.calculate_mesh(a, b)

        aa, bb = np.meshgrid(a, b, indexing='ij')
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T
        tri = Delaunay(angles_2d)

        # It's also possible to triangulate using the x,y coordinates.
        # This is not recommended unless there is some specific need.
        #
        # points_2d = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        # tri = Delaunay(points_2d)

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        normals = np.stack((normal_xx.flatten(), normal_yy.flatten(), normal_zz.flatten())).T

        faces = tri.simplices

        profiler.stop('generate_mesh')
        return points, faces, normals

