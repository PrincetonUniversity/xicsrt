# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

Define the :class:`ShapeMeshcylinder` class.
"""
import numpy as np
from scipy.spatial import Delaunay
from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm

from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper
class ShapeMeshCylinder(ShapeMesh):
    """
    A toroidal crystal implemented using a mesh-grid.

    This class meant to be used for three reasons:
    - A toroidal optic shape usable with large radii of curvature
    - As an example and template for how to implement a mesh-grid optic.
    - As a verification of the mesh-grid implementation.

    The analytical :class:`Shapecylinder` object will be much faster.

    **Programming Notes**

    This optic is built in local coordinates with the mesh surface normal
    generally in the local z = [0, 0, 1] direction and with
    config['trace_local'] = True. This is recommended because the mesh
    implementation performs triangulation and interpolation in the local
    x-y plane.

    """

    def default_config(self):
        """
        radius: float (1.0)
            The radius of curvature of the crystal.

        mesh_size : (float, float) ((11,11))
          The number of mesh points in the x and y directions.

        mesh_coarse_size : (float, float) ((5,5))
          The number of mesh points in the x and y directions.
        """
        config = super().default_config()
        config['mesh_refine'] = True
        config['mesh_size'] = (11,11)
        config['mesh_coarse_size'] = (5,5)
        config['mesh_xsize'] = None
        config['mesh_ysize'] = None

        config['radius'] = 1.0

        # It's good practice to always define meshes in local coordinates.
        config['trace_local'] = True

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

        # Calculate the angles that define the physical mesh size
        if (xsize := self.param['mesh_xsize']) is None:
            xsize = self.param['xsize']
        if (ysize := self.param['mesh_ysize']) is None:
            ysize = self.param['ysize']
        self.param['x_range'] = [-1*xsize/2, xsize/2]
        half_angle = np.arcsin(ysize/2/(self.param['radius']))
        self.param['angle_range'] = [-1*half_angle, half_angle]

        # Generate the fine mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_mesh(self.param['mesh_size'])
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.param['mesh_faces'] = mesh_faces

        # Generate the coarse mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_mesh(self.param['mesh_coarse_size'])
        self.param['mesh_coarse_points'] = mesh_points
        self.param['mesh_coarse_normals'] = mesh_normals
        self.param['mesh_coarse_faces'] = mesh_faces

        # Calculate the final width and height of the optic.
        mesh_local = self.param['mesh_points']
        mesh_xsize = np.max(mesh_local[:,0])-np.min(mesh_local[:,0])
        mesh_ysize = np.max(mesh_local[:,1])-np.min(mesh_local[:,1])
        self.log.debug(f"Mesh xsize x ysize: {mesh_xsize:0.3f}x{mesh_ysize:0.3f}")

    def cylinder(self, x, angle):
        """
        Return a 3D surface coordinate given a distance and angle.
        """
        C0 = np.array([0.0, 0.0, 0.0])
        C0_zaxis  = np.array([0.0, 0.0, 1.0])
        C0_xaxis  = np.array([1.0, 0.0, 0.0])
        radius     = self.param['radius']

        x_vec = np.array([x, 0.0, 0.0])

        O = C0 + radius * C0_zaxis + x_vec

        X_norm = xm.vector_rotate(C0_zaxis, C0_xaxis, angle)
        X = O - radius * X_norm

        return X, X_norm

    def shape(self, a, b):
        return self.cylinder(a, b)

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

                xyz, norm = self.shape(a, b)

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

        a_range = self.param['x_range']
        b_range = self.param['angle_range']

        num_a = mesh_size[0]
        num_b = mesh_size[1]

        self.log.debug(f'num_a, num_b: {num_a}, {num_b}, total: {num_a*num_b}')

        a = np.linspace(a_range[0], a_range[1], num_a)
        b = np.linspace(b_range[0], b_range[1], num_b)

        xx, yy, zz, normal_xx, normal_yy, normal_zz = \
            self.calculate_mesh(a, b)

        aa, bb = np.meshgrid(a, b, indexing='ij')
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        normals = np.stack((normal_xx.flatten(), normal_yy.flatten(), normal_zz.flatten())).T

        delaunay = Delaunay(angles_2d)
        faces = delaunay.simplices

        # It's also possible to triangulate using the x,y coordinates.
        # This is not recommended unless there is some specific need.
        #
        # delaunay = Delaunay(points[:, 0:2])
        # faces = delaunay.simplices


        profiler.stop('generate_mesh')
        return points, normals, faces

