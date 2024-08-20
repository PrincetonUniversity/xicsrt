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

    This class meant to be used for three reasons:
    - A toroidal optic shape usable with large radii of curvature
    - As an example and template for how to implement a mesh-grid optic.
    - As a verification of the mesh-grid implementation.

    The analytical :class:`ShapeTorus` object will be much faster.

    **Programming Notes**

    This optic is built in local coordinates with the mesh surface normal
    generally in the local z = [0, 0, 1] direction and with
    config['trace_local'] = True. This is recommended because the mesh
    implementation performs triangulation and interpolation in the local
    x-y plane.

    """

    def default_config(self):
        """
        radius_major: float (1.0)
            The radius of curvature of the crystal in the toroidal (xaxis)
            direction. This is not the same as the geometric major radius of the
            axis of a toroid, which in our case would be r_major-r_minor.

        radius_minor: float (0.1)
            The radius of the curvature of the crystal in the poloidal (yaxis)
            direction. This is the same as the geometric minor radius of a
            toroid.

        normal_method: str ('analytic')
            Specify how to calculate the normal vectors at each of the grid
            points.  Supported values are 'analytic' and 'fd'.

            When set to 'fd' a finite difference method will be used. This is
            primarily here as an example for cases in which the surface position
            is easy to calculate but where surface normals are difficult. A
            better option in these cases however is to use auto-differentiation
            to calculate analytical derivatives (for example using jax).

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

        config['radius_major'] = 1.0
        config['radius_minor'] = 0.2
        config['convex'] = [False, False]

        config['normal_method'] = 'analytic'

        # The meshgrid is defined in local coordinates.
        config['trace_local'] = True

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

        if np.all(self.param['convex'] == [False, False]):
            self.param['torus_sign_major'] = 1
            self.param['torus_sign_minor'] = 1
        elif np.all(self.param['convex'] == [False, True]):
            self.param['torus_sign_major'] = 1
            self.param['torus_sign_minor'] = -1
        elif np.all(self.param['convex'] == [True, False]):
            self.param['torus_sign_major'] = -1
            self.param['torus_sign_minor'] = 1
        elif np.all(self.param['convex'] == [True, True]):
            self.param['torus_sign_major'] = -1
            self.param['torus_sign_minor'] = -1
        else:
            raise Exception(f"Cannot be parse convex config option: {self.param['convex']}")


        # Calculate the angles that define the physical mesh size
        if (xsize := self.param['mesh_xsize']) is None:
            xsize = self.param['xsize']
        if (ysize := self.param['mesh_ysize']) is None:
            ysize = self.param['ysize']
        half_major = np.arcsin(xsize/2/(self.param['radius_major']))
        half_minor = np.arcsin(ysize/2/self.param['radius_minor'])
        self.param['angle_major'] = [-1*half_major, half_major]
        self.param['angle_minor'] = [-1*half_minor, half_minor]

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

        # Calculate final width and height of the optic for debugging.
        mesh_local = self.param['mesh_points']
        mesh_xsize = np.max(mesh_local[:,0])-np.min(mesh_local[:,0])
        mesh_ysize = np.max(mesh_local[:,1])-np.min(mesh_local[:,1])
        self.log.debug(f"Mesh xsize x ysize: {mesh_xsize:0.3f}x{mesh_ysize:0.3f}")

    def torus(self, a, b):
        """
        Return a 3D surface coordinate given a set of two angles.
        """
        C0_zaxis = np.asarray([0.0, 0.0, 1.0])
        C0_xaxis = np.asarray([1.0, 0.0, 0.0])
        s_maj  = self.param['torus_sign_major']
        s_min  = self.param['torus_sign_minor']
        r_maj = self.param['radius_major']
        r_min = self.param['radius_minor']

        C0_yaxis = np.cross(C0_zaxis, C0_xaxis)

        # r_maj is the radius of curvature of the surface, so the torus
        # center is always defined only by this value.
        center = r_maj*C0_zaxis*s_maj

        # This is the projection of point in the x-z plane.
        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = center - r_maj * C_norm * s_maj

        # Find the associated point on the torus axis
        Q = C + r_min*C_norm*s_min

        axis = np.cross(C_norm * s_min, C0_yaxis)
        X_norm = xm.vector_rotate(C_norm * s_min, axis, b)
        X = Q - X_norm * r_min

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

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        normals = np.stack((normal_xx.flatten(), normal_yy.flatten(), normal_zz.flatten())).T

        delaunay = Delaunay(angles_2d)
        faces = delaunay.simplices

        # It's also possible to triangulate using the x,y coordinates.
        # This does not work well for the toroidal shape.
        #
        # delaunay = Delaunay(points[:, 0:2])
        # faces = delaunay.simplices


        profiler.stop('generate_mesh')
        return points, normals, faces

