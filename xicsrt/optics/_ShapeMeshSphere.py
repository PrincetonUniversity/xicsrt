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
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm

from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper
class ShapeMeshSphere(ShapeMesh):
    """
    A spherical crystal implemented using a mesh-grid.

    This class meant to be used for two reasons:
    - As an example and template for how to implement a mesh-grid optic.
    - As a verification of the mesh-grid implementation.

    The analytical :class:`ShapeSphere` object should be used for all normal
    raytracing purposes.
    """

    def default_config(self):
        """
        radius : float (1.0)
          The radius of the sphere.

        mesh_size : (float, float) ((11,11))
          The number of mesh points in the x and y directions.

        mesh_coarse_size : (float, float) ((5,5))
          The number of mesh points in the x and y directions.
        """
        config = super().default_config()
        config['radius'] = 1.0
        config['mesh_size'] = (11,11)
        config['mesh_coarse_size'] = (5,5)

        return config

    def setup(self):
        super().setup()

        # Generate the fine mesh.
        mesh_points, mesh_faces, mesh_normals = self.generate_mesh(self.param['mesh_size'])
        mesh_points_ext = self.point_to_external(mesh_points)
        mesh_normals_ext = self.vector_to_external(mesh_normals)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points_ext
        self.param['mesh_normals'] = mesh_normals_ext

        # Generate the coarse mesh.
        mesh_points, mesh_faces, mesh_normals = self.generate_mesh(self.param['mesh_coarse_size'])
        mesh_points_ext = self.point_to_external(mesh_points)
        mesh_normals_ext = self.vector_to_external(mesh_normals)
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points_ext
        self.param['mesh_coarse_normals'] = mesh_normals_ext

    def generate_mesh(self, meshsize):
        """
        Create a spherical meshgrid in local coordinates.
        """

        xsize = self.param['xsize']
        ysize = self.param['ysize']

        x = np.linspace(-xsize/2, xsize/2, meshsize[0])
        y = np.linspace(-ysize/2, ysize/2, meshsize[1])

        # Center of the sphere in local coordinates
        center = np.array([0.0, 0.0, self.param['radius']])

        # Create x,y meshgrid arrays, then calculate z coords
        xx, yy = np.meshgrid(x, y)
        zz = self.param['radius'] - np.sqrt(self.param['radius']**2 - xx**2 - yy**2)

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        norm = xm.normalize(center - points)

        tri = Delaunay(points[:, 0:2])
        faces = tri.simplices

        return points, faces, norm
