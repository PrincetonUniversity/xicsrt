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
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

@dochelper
class XicsrtOpticCrystalSphericalMesh(XicsrtOpticCrystal):
    """
    A spherical reflector implemented using a mesh-grid.

    This class meant to be used for two reasons:
    - As an example and template for how to implement a mesh-grid optic.
    - For verification of mesh-grid implementation.

    The analytical `XicsrtOpticCrystalSpherical` object should be used for all
    normal raytracing application.

    .. Warning::
        This module is significantly out of date. Do not use as is.
    """

    def default_config(self):
        """
        config['grid_resolution']:
          Refers to the density of the surface grid it should be
          an int 1-8, 1 being the lowest resolution. The int 1-8 is
          simply for the selection structure in 'generate_crystal_mesh'
          This is a temporary fix for convenience while working in jupyter
          and it should eventually be changed
        """
        config = super().default_config()
        config['radius'] = 1.0
        config['use_meshgrid'] = True
        config['grid_resolution'] = None

        return config

    def setup(self):
        super().setup()

        # Generate the fine mesh.
        mesh_points, mesh_faces = self.generate_crystal_mesh()
        mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points_ext
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_faces = self.generate_crystal_mesh(res=0)
        mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points_ext
        self.log.debug(f'Coarse mesh points: {mesh_points.shape[0]}')

    def generate_crystal_mesh(self, res=None):
        """
        This method creates the meshgrid for the crystal
        """

        if res is None:
            res = self.param['grid_resolution']

        # Create series of x,y points
        x_lim = self.param['xsize']
        y_lim = self.param['ysize']

        #Ensure that crystal has same ratio of mesh points per cm along height & width
        pts_ratio = self.param['ysize']/self.param['xsize']

        if res == 1:
            x_pts = self.param['xsize']*50
            y_pts = x_pts * pts_ratio
        elif res == 2:
            x_pts = self.param['xsize'] * 100
            y_pts = x_pts * pts_ratio
        elif res == 3:
            x_pts = self.param['xsize'] * 200
            y_pts = x_pts * pts_ratio
        elif res == 4:
            x_pts = self.param['xsize'] * 300
            y_pts = x_pts * pts_ratio
        elif res == 5:
            x_pts = self.param['xsize'] * 400
            y_pts = x_pts * pts_ratio
        elif res == 6:
            x_pts = self.param['xsize'] * 500
            y_pts = x_pts * pts_ratio
        elif res == 7:
            x_pts = self.param['xsize'] * 600
            y_pts = x_pts * pts_ratio
        elif res == 8:
            x_pts = self.param['xsize'] * 700
            y_pts = x_pts * pts_ratio
        elif res == 20:
            x_pts = self.param['xsize'] * 2000
            y_pts = x_pts * pts_ratio
        else:
            x_pts = 4
            y_pts = 10

        x = np.linspace(-x_lim/2, x_lim/2, int(x_pts))
        y = np.linspace(-y_lim/2, y_lim/2, int(y_pts))

        # Create x,y meshgrid arrays, calculate z coords
        xx, yy = np.meshgrid(x, y)
        zz = self.param['radius'] - np.sqrt(self.param['radius']** 2 - xx ** 2 - yy ** 2)

        # Combine x & y arrays, add z dimension
        points = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        mesh_points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        tri = Delaunay(points)
        mesh_faces = tri.simplices
        return mesh_points, mesh_faces
