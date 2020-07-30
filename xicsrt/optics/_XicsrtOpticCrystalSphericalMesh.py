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
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

class XicsrtOpticCrystalSphericalMesh(XicsrtOpticCrystal):

    def get_default_config(self):
        """
        config['grid_resolution']:
          Refers to the density of the surface grid it should be
          an int 1-8, 1 being the lowest resolution. The int 1-8 is
          simply for the selection structure in 'generate_crystal_mesh'
          This is a temporary fix for convenience while working in jupyter
          and it should eventually be changed
        """
        config = super().get_default_config()
        config['radius'] = 1.0
        config['use_meshgrid'] = True
        config['grid_resolution'] = None

        return config

    def setup(self):
        super().setup()
        mesh_faces, mesh_points = self.generate_crystal_mesh()
        mesh_points_ext = self.point_to_external(mesh_points)

        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points_ext

    def generate_crystal_mesh(self):
        """
        This method creates the meshgrid for the crystal
        """

        # Create series of x,y points
        x_lim = self.param['width']
        y_lim = self.param['height']

        #Ensure that crystal has same ratio of mesh points per cm along height & width
        pts_ratio = self.param['height']/self.param['width']

        if self.param['grid_resolution'] == 1:
            x_pts = self.param['width']*50
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 2:
            x_pts = self.param['width'] * 100
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 3:
            x_pts = self.param['width'] * 200
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 4:
            x_pts = self.param['width'] * 300
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 5:
            x_pts = self.param['width'] * 400
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 6:
            x_pts = self.param['width'] * 500
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 7:
            x_pts = self.param['width'] * 600
            y_pts = x_pts * pts_ratio
        elif self.param['grid_resolution'] == 8:
            x_pts = self.param['width'] * 700
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
        return mesh_faces, mesh_points
