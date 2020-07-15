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
import matplotlib.tri as mtri
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

class XicsrtOpticCrystalSphericalMesh(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['radius'] = 1.0
        config['use_meshgrid'] = True
        return config

    def initialize(self):
        super().initialize()
        #self.param['center'] = self.param['radius'] * self.param['zaxis'] + self.param['origin']
        mesh_faces, mesh_points = self.generate_crystal_mesh()
        self.param['mesh_faces'] = mesh_faces
        mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_points'] = mesh_points_ext

    def generate_crystal_mesh(self):
        """
        This method creates the meshgrid for the crystal
        """
        # Create series of x,y points
        x = np.linspace(-0.02, 0.02, 8)
        y = np.linspace(-0.05, 0.05, 20)

        # Create x,y meshgrid arrays, calculate z coords
        xx, yy = np.meshgrid(x, y)
        zz = np.sqrt(xx ** 2 + yy ** 2 + 1.4503999948501587 ** 2) - 1.4503999948501587

        # Combine x & y arrays, add z dimension
        points = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        mesh_points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        tri = Delaunay(points)
        mesh_faces = tri.simplices
        return mesh_faces, mesh_points
