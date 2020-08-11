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

class XicsrtOpticVariableRadiiToroid(XicsrtOpticCrystal):

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

        major_radius = 2.0

        crystal_spacing = 2.82868 / 2
        crystal_location = np.array([0.0, 0.0, 0.0])
        crystal_zaxis = np.array([0.0, 0.0, -1.0])
        crystal_xaxis = np.array([1.0, 0.0, 0.0])
        crystal_yaxis = np.cross(crystal_xaxis, crystal_zaxis)

        # Corresponds to an energy of 9.57 keV
        wavelength = 1.2716

        source_distance = 0.3

        # Start calculations.
        crystal_center = crystal_location + crystal_zaxis * major_radius

        bragg_angle = xicsrt_math.bragg_angle(wavelength, crystal_spacing)
        print(f'crystal    : {crystal_name}')
        print(f'wavelength : {wavelength:0.4f} Å')
        print(f'bragg_angle: {np.degrees(bragg_angle):0.3f}°')

        # Everything sholud be coded to handle an arbitrary source location
        # this is just a good starting point.
        source_location = (
            crystal_location
            + source_distance * vector_rotate(
                crystal_zaxis
                ,crystal_yaxis
                ,(np.pi / 2 - bragg_angle))
            )

        S = source_location

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

        # Define matrixes for the major and minor radius centers.
        #
        # All of these only vary with alpha for our the geometries
        # that we care about, but I might as well generalize this.
        # maj_center = np.empty((num_alpha, num_beta, 3))
        # min_center = np.empty((num_alpha, num_beta, 3))
        # maj_radius = np.empty((num_alpha, num_beta))
        # min_radius = np.empty((num_alpha, num_beta))

        # ---------------------------
        # Setup the major radius.
        # maj_radius[:] = major_radius

        # Setup some variables for saving parameters on the grid.
        D_save = np.empty((num_a, num_b, 3))

        # ------------------------------------------------
        # Now calculate the xyz values at each grid point.
        #
        # This is my first attempt at the geometry; I have made no attempt
        # at all to do this efficently, and I am probably caculating a lot
        # of things that are not needed.
        C = np.zeros(3)
        C_norm = np.zeros(3)
        for ii_a in range(num_a):
            for ii_b in range(num_b):
                # ii_a = 0
                # ii_b = 0

                a = aa[ii_a, ii_b]
                b = bb[ii_a, ii_b]

                C_tmp = vector_rotate(-1 * crystal_zaxis, crystal_yaxis, a)
                C = crystal_center + major_radius * C_tmp

                C_norm[0] = crystal_center[0] - C[0]
                C_norm[2] = crystal_center[2] - C[2]
                C_norm /= np.linalg.norm(C_norm)

                SC = C - S
                SC_dist = np.linalg.norm(SC)
                SC_norm = SC / np.linalg.norm(SC)

                SC_bragg = np.abs(np.pi / 2 - vector_angle(SC_norm, C_norm))

                D_dist = major_radius * np.cos(SC_bragg)
                D = crystal_center + D_dist * vector_rotate(-1 * C_norm, crystal_yaxis, SC_bragg)

                SD = D - S
                SD_norm = SD / np.linalg.norm(SD)

                PC_norm = np.cross(SD, crystal_yaxis)
                PC_norm /= np.linalg.norm(PC_norm)

                eta = vector_angle(SD_norm, SC_norm)
                PC_dist = SC_dist * np.sin(eta)

                P = C + (-1) * PC_norm * PC_dist

                vector = vector_rotate(PC_norm, SD_norm, b)

                xyz = P + vector * PC_dist
                xx[ii_a, ii_b] = xyz[0]
                yy[ii_a, ii_b] = xyz[1]
                zz[ii_a, ii_b] = xyz[2]

                D_save[ii_a, ii_b] = D
                # break
            # break

        # Combine x & y arrays, add z dimension
        points_2d = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T
        tri = Delaunay(angles_2d)
        # tri = Delaunay(points_2d)

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        faces = tri.simplices

        return points, faces

