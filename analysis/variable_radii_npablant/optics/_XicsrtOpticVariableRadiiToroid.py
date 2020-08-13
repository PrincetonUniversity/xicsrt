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
        config['mesh_size'] = 0.0005
        config['mesh_coarse_size'] = 0.005
        config['major_radius'] = 1.0
        config['minor_radius'] = None

        return config

    def setup(self):
        super().setup()

        self.log.debug('Yo mama is here.')

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


    def vr_toroid_calculate(self, a, b, input):
        C_center = input['crystal_center']
        C_zaxis  = input['crystal_zaxis']
        C_yaxis  = input['crystal_yaxis']
        S        = input['source_location']
        maj_r    = input['major_radius']

        C_norm = vector_rotate(C_zaxis, C_yaxis, a)
        C = C_center - maj_r * C_norm

        SC = C - S
        SC_dist = np.linalg.norm(SC)
        SC_norm = SC / np.linalg.norm(SC)

        SC_bragg = np.abs(np.pi / 2 - vector_angle(SC_norm, C_norm))

        D_dist = maj_r * np.cos(SC_bragg)
        D = C_center + D_dist * vector_rotate(-1 * C_norm, C_yaxis, SC_bragg)

        SD = D - S
        SD_norm = SD / np.linalg.norm(SD)

        PC_norm = np.cross(SD, C_yaxis)
        PC_norm /= np.linalg.norm(PC_norm)

        eta = vector_angle(SD_norm, SC_norm)
        PC_dist = SC_dist * np.sin(eta)

        P = C + (-1) * PC_norm * PC_dist

        PC_norm_3d = vector_rotate(PC_norm, SD_norm, b)

        xyz = P + PC_norm_3d * PC_dist

        return xyz

    def generate_crystal_mesh(self, mesh_size=None):
        """
        This method creates the meshgrid for the crystal
        """
        if mesh_size is None: mesh_size = self.param['mesh_size']

        crystal_name = 'Ge (400)'

        major_radius = self.param['major_radius']
        crystal_spacing = 2.82868 / 2
        crystal_location = np.array([0.0, 0.0, 0.0])
        crystal_zaxis = np.array([0.0, 0.0, -1.0])
        crystal_xaxis = np.array([1.0, 0.0, 0.0])
        crystal_yaxis = np.cross(crystal_xaxis, crystal_zaxis)

        # Corresponds to an energy of 9.75 keV
        wavelength = 1.2716
        source_distance = 0.3

        # Start calculations.
        crystal_center = crystal_location + crystal_zaxis * major_radius

        # Calculate the source location for Manfred's setup.
        bragg_angle = xicsrt_math.bragg_angle(wavelength, crystal_spacing)
        self.log.debug(f'crystal    : {crystal_name}')
        self.log.debug(f'wavelength : {wavelength:0.4f} Å')
        self.log.debug(f'bragg_angle: {np.degrees(bragg_angle):0.3f}°')
        source_location = (
            crystal_location
            + source_distance * vector_rotate(
                crystal_zaxis
                ,crystal_yaxis
                ,(np.pi / 2 - bragg_angle))
            )

        input = {}
        input['crystal_center'] = crystal_center
        input['crystal_zaxis'] = crystal_zaxis
        input['crystal_yaxis'] = crystal_yaxis
        input['source_location'] = source_location
        input['major_radius'] = major_radius

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

        # ------------------------------------------------
        # Now calculate the xyz values at each grid point.
        #
        # This is my first attempt at the geometry; I have made no attempt
        # at all to do this efficently, and I am probably caculating a lot
        # of things that are not needed.
        for ii_a in range(num_a):
            for ii_b in range(num_b):

                a = aa[ii_a, ii_b]
                b = bb[ii_a, ii_b]
                xyz = self.vr_toroid_calculate(a, b, input)
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

