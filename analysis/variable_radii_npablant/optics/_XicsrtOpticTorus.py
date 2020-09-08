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

from xicsrt.util import profiler
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

class XicsrtOpticTorus(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size_a'] = 0.005
        config['mesh_size_b'] = 0.01
        config['mesh_coarse_size_a'] = 0.01
        config['mesh_coarse_size_b'] = 0.02
        config['range_a'] = [-0.01, 0.01]
        config['range_b']  = [-0.04, 0.04]

        # Parameters needed to define geometry.
        config['major_radius'] = 1.0
        config['minor_radius'] = 0.1
        config['crystal_spacing'] = 2.82868/2
        config['lambda0']   = 1.2716
        config['source_distance']  = 0.3

        # Calculation options.
        config['normal_method'] = 'analytic'

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

        # Generate the fine mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_crystal_mesh(
            self.param['mesh_size_a'], self.param['mesh_size_b'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_crystal_mesh(
            self.param['mesh_coarse_size_a'], self.param['mesh_coarse_size_b'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points
        self.param['mesh_coarse_normals'] = mesh_normals
        self.log.debug(f'Coarse mesh points: {mesh_points.shape[0]}')

        # Calculate height at width.
        self.param['width'] = np.max(
            self.param['mesh_points'][:,0])-np.min(self.param['mesh_points'][:,0])
        self.param['height'] = np.max(
            self.param['mesh_points'][:,1])-np.min(self.param['mesh_points'][:,1])
        self.log.debug(f"WxH: {self.param['width']:0.3f}x{self.param['height']:0.3f}")

    def torus(self, a, b, param, extra=False):
        C0 = param['crystal_location']
        C0_zaxis  = param['crystal_zaxis']
        C0_xaxis  = param['crystal_xaxis']
        maj_r     = param['major_radius']
        min_r     = param['minor_radius']

        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)
        O = C0 + maj_r * C0_zaxis

        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = O - maj_r * C_norm
        Q = C + C_norm * min_r

        axis = np.cross(C_norm, C0_yaxis)
        X_norm = xm.vector_rotate(C_norm, axis, b)
        X = Q - X_norm * min_r

        if extra:
            S = param['source_location']
            SC = C - S
            SC_dist = np.linalg.norm(SC)
            SC_hat = SC / SC_dist
            bragg = np.abs(np.pi / 2 - xm.vector_angle(SC_hat, C_norm))

            # Calculate the location where the rays intersect
            # the tangency circle. These will be out of focus.
            D2_dist = maj_r * np.cos(bragg)
            D2 = O + D2_dist * xm.vector_rotate(-1 * C_norm, C0_yaxis, bragg)
            CD2 = D2 - C
            CD2_dist = np.linalg.norm(CD2)
            CD2_hat = CD2 / CD2_dist

            # Calculate the location of approximate best focus.
            SQ = Q - S
            SQ_hat = SQ / np.linalg.norm(SQ)
            aQSC = np.arccos(np.dot(SQ_hat, SC_hat))
            CQ_dist = SC_dist * np.sin(aQSC)
            aDCQ = 2 * bragg - aQSC
            CD_dist = CQ_dist / np.sin(aDCQ)
            D = C + CD2_hat * CD_dist



            out = {}
            out['O'] = O
            out['C'] = C
            out['X'] = X
            out['Q'] = Q
            out['S'] = S
            out['D'] = D
            #out['D2'] = D2
            return out

        else:
            return X

    def torus_analytic(self, a, b, param):
        out = self.torus(a, b, param, extra=True)
        X = out['X']
        XQ = (out['Q'] - out['X'])
        XQ_hat = XQ/np.linalg.norm(XQ)
        return X, XQ_hat

    def torus_fd(self, a, b, param, delta=None):
        profiler.start('finite difference')
        if delta is None: delta = 1e-8

        xyz = self.torus(a, b, param)
        xyz1 = self.torus(a + delta, b, param)
        xyz2 = self.torus(a, b + delta, param)

        vec1 = xyz1 - xyz
        vec2 = xyz2 - xyz
        norm_fd = np.cross(vec1, vec2)
        norm_fd /= np.linalg.norm(norm_fd)
        profiler.stop('finite difference')
        return xyz, norm_fd

    def get_t_param(self):
        major_radius = self.param['major_radius']
        minor_radius = self.param['minor_radius']

        crystal_spacing = self.param['crystal_spacing']
        crystal_location = self.param['origin']
        crystal_zaxis = self.param['zaxis']
        crystal_xaxis = self.param['xaxis']
        crystal_yaxis = np.cross(crystal_xaxis, crystal_zaxis)

        lambda0 = self.param['lambda0']
        source_distance = self.param['source_distance']

        # Calculate the source location.
        bragg_angle = xm.bragg_angle(lambda0, crystal_spacing)
        self.log.debug(f'lambda0 : {lambda0:0.4f} Å')
        self.log.debug(f'bragg_angle: {np.degrees(bragg_angle):0.3f}°')
        source_location = (
            crystal_location
            + source_distance
            * xm.vector_rotate(
                crystal_zaxis
                ,crystal_yaxis
                ,(np.pi / 2 - bragg_angle))
            )

        param = {}
        param['crystal_location'] = crystal_location
        param['crystal_xaxis']    = crystal_xaxis
        param['crystal_zaxis']    = crystal_zaxis
        param['source_location']  = source_location
        param['major_radius']     = major_radius
        param['minor_radius']     = minor_radius

        return param

    def generate_crystal_mesh(self, mesh_size_a=None, mesh_size_b=None):
        """
        This method creates the meshgrid for the crystal
        """
        profiler.start('generate_crystal_mesh')
        major_radius = self.param['major_radius']
        minor_radius = self.param['minor_radius']

        t_param = self.get_t_param()

        # --------------------------------
        # Setup the basic grid parameters.

        a_range = self.param['range_a']
        b_range = self.param['range_b']
        a_span = a_range[1] - a_range[0]
        b_span = b_range[1] - b_range[0]

        # For now assume the mesh_size is in radians.
        num_a = a_span/mesh_size_a
        num_b = b_span/mesh_size_b
        num_a = np.ceil(num_a).astype(int)
        num_b = np.ceil(num_b).astype(int)

        self.log.debug(f'num_a, num_b: {num_a}, {num_b}, total: {num_a*num_b}')

        a = np.linspace(a_range[0], a_range[1], num_a)
        b = np.linspace(b_range[0], b_range[1], num_b)

        aa, bb = np.meshgrid(a, b, indexing='ij')
        xx = np.empty((num_a, num_b))
        yy = np.empty((num_a, num_b))
        zz = np.empty((num_a, num_b))

        normal_xx = np.empty((num_a, num_b))
        normal_yy = np.empty((num_a, num_b))
        normal_zz = np.empty((num_a, num_b))

        for ii_a in range(num_a):
            for ii_b in range(num_b):
                b = bb[ii_a, ii_b]
                a = aa[ii_a, ii_b]
                # Temporary for development.
                if self.param['normal_method'] == 'analytic':
                    xyz, norm = self.torus_analytic(a, b, t_param)
                elif self.param['normal_method'] == 'fd':
                    xyz, norm = self.torus_fd(a, b, t_param)
                else:
                    raise Exception(f"normal_method: {self.param['normal_method']} not supported")

                xx[ii_a, ii_b] = xyz[0]
                yy[ii_a, ii_b] = xyz[1]
                zz[ii_a, ii_b] = xyz[2]
                normal_xx[ii_a, ii_b] = norm[0]
                normal_yy[ii_a, ii_b] = norm[1]
                normal_zz[ii_a, ii_b] = norm[2]

        # Combine x & y arrays, add z dimension
        # points_2d = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        # tri = Delaunay(points_2d)
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T
        tri = Delaunay(angles_2d)


        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        normals = np.stack((normal_xx.flatten(), normal_yy.flatten(), normal_zz.flatten())).T

        faces = tri.simplices

        profiler.stop('generate_crystal_mesh')
        return points, normals, faces

