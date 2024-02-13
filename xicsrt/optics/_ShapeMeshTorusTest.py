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
import math
from xicsrt.tools import xicsrt_math as xm
import plotly.graph_objects as go
from xicsrt.optics._ShapeMesh import ShapeMesh

@dochelper

class ShapeMeshTorusTest(ShapeMesh):


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
        config['angle_major1'] = [0,2*np.pi]
        config['angle_minor1'] = [0,2*np.pi] 
        config['surfaces'] = 0
        config['xsize'] = 0.040
        config['ysize'] = 0.080

        # When surface = 0 means the outer surface = concave/concave
        # When surface = 1 means the inner surface = concave/convex 
        # When surface = 2 means the inner surface = convex/concave
        # When surface = 3 means the outer surface = convex/convex
        
        # Parameters needed to define geometry.
        config['radius_major'] = 1.0
        config['radius_minor'] = 0.1
        config['normal_method'] = 'analytic'

        return config

    def setup(self):
        super().setup()
        self.log.debug('Yo mama was here.')

        # Generate the fine mesh.
        mesh_points, mesh_faces, mesh_normals, mesh_points1, mesh_faces1, mesh_normals1 = self.generate_mesh(self.param['mesh_size'])
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.param['mesh_faces1'] = mesh_faces1
        self.param['mesh_points1'] = mesh_points1
        self.param['mesh_normals1'] = mesh_normals1
        # Generate the coarse mesh.
        mesh_points, mesh_faces, mesh_normals,mesh_points1, mesh_faces1, mesh_normals1 = self.generate_mesh(self.param['mesh_coarse_size'])
        self.param['mesh_coarse_faces'] = mesh_faces
        self.param['mesh_coarse_points'] = mesh_points
        self.param['mesh_coarse_normals'] = mesh_normals
        self.param['mesh_coarse_faces1'] = mesh_faces
        self.param['mesh_coarse_points1'] = mesh_points
        self.param['mesh_coarse_normals1'] = mesh_normals
        width = np.max(
            self.param['mesh_points'][:,0])-np.min(self.param['mesh_points'][:,0])
        height = np.max(
            self.param['mesh_points'][:,1])-np.min(self.param['mesh_points'][:,1])
        width1 = np.max(
            self.param['mesh_points1'][:,0])-np.min(self.param['mesh_points1'][:,0])
        height1 = np.max(
            self.param['mesh_points1'][:,1])-np.min(self.param['mesh_points1'][:,1])

        # Calculate width and height of the optic.
#        self.param['xsize'] = np.max(
#            self.param['mesh_points'][:,0])-np.min(self.param['mesh_points'][:,0])
#        self.param['ysize'] = np.max(
#            self.param['mesh_points'][:,1])-np.min(self.param['mesh_points'][:,1])
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

    def torus1(self, a, b):
        """
        Return a 3D surface coordinate given a set of two angles.
        """
        C0 = self.param['origin']
        C0_zaxis  = self.param['zaxis']
        C0_xaxis  = self.param['xaxis']
        maj_r     = self.param['radius_major']
        min_r     = self.param['radius_minor']
        # This is concave/convex side 
        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)
        O = C0 + (maj_r ) * C0_zaxis 

        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = O - (maj_r )* C_norm
        Q = C - C_norm * ( min_r)

        axis = np.cross(C_norm, C0_yaxis)
        X_norm = xm.vector_rotate(C_norm, axis, b)
        X = Q + X_norm *min_r

        return X, X_norm

    def torus2(self, a, b):
        """
        Return a 3D surface coordinate given a set of two angles.
        """
        C0 = self.param['origin']
        C0_zaxis  = self.param['zaxis']
        C0_xaxis  = self.param['xaxis']
        maj_r     = self.param['radius_major']
        min_r     = self.param['radius_minor']

        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)
        O = C0 + (-maj_r ) * C0_zaxis

        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = O - (-maj_r)* C_norm
        Q = C + C_norm * ( min_r)

        axis = np.cross(C_norm, C0_yaxis)
        X_norm = xm.vector_rotate(C_norm, axis, b)
        X = Q - X_norm *min_r

        return X, X_norm
    
    def torus3(self, a, b):
        """
        Return a 3D surface coordinate given a set of two angles.
        """
        C0 = self.param['origin']
        C0_zaxis  = self.param['zaxis']
        C0_xaxis  = self.param['xaxis']
        maj_r     = self.param['radius_major']
        min_r     = self.param['radius_minor']
        C0_yaxis = np.cross(C0_xaxis, C0_zaxis)
        O = C0 + (-maj_r  ) * C0_zaxis

        C_norm = xm.vector_rotate(C0_zaxis, C0_yaxis, a)
        C = O - (-maj_r )* C_norm
        Q = C + C_norm * ( -min_r)

        axis = np.cross(C_norm, C0_yaxis)
        X_norm = xm.vector_rotate(C_norm, axis, b)
        X = Q + X_norm *min_r

        return X, X_norm
    
    def shape(self, a, b):
        if self.param['surfaces'] == 0:
            return self.torus(a, b)
        if self.param['surfaces'] == 1:
            return self.torus1(a, b)
        if self.param['surfaces'] == 2:
            return self.torus2(a, b)
        if self.param['surfaces'] == 3:
            return self.torus3(a, b)
        
    def shape_fd(self, a, b, delta=None):
        profiler.start('finite difference')
        if delta is None: delta = 1e-8
            
        if self.param['surfaces'] == 0:
            xyz, _ = self.torus(a, b)
            xyz1, _ = self.torus(a + delta, b)
            xyz2, _ = self.torus(a, b + delta)
        if self.param['surfaces'] == 1:
            xyz, _ = self.torus1(a, b)
            xyz1, _ = self.torus1(a + delta, b)
            xyz2, _ = self.torus1(a, b + delta)
        if self.param['surfaces'] == 2:
            xyz, _ = self.torus2(a, b)
            xyz1, _ = self.torus2(a + delta, b)
            xyz2, _ = self.torus2(a, b + delta)
        if self.param['surfaces'] == 3:
            xyz, _ = self.torus3(a, b)
            xyz1, _ = self.torus3(a + delta, b)
            xyz2, _ = self.torus3(a, b + delta)
            
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

        a_11 = self.param['xsize']/self.param['radius_major']
        b_11 = self.param['ysize']/self.param['radius_minor'] 
        a_11 = np.arcsin(a_11)
        b_11 = np.arcsin(b_11)
         
        a_range = [-a_11,a_11]
        b_range = [-b_11,b_11]
        a_range = self.param['angle_major']
        b_range = self.param['angle_minor']      
        a_range1 = self.param['angle_major1']
        b_range1 = self.param['angle_minor1']

        num_a = mesh_size[0]
        num_b = mesh_size[1]

        num_a1 = mesh_size[0]
        num_b1 = mesh_size[1]

        self.log.debug(f'num_a, num_b: {num_a}, {num_b}, total: {num_a*num_b}')

        a = np.linspace(a_range[0], a_range[1], num_a)

        b = np.linspace(b_range[0], b_range[1], num_b)

        a1 = np.linspace(a_range1[0], a_range1[1], num_a1)
        b1 = np.linspace(b_range1[0], b_range1[1], num_b1)
        
        xx, yy, zz, normal_xx, normal_yy, normal_zz = \
            self.calculate_mesh(a, b)

        xx1, yy1, zz1, normal_xx1, normal_yy1, normal_zz1 = \
            self.calculate_mesh(a1, b1)

        aa, bb = np.meshgrid(a, b, indexing='ij')
        angles_2d = np.stack((aa.flatten(), bb.flatten()), axis=0).T
        tri = Delaunay(angles_2d)

        aa1, bb1 = np.meshgrid(a1, b1, indexing='ij')
        angles_2d1 = np.stack((aa1.flatten(), bb1.flatten()), axis=0).T
        tri1 = Delaunay(angles_2d1)
        # It's also possible to triangulate using the x,y coordinates.
        # This is not recommended unless there is some specific need.
        #
        # points_2d = np.stack((xx.flatten(), yy.flatten()), axis=0).T
        # tri = Delaunay(points_2d)

        points = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T
        normals = np.stack((normal_xx.flatten(), normal_yy.flatten(), normal_zz.flatten())).T

        points1 = np.stack((xx1.flatten(), yy1.flatten(), zz1.flatten())).T
        normals1 = np.stack((normal_xx1.flatten(), normal_yy1.flatten(), normal_zz1.flatten())).T

        faces = tri.simplices
        faces1 = tri1.simplices
        
        lines = []
        for f in faces:
            lines.append([f[0], f[1]])
            lines.append([f[0], f[2]])
            lines.append([f[1], f[2]])
        lines = np.array(lines)
        lines = np.unique(lines, axis=0)

        x0 = points[lines[:, 0], 0]
        y0 = points[lines[:, 0], 1]
        z0 = points[lines[:, 0], 2]
        x1 = points[lines[:, 1], 0]
        y1 = points[lines[:, 1], 1]
        z1 = points[lines[:, 1], 2]
        nan = np.zeros(len(x0)) * np.nan

        # Add nans between each line which Scatter3D will use to define linebreaks.
        x = np.dstack((x0, x1, nan)).flatten()
        y = np.dstack((y0, y1, nan)).flatten()
        z = np.dstack((z0, z1, nan)).flatten()

        lines1 = []
        for f in faces1:
            lines1.append([f[0], f[1]])
            lines1.append([f[0], f[2]])
            lines1.append([f[1], f[2]])
        lines1 = np.array(lines1)
        lines1 = np.unique(lines1, axis=0)
        
        x01 = points1[lines1[:, 0], 0]
        y01 = points1[lines1[:, 0], 1]
        z01 = points1[lines1[:, 0], 2]
        x11 = points1[lines1[:, 1], 0]
        y11 = points1[lines1[:, 1], 1]
        z11 = points1[lines1[:, 1], 2]
        nan1 = np.zeros(len(x01)) * np.nan

        # Add nans between each line which Scatter3D will use to define linebreaks.
        x1 = np.dstack((x01, x11, nan1)).flatten()
        y1 = np.dstack((y01, y11, nan1)).flatten()
        z1 = np.dstack((z01, z11, nan1)).flatten()

        trace = go.Scatter3d(
            x=x
            , y=y
            , z=z
            , mode='lines'
            , line={'color': 'black'}
            , connectgaps=False
            , showlegend=False)
        trace1 = go.Scatter3d(
            x=x1
            , y=y1
            , z=z1
            , mode='lines'
            , line={'color': 'red'}
            , connectgaps=False
            , showlegend=False)
        fig = go.Figure(data = [trace,trace1])
        fig.show()
        
        profiler.stop('generate_mesh')
        return points, faces, normals, points1, faces1 , normals1


