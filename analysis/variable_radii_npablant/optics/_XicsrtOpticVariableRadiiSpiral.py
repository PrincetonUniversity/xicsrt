# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
"""
import numpy as np
from scipy.spatial import Delaunay


from xicsrt.util import profiler
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import sinusoidal_spiral
from xicsrt.objects._GeometryObject import GeometryObject
from xicsrt.optics._XicsrtOpticCrystal import XicsrtOpticCrystal

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from xicsrt.tools import xicsrt_math_jax as xmj

class XicsrtOpticVariableRadiiSpiral(XicsrtOpticCrystal):

    def get_default_config(self):
        config = super().get_default_config()
        config['use_meshgrid'] = True
        config['mesh_size_a'] = 0.005
        config['mesh_size_b'] = 0.01
        config['mesh_coarse_size_a'] = 0.01
        config['mesh_coarse_size_b'] = 0.02
        config['range_a'] = [-0.01, 0.01]
        config['range_b']  = [-0.04, 0.04]

        config['r0'] = 1.0
        config['b'] = 0.356
        config['theta0'] = np.radians(25)

        config['sC'] = None
        config['thetaC'] = None
        config['phiC'] = 0.0

        # These are not used, but are added for compatibility
        # with the VR-Toroid and the Standard-Torus.
        config['major_radius'] = None
        config['minor_radius'] = None

        # Temporary for development.
        config['normal_method'] = 'jax'

        return config

    def get_spiral_inp(self):
        inp = {}
        inp['r0'] = None
        inp['b'] = None
        inp['theta0'] = None
        inp['sC'] = None
        inp['thetaC'] = None
        inp['phiC'] = None

        for key in inp:
            inp[key] = self.param[key]

        S = sinusoidal_spiral.get_source_origin(inp)
        inp['S'] = S

        return inp

    def setup_geometry(self):
        """
        Setup a geometry object to allow transformation of the
        spiral geometry to be centered on the crystal with the
        crystal normal as the z-axis. This is what is usually
        used when generating raytracing scenarios.
        """
        inp = self.get_spiral_inp()

        out = self.spiral(inp['phiC'], 0.0, inp, extra=True)
        config = {}
        config['origin'] = out['X']
        config['zaxis'] = (out['Q'] - out['X'])/np.linalg.norm(out['Q'] - out['X'])
        config['yaxis'] = np.array([0.0, 0.0, 1.0])
        config['xaxis'] = np.cross(config['zaxis'], config['yaxis'])
        self.geometry = GeometryObject(config, strict=False)

    def setup(self):
        super().setup()

        self.log.debug('Yo mama is here.')

        self.setup_geometry()

        # Generate the fine mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_mesh(
            self.param['mesh_size_a'], self.param['mesh_size_b'])
        # mesh_points_ext = self.point_to_external(mesh_points)
        self.param['mesh_faces'] = mesh_faces
        self.param['mesh_points'] = mesh_points
        self.param['mesh_normals'] = mesh_normals
        self.log.debug(f'Fine mesh points: {mesh_points.shape[0]}')

        # Generate the coarse mesh.
        mesh_points, mesh_normals, mesh_faces = self.generate_mesh(
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

    def spiral(self, *args, **kwargs):
        out = sinusoidal_spiral.spiral(*args, **kwargs)
        return out

    def spiral_centered(self, a, b, inp, extra=None):
        out = self.spiral(a, b, inp, extra=extra)
        if extra:
            for key in out:
                out[key] = self.geometry.point_to_local(out[key])
        else:
            out = self.geometry.point_to_local(out)
        return out

    def spiral_jax(self, a, b, inp):
        xyz = self.spiral(a, b, inp)
        dfda, dfdb = jax.jacfwd(self.spiral, (0, 1))(a, b, inp)
        norm_ad = np.cross(dfda, dfdb)
        norm_ad /= np.linalg.norm(norm_ad)
        return xyz, norm_ad

    def spiral_centered_jax(self, a, b, inp):
        xyz_ext, norm_ext = self.spiral_jax(a, b, inp)
        xyz_local = self.geometry.point_to_local(xyz_ext)
        norm_local = self.geometry.vector_to_local(norm_ext)
        return xyz_local, norm_local

    def generate_mesh(self, mesh_size_a=None, mesh_size_b=None):
        """
        This method creates the meshgrid for the crystal
        """
        profiler.start('generate_mesh')

        inp = self.get_spiral_inp()

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

        # ------------------------------------------------
        # Now calculate the xyz values at each grid point.
        for ii_a in range(num_a):
            for ii_b in range(num_b):

                a = aa[ii_a, ii_b]
                b = bb[ii_a, ii_b]

                # Temporary for development.
                if self.param['normal_method'] == 'jax':
                    xyz, norm = self.spiral_centered_jax(a, b, inp)
                elif self.param['normal_method'] == 'ideal_np':
                    out = self.spiral_centered(a, b, inp, extra=True)
                    xyz = out['X']
                    norm = (out['Q'] - out['X'])/np.linalg.norm(out['Q'] - out['X'])
                else:
                    raise Exception(f"normal_method {self.param['normal_method']} unknown.")

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

        profiler.stop('generate_mesh')
        return points, normals, faces

