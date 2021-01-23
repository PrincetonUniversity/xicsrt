# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
import numpy as np

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._XicsrtOpticGeneric import XicsrtOpticGeneric

from scipy.spatial import cKDTree
from scipy.interpolate import CloughTocher2DInterpolator as Interpolator

@dochelper
class XicsrtOpticMesh(XicsrtOpticGeneric):
    """
    A optic that can (optionally) use a mesh grid instead of an analytical
    shape.  The use of the grid is controlled by the config option
    'use_meshgrid'.

    This object can be considered to be 'generic' and it is appropriate for all
    other optics to inherit this object instead of :any:`XicsrtOpticGeneric`.

    **Programming Notes**

    Raytracing of mesh optics is fundamentally slow, because of the need to
    find which mesh face is intersected by each ray.  For the simplest
    implementations this requires testing each ray against each mesh face
    leading to the speed scaling as the number of mesh faces
    (or equivilently mesh_density^2).

    Some optimization of the basic calculation been completed. The
    mesh_intersect_1 method implements the Möller–Trumbore algorithm and is
    the fastest pure python algorithm found so far. Other variations that have
    been tried are saved in the archive folder, along with some documentation
    on performance.

    To further improve performance this class (optionally) also makes use of
    pre-selection with a course mesh. First the intersection with the course
    mesh is found for each ray. Then only the 8 nearby faces on the full mesh
    are checked for the final intersection location. This method improves
    the performance to (num_faces_course + 8).

    The current algorithm for pre-selection (mesh refinement) is not perfect
    in that the nearby faces are not always appropriately chosen leading to a
    small number of rays being 'lost'. These errors can be minimized by
    increasing the resolution of the course mesh and ensuring that the grid
    spacing is approximately equal in the x and y directions.

    Further performance improvement could be gained by using numba or jax.
    This would allow the Möller–Trumbore algorithm to be implemented as a
    loop (instead of being vectorized) where the calculation can be terminated
    early when no hit is found.

    .. Todo::
      XicsrtOpticMesh: Improve the pre-selection (mesh refinement algorithm) to
      eliminate ray losses. The current method is as follows:

      1. Calculate intersection with course grid.
      2. Find the point on the fine grid closest to the intersection.
      3. Test all faces on the fine grid that contain this point.

      The problem is that the closest point may not always be part of the face
      that actually has the intersection. This can happen if the fine and course
      grid have very different densities, but also even in the perfect case if
      the ray hits near the edge of a face and the grid density is not even in
      the x and y directions.

      What is needed is a better selection of nearby faces. There is also a
      potential to improve computational speed slightly by testing fewer faces
      on the fine grid.
    """

    def default_config(self):
        config = super().default_config()

        # mesh information
        config['use_meshgrid'] = False

        config['mesh_points'] = None
        config['mesh_faces'] = None
        config['mesh_normals'] = None

        config['mesh_coarse_points'] = None
        config['mesh_coarse_faces'] = None
        config['mesh_coarse_normals'] = None

        config['mesh_interpolate'] = True
        config['mesh_refine'] = None

        return config

    def check_param(self):
        super().check_param()

        if self.param['mesh_normals'] is None:
            self.param['mesh_interpolate'] = False

        if self.param['mesh_refine'] is None:
            if self.param['mesh_coarse_points'] is not None:
                self.param['mesh_refine'] = True

    def initialize(self):
        super().initialize()

        if self.param['use_meshgrid'] is True:
            self.mesh_initialize()

    def trace(self, rays):
        """
        This is the main method that is called to perform ray-tracing
        for this optic.  Different pathways are taken depending on
        whether a meshgrid is being used.
        """

        if self.param['use_meshgrid'] is False:
            rays = super().trace(rays)
        else:
            if not self.param['mesh_refine']:
                X, rays, hits = self.mesh_intersect_1(rays, self.param['mesh'])
                self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, np.sum(rays['mask'])))
            else:
                X_c, rays, hits_c = self.mesh_intersect_1(rays, self.param['mesh_coarse'])
                num_rays_coarse = np.sum(rays['mask'])
                faces_idx, faces_mask = self.find_near_faces(X_c, self.param['mesh'])
                X, rays, hits = self.mesh_intersect_2(
                    rays
                    ,self.param['mesh']
                    ,faces_idx
                    ,faces_mask)
                num_rays_fine = np.sum(rays['mask'])

                num_rays_lost = num_rays_coarse - num_rays_fine
                if not num_rays_lost == 0:
                    self.log.warning(f'Rays lost in mesh refinement: {num_rays_lost:0.0f} of {num_rays_coarse:6.2e}')

            self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, np.sum(rays['mask'])))
            if self.param['mesh_interpolate']:
                X, normals = self.mesh_interpolate(X, self.param['mesh'], rays['mask'])
            else:
                normals = self.mesh_normals(hits, self.param['mesh'])
            rays = self.reflect_vectors(X, rays, normals)
            self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, np.sum(rays['mask'])))

        return rays

    def mesh_interpolate(self, X, mesh, mask):
        profiler.start('mesh_interpolate')
        X[:, 2] = mesh['interp']['z'](X[:, 0], X[:, 1])

        normals = np.empty(X.shape)
        normals[:, 0] = mesh['interp']['normal_x'](X[:, 0], X[:, 1])
        normals[:, 1] = mesh['interp']['normal_y'](X[:, 0], X[:, 1])
        normals[:, 2] = mesh['interp']['normal_z'](X[:, 0], X[:, 1])

        profiler.start('normalize')
        normals = np.einsum(
            'i,ij->ij'
            ,1.0/np.linalg.norm(normals, axis=1)
            ,normals
            ,optimize=True)
        profiler.stop('normalize')

        profiler.stop('mesh_interpolate')
        return X, normals

    def _mesh_precalc(self, points, normals, faces):
        profiler.start('_mesh_precalc')

        output = {}
        output['faces'] = faces
        output['points'] = points
        output['normals'] = normals

        if self.param['mesh_interpolate']:
            # Create a set of interpolators.
            # For now create these in 2D using the x and y locations.
            # This will not make sense for all geometries. In principal
            # the interpolation could be done in 3D, but this needs more
            # investigation and will ultimately be less accurate.
            #
            # It is recommended that mesh optics be built using a local
            # coordinate system that makes the x and y coordinates sensible
            # for 2d interpolation.
            profiler.start('Create Interpolators')
            interp = {}
            output['interp'] = interp
            interp['z'] = Interpolator(points[:, 0:2], points[:, 2].flatten())
            interp['normal_x'] = Interpolator(points[:, 0:2], normals[:, 0].flatten())
            interp['normal_y'] = Interpolator(points[:, 0:2], normals[:, 1].flatten())
            interp['normal_z'] = Interpolator(points[:, 0:2], normals[:, 2].flatten())
            profiler.stop('Create Interpolators')

        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        p0 = points[faces[..., 0], :]
        p1 = points[faces[..., 1], :]
        p2 = points[faces[..., 2], :]

        # Calculate the normals at each face.
        faces_center = np.mean(np.array([p0, p1, p2]), 0)
        faces_normal = np.cross((p0 - p1), (p2 - p1))
        faces_normal /= np.linalg.norm(faces_normal, axis=1)[:, None]
        output['faces_center'] = faces_center
        output['faces_normal'] = faces_normal

        # Generate a tree for the points.
        points_tree = cKDTree(points)
        output['points_tree'] = points_tree

        # Build up a lookup table for the faces around each point.
        # This is currently slow for large arrays.
        points_idx = np.arange(len(points))
        p_faces_idx, p_faces_mask = \
            self.find_point_faces(points_idx, faces)
        output['p_faces_idx'] = p_faces_idx
        output['p_faces_mask'] = p_faces_mask

        # centers_tree = cKDTree(points)
        # output['centers_tree'] = centers_tree

        profiler.stop('_mesh_precalc')
        return output

    def mesh_initialize(self):
        """
        Pre-calculate a number of mesh properties that are
        needed in the other mesh methods.
        """
        profiler.start('mesh_initialize')

        dummy = self._mesh_precalc(
            self.param['mesh_points']
            , self.param['mesh_normals']
            , self.param['mesh_faces'])
        self.param['mesh'] = {}
        for key in dummy:
            self.param['mesh'][key] = dummy[key]

        if self.param['mesh_coarse_points'] is not None:
            dummy = self._mesh_precalc(
                self.param['mesh_coarse_points']
                , self.param['mesh_coarse_normals']
                , self.param['mesh_coarse_faces'])
            self.param['mesh_coarse'] = {}
            for key in dummy:
                self.param['mesh_coarse'][key] = dummy[key]

        profiler.stop('mesh_initialize')

    def mesh_intersect_1(self, rays, mesh):
        """
        Find the intersection of rays with the mesh using the Möller–Trumbore
        algorithm.
        """
        profiler.start('mesh_intersect_1')
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        X = np.zeros(D.shape, dtype=np.float64)

        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        p0 = mesh['points'][mesh['faces'][..., 0], :]
        p1 = mesh['points'][mesh['faces'][..., 1], :]
        p2 = mesh['points'][mesh['faces'][..., 2], :]

        epsilon = 1e-15

        num_rays = len(m)
        hits = np.empty(num_rays, dtype=np.int)
        m_temp = np.empty(num_rays, dtype=bool)
        m_temp_2 = np.zeros(num_rays, dtype=bool)

        for ii in range(mesh['faces'].shape[0]):
            m_temp[:] = m
            edge1 = p1[ii, :] - p0[ii, :]
            edge2 = p2[ii, :] - p0[ii, :]
            h = np.cross(D[m_temp], edge2)
            f = np.einsum('i,ji->j', edge1, h, optimize=True)
            m_temp &= ~((f > -epsilon) & (f < epsilon))
            if not np.any(m_temp):
                continue

            f = 1.0 / f
            s = O - p0[ii, :]
            u = f * np.einsum('ij,ij->i', s, h, optimize=True)
            m_temp &= ~((u < 0.0) | (u > 1.0))
            if not np.any(m_temp):
                continue

            q = np.cross(s, edge1)
            v = f * np.einsum('ij,ij->i', D, q, optimize=True)
            m_temp &= ~((v < 0.0) | (u + v > 1.0))
            if not np.any(m_temp):
                continue

            t = f * np.einsum('i,ji->j', edge2, q, optimize=True)

            # Update overall hit array and hit mask.
            m_temp_2[m_temp] = m_temp[m_temp]
            hits[m_temp] = ii
            X[m_temp] = O[m_temp] + t[m_temp, None] * D[m_temp, :]

        # Update the mask not to include any rays that didn't hit the mesh.
        m &= m_temp_2
        profiler.stop('mesh_intersect_1')

        return X, rays, hits

    def mesh_intersect_2(
            self
            , rays
            , mesh
            , faces_idx
            , faces_mask):
        """
        Check for ray intersection with a list of mesh faces.

        Programming Notes
        -----------------
        Because of the mesh indexing, the arrays here have different
        dimensions than in mesh_intersect_1, and need a different
        vectorization.

        At the moment I am using an less efficient mesh intersection
        method. This should be updated to use the same method as
        mesh_intersect_1, but with the proper vectorization.
        """

        profiler.start('mesh_intersect_2')

        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        num_rays = len(m)

        X = np.zeros(D.shape, dtype=np.float64)
        hits = np.empty(num_rays, dtype=np.int)
        epsilon = 1e-15

        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        faces = mesh['faces'][faces_idx]
        p0 = mesh['points'][faces[..., 0], :]
        p1 = mesh['points'][faces[..., 1], :]
        p2 = mesh['points'][faces[..., 2], :]
        n = mesh['faces_normal'][faces_idx]

        # distance = np.dot((p0 - O), n) / np.dot(D, n)
        t0 = p0[:, :, :] - O[None, :, :]
        t1 = np.einsum('ijk, ijk -> ij', t0, n, optimize=True)
        t2 = np.einsum('jk, ijk -> ij', D, n, optimize=True)
        dist = t1 / t2
        t3 = np.einsum('jk,ij -> ijk', D, dist, optimize=True)
        intersect = t3 + O

        # Pre-calculate some vectors and do the calculation in a
        # single step in an attempt to optimize this calculation.
        a = intersect - p0
        b = intersect - p1
        c = intersect - p2

        diff = (np.linalg.norm(np.cross(b, c), axis=2)
                + np.linalg.norm(np.cross(c, a), axis=2)
                + np.linalg.norm(np.cross(a, b), axis=2)
                - np.linalg.norm(np.cross((p0 - p1), (p0 - p2)), axis=2)
                )

        # For now hard code the floating point tolerance.
        # A better way of handling floating point errors is needed.
        test = (diff < 1e-10) & (dist >= 0) & faces_mask
        m &= np.any(test, axis=0)

        # idx_hits tells us which of the 8 faces had a hit.
        idx_hits = np.argmax(test[:, m], axis=0)
        # Now index the faces_idx to git the actual face number.
        hits[m] = faces_idx[idx_hits, m]
        X[m] = intersect[idx_hits, m, :]

        profiler.stop('mesh_intersect_2')

        return X, rays, hits

    def mesh_normals(self, hits, mesh):
        normals = mesh['faces_normal'][hits, :]
        return normals

    def mesh_get_index(self, hits, faces):
        """
        Match faces to face indexes, with a loop over faces.
        """
        profiler.start('mesh_get_index')
        idx_hits = np.empty(hits.shape[0], dtype=np.int)
        for ii, ff in enumerate(faces):
            m_temp = np.all(np.equal(ff, hits), axis=1)
            idx_hits[m_temp] = ii
        profiler.stop('mesh_get_index')
        return idx_hits

    def find_point_faces(self, p_idx, faces, mask=None):
        """
        Find all of the the faces that include a given point.
        """
        profiler.start('find_point_faces')
        if mask is None:
            mask = np.ones(p_idx.shape, dtype=np.bool)
        m = mask
        p_faces_idx = np.zeros((8, len(m)), dtype=np.int)
        p_faces_mask = np.zeros((8, len(m)), dtype=np.bool)
        for ii_p in p_idx:
            ii_f = np.nonzero(np.equal(faces, p_idx[ii_p]))[0]
            faces_num = len(ii_f)
            p_faces_idx[:faces_num, ii_p] = ii_f
            p_faces_mask[:faces_num, ii_p] = True
        profiler.stop('find_point_faces')
        return p_faces_idx, p_faces_mask

    def find_near_faces(self, X, mesh):
        profiler.start('find_near_faces')
        idx = mesh['points_tree'].query(X)[1]
        faces_idx = mesh['p_faces_idx'][:, idx]
        faces_mask = mesh['p_faces_mask'][:, idx]
        profiler.stop('find_near_faces')
        return faces_idx, faces_mask
