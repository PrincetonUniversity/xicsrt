# -*- coding: utf-8 -*-
"""
Authors
-------
  - Novimir Pablant <npablant@pppl.gov>
  - James Kring <jdk0026@tigermail.auburn.edu>
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
"""
from PIL import Image
import numpy as np
import logging

from xicsrt.util import profiler

from xicsrt.xicsrt_objects import TraceObject

class XicsrtOpticGeneric(TraceObject):
    """
    A generic optical element. 
    Optical elements have a position and rotation in 3D space and a finite 
    extent. Additional properties, such as as crystal spacing, rocking curve, 
    and reflectivity, should be defined in derived classes.
    """
        
    def get_default_config(self):
        config = super().get_default_config()
        
        # boolean settings
        config['do_miss_check'] = False
        
        # spatial information
        config['width']          = 0.0
        config['height']         = 0.0
        config['depth']          = 0.0
        config['pixel_size']     = None
        config['pixel_width']    = None
        config['pixel_height']   = None  
        
        # mesh information
        config['use_meshgrid']   = False
        config['mesh_points']    = None
        config['mesh_faces']     = None

        # Temporary for doing some profiling of diferent
        # mesh intersect checks.
        config['mesh_method']    = 0

        return config

    def check_param(self):
        super().check_param()

        # Check the optic size compare to the meshgrid size.

        # This is temporary until handling of oversized meshes
        # are implemented.
        if self.param['use_meshgrid'] is True:
            mesh_loc = self.point_to_local(self.param['mesh_points'])

            # If any mesh points fall outside of the optic width, test fails.
            test = True
            test &= np.all(abs(mesh_loc[:, 0]) <= (self.param['width'] / 2))
            test &= np.all(abs(mesh_loc[:, 1]) <= (self.param['height'] / 2))
            if not test:
                raise Exception('Optic dimentions too small to contain meshgrid.')

    def initialize(self):
        super().initialize()

        # autofill pixel grid sizes
        if self.param['pixel_size'] is None:
            self.param['pixel_size'] = self.param['width']/100
        if self.param['pixel_width'] is None:
            self.param['pixel_width'] = int(np.ceil(self.param['width'] / self.param['pixel_size']))
        if self.param['pixel_height'] is None:
            self.param['pixel_height'] = int(np.ceil(self.param['height'] / self.param['pixel_size']))

        self.image = np.zeros((self.param['pixel_width'], self.param['pixel_height']))
        self.photon_count = 0

        if self.param['use_meshgrid'] is True:
            self.mesh_initialize()

    def light(self, rays):
        """
        This is the main method that is called to perform ray-tracing
        for this optic.  Different pathways are taken depending on
        whether a meshgrid is being used.
        """
        m = rays['mask']
        if self.param['use_meshgrid'] is False:
            distance = self.intersect(rays)
            X, rays  = self.intersect_check(rays, distance)
            self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
            normals  = self.generate_normals(X, rays)
            rays     = self.reflect_vectors(X, rays, normals)
            self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
        else:
            # Temporarily switch between mesh methods just so that
            # I can verify/debug that the new ones are working and
            # provide the expected performance boost.
            if self.param['mesh_method'] == 0:
                self.mesh_initialize()
                X, rays, hits = self.mesh_intersect_check_old(rays)
                self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
                normals = self.mesh_generate_normals_old(X, rays, hits)
                rays = self.reflect_vectors(X, rays, normals)
                self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
            elif self.param['mesh_method'] == 1:
                self.mesh_initialize()
                X, rays, hits = self.mesh_intersect_check(rays)
                self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
                normals  = self.mesh_generate_normals(X, rays, hits)
                rays     = self.reflect_vectors(X, rays, normals)
                self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
            elif self.param['mesh_method'] == 2:
                self.mesh_initialize()
                X, rays, hits = self.mesh_intersect_check_2(rays)
                self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
                normals  = self.mesh_generate_normals(X, rays, hits)
                rays     = self.reflect_vectors(X, rays, normals)
                self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
            elif self.param['mesh_method'] == 3:
                self.mesh_initialize()
                X, rays, hits = self.mesh_intersect_check_3(rays)
                self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
                normals = self.mesh_generate_normals(X, rays, hits)
                rays = self.reflect_vectors(X, rays, normals)
                self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))

        return rays

    def normalize(self, vector):
        magnitude = self.norm(vector)
        vector_norm = vector / magnitude[:, np.newaxis]
        return vector_norm
        
    def norm(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        return magnitude

    def distance_point_to_line(origin, normal, point):
        o = origin
        n = normal
        p = point
        t = np.dot(p - o, n) / np.dot(n, n)
        d = np.linalg.norm(np.outer(t, n) + o - p, axis=1)
        return d

    def intersect(self, rays):
        """
        Intersection with a plane.
        """
        
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m] = np.dot((self.param['origin'] - O[m]), self.param['zaxis']) / np.dot(D[m], self.param['zaxis'])

        # Update the mask to only include positive distances.
        m &= (distance >= 0)

        return distance
    
    def intersect_check(self, rays, distance):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X = np.zeros(O.shape, dtype=np.float64)
        X_local = np.zeros(O.shape, dtype=np.float64)
        
        # X is the 3D point where the ray intersects the optic
        # There is no reason to make a new X array here
        # instead of modifying O except to make debugging easier.
        X[m] = O[m] + D[m] * distance[m,np.newaxis]

        X_local[m] = self.point_to_local(X[m])
        
        # Find which rays hit the optic, update mask to remove misses
        if self.param['do_miss_check'] is True:
            m[m] &= (np.abs(X_local[m,0]) < self.param['width'] / 2)
            m[m] &= (np.abs(X_local[m,1]) < self.param['height'] / 2)

        return X, rays
    
    def generate_normals(self, X, rays):
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        normals[m] = self.param['zaxis']
        return normals
    
    def reflect_vectors(self, X, rays, normals=None):
        """
        Generic optic has no reflection, rays just pass through.
        """
        O = rays['origin']
        O[:]  = X[:]
        
        return rays

    def mesh_initialize(self):
        """
        A precalculation of the mesh_centers and mesh_normals
        which are needed in the other mesh methods.
        """
        profiler.start('mesh_initialize')

        # Just for clarity.
        points = self.param['mesh_points']
        faces  = self.param['mesh_faces']

        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        p0 = points[faces[:, 0], :]
        p1 = points[faces[:, 1], :]
        p2 = points[faces[:, 2], :]

        centers = np.mean(np.array([p0, p1, p2]), 0)
        normals = np.cross((p0 - p1), (p2 - p1))
        normals /= np.linalg.norm(normals, axis=1)[:, None]

        self.param['mesh_centers'] = centers
        self.param['mesh_normals'] = normals

        profiler.stop('mesh_initialize')

    def mesh_intersect_check(self, rays):
        """
        Calculate the intersections between the rays and the mesh.
        Also save the mesh faces that each ray hit.

        Programming Notes
        -----------------
          This method should be quite CPU efficient, but requires
          a significant amount of memory. If the computational
          environment is memory limited, it may be worth re-writing
          this method with a loop over mesh faces to avoid very large
          arrays.
        """
        profiler.start('mesh_intersect_check')
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        X = np.zeros(D.shape, dtype=np.float64)
        hits = np.zeros(m.shape, dtype=np.int)

        points = self.param['mesh_points']
        faces  = self.param['mesh_faces']
        centers = self.param['mesh_centers']
        n = self.param['mesh_normals']

        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        p0 = points[faces[:, 0], :]
        p1 = points[faces[:, 1], :]
        p2 = points[faces[:, 2], :]

        # distance = np.dot((p0 - O), n) / np.dot(D, n)
        t0 = p0[:, None, :] - O[None, :, :]
        t1 = np.einsum('ijk, ik -> ij', t0, n)
        t2 = np.einsum('jk, ik -> ij', D, n)
        dist = t1 / t2
        t3 = np.einsum('jk,ij -> ijk', D, dist)
        intersect = t3 + O

        tri_area = np.linalg.norm(np.cross((p0 - p1), (p0 - p2)), axis=1)
        alpha = np.linalg.norm(np.cross((intersect - p1[:, None, :]), (intersect - p2[:, None, :])), axis=2)
        beta = np.linalg.norm(np.cross((intersect - p2[:, None, :]), (intersect - p0[:, None, :])), axis=2)
        gamma = np.linalg.norm(np.cross((intersect - p0[:, None, :]), (intersect - p1[:, None, :])), axis=2)
        diff = alpha + beta + gamma - tri_area[:, None]

        # For now hard code the floating point tolerance.
        # A better way of handling floating point errors is needed.
        test = (diff < 1e-15)
        test &= (dist >= 0)

        # Update the mask not to include any rays that didn't hit the mesh.
        m &= np.any(test, axis=0)

        # Assume that the mesh is well behaved, and that each ray
        # should only hit one face. In this case we can just use argmax.
        hits[m] = np.argmax(test[:, m], axis=0)

        idx_rays = np.arange(len(m))
        X[m] = intersect[hits[m], idx_rays[m], :]

        profiler.stop('mesh_intersect_check')
        return X, rays, hits

    def triangle_area_3d(self, p0, p1, p2):
        cross = np.empty(p0.shape, dtype=np.float64)

        profiler.start('vector cross_product')
        cross[...,0] = (p0-p1)[...,1]*(p0-p2)[...,2] - (p0-p1)[...,2]*(p0-p2)[...,1]
        cross[...,1] = (p0-p1)[...,2]*(p0-p2)[...,0] - (p0-p1)[...,0]*(p0-p2)[...,2]
        cross[...,2] = (p0-p1)[...,0]*(p0-p2)[...,1] - (p0-p1)[...,1]*(p0-p2)[...,0]
        profiler.stop('vector cross_product')
        profiler.start('vector norm')
        area = np.sqrt(cross[...,0]**2 + cross[...,1]**2 + cross[...,2]**2)
        profiler.stop('vector norm')
        return area

    def mesh_intersect_check_2(self, rays):
        """
        Calculate the intersections between the rays and the mesh.
        Also save the mesh faces that each ray hit.

        Programming Notes
        -----------------
          This is a second version of the intersect check code that
          includes a loop over the mesh faces. This is much more
          memory efficient, and ultimately faster than the method
          with the very large arrays, but still quite slow.
        """
        profiler.start('mesh_intersect_check_2')
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        X = np.zeros(D.shape, dtype=np.float64)
        hits = np.zeros(m.shape, dtype=np.int)

        points = self.param['mesh_points']
        faces  = self.param['mesh_faces']
        centers = self.param['mesh_centers']
        n = self.param['mesh_normals']

        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        p0 = points[faces[:, 0], :]
        p1 = points[faces[:, 1], :]
        p2 = points[faces[:, 2], :]

        m_temp = np.zeros(m.shape, dtype=np.bool)
        for ii in range(faces.shape[0]):
            profiler.start('mesh_2 inner loop')
            profiler.start('mesh_2 inner loop: 1')
            # distance = np.dot((p0 - O), n) / np.dot(D, n)
            dist = np.dot((p0[ii, :] - O), n[ii, :]) / np.dot(D, n[ii, :])
            intersect = D * dist[:, None] + O
            profiler.stop('mesh_2 inner loop: 1')
            profiler.start('mesh_2 inner loop: 2')

            #tri_area = np.linalg.norm(np.cross((p0[ii, :] - p1[ii, :]), (p0[ii, :] - p2[ii, :])))
            #alpha = np.linalg.norm(np.cross((intersect - p1[ii, :]), (intersect - p2[ii, :])), axis=1)
            #beta = np.linalg.norm(np.cross((intersect - p2[ii, :]), (intersect - p0[ii, :])), axis=1)
            #gamma = np.linalg.norm(np.cross((intersect - p0[ii, :]), (intersect - p1[ii, :])), axis=1)
            #diff = alpha + beta + gamma - tri_area

            #diff = (np.linalg.norm(np.cross((intersect - p1[ii, :]), (intersect - p2[ii, :])), axis=1)
            #        + np.linalg.norm(np.cross((intersect - p2[ii, :]), (intersect - p0[ii, :])), axis=1)
            #        + np.linalg.norm(np.cross((intersect - p0[ii, :]), (intersect - p1[ii, :])), axis=1)
            #        - np.linalg.norm(np.cross((p0[ii, :] - p1[ii, :]), (p0[ii, :] - p2[ii, :])))
            #        )

            a = intersect - p0[ii, :]
            b = intersect - p1[ii, :]
            c = intersect - p2[ii, :]

            diff = (np.linalg.norm(np.cross(b, c), axis=1)
                    + np.linalg.norm(np.cross(c, a), axis=1)
                    + np.linalg.norm(np.cross(a, b), axis=1)
                    - np.linalg.norm(np.cross((p0[ii, :] - p1[ii, :]), (p0[ii, :] - p2[ii, :])))
                    )

            #diff = (self.triangle_area_3d(intersect, p1[ii, :], p2[ii, :])
            #        + self.triangle_area_3d(intersect, p2[ii, :], p0[ii, :])
            #        + self.triangle_area_3d(intersect, p0[ii, :], p1[ii, :])
            #        - self.triangle_area_3d(p0[ii, :], p1[ii, :], p2[ii, :])
            #        )
            profiler.stop('mesh_2 inner loop: 2')
            test = (diff < 1e-15)
            test &= (dist >= 0)
            m_temp |= test
            profiler.stop('mesh_2 inner loop')
            # Assume that the mesh is well behaved, and that each ray
            # should only hit one face. In this case we can just use argmax.
            hits[test] = ii
            X[m_temp] = intersect[m_temp, :]

        # Update the mask not to include any rays that didn't hit the mesh.
        m &= m_temp

        profiler.stop('mesh_intersect_check_2')
        return X, rays, hits

    def _dist_point_to_line(self, O, D, p):
        # Here I am assuming that D is normalized.
        t0 = p[:, None, :] - O[None, :, :]
        t = np.einsum('ijk, jk -> ij', t0, D)
        d = np.linalg.norm(np.einsum('ij, jk -> ijk', t, D) + O[None, :, :] - p[:, None, :], axis=2)
        return d

    def mesh_intersect_check_3(self, rays):
        """
        Calculate the intersections between the rays and the mesh.
        Also save the mesh faces that each ray hit.

        Programming Notes
        -----------------
          This is a second version of the intersect check code that
          includes a loop over the mesh faces. This is much more
          memory efficient, and ultimately faster than the method
          with the very large arrays, but still quite slow.
        """
        print('mesh_intersect_check')
        profiler.start('mesh_intersect_check_3')
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        X = np.zeros(D.shape, dtype=np.float64)
        hits = np.zeros(m.shape, dtype=np.int)

        points = self.param['mesh_points']
        faces = self.param['mesh_faces']
        centers = self.param['mesh_centers']
        normals = self.param['mesh_normals']


        profiler.start('mesh_3 find distance')
        # Calculate the distance between the rays and each of the
        # mesh points.  This is a big array.
        point_dist = self._dist_point_to_line(O, D, points)
        ii_point = np.argmin(point_dist, axis=0)
        profiler.stop('mesh_3 find distance')

        profiler.start('mesh_3 find faces')
        # Find all of the faces that include the closest point.
        # For a rectangular grid this should never be more than
        # eight. (I have not verified this.)
        #
        # At the moment I can't think of a way to do this without
        # looping over the rays, but I feel like there must be a
        # more efficient way
        faces_idx = np.zeros((8, len(m)), dtype=np.int)
        faces_near = np.zeros((8, len(m), 3), dtype=np.int)
        faces_mask = np.zeros((8, len(m)), dtype=np.bool)
        faces_num = np.zeros(len(m), dtype=np.int)
        for jj in range(len(m)):
            #profiler.start('mesh_3 inner loop')
            ii_faces = np.nonzero(faces == ii_point[jj])[0]
            faces_num[jj] = len(ii_faces)
            faces_idx[:faces_num[jj], jj] = ii_faces
            faces_mask[:faces_num[jj], jj] = True
            #profiler.stop('mesh_3 inner loop')
        faces_near  = faces[faces_idx,:]

        profiler.stop('mesh_3 find faces')
        profiler.start('mesh_3 intersect')
        # Copying these makes the code easier to read,
        # but may increase memory usage for dense meshes.
        p0 = points[faces_near[:, :, 0], :]
        p1 = points[faces_near[:, :, 1], :]
        p2 = points[faces_near[:, :, 2], :]

        n = normals[faces_idx]

        # distance = np.dot((p0 - O), n) / np.dot(D, n)
        t0 = p0[:, :, :] - O[None, :, :]
        t1 = np.einsum('ijk, ijk -> ij', t0, n)
        t2 = np.einsum('jk, ijk -> ij', D, n)
        dist = t1 / t2
        t3 = np.einsum('jk,ij -> ijk', D, dist)
        intersect = t3 + O

        tri_area = np.linalg.norm(np.cross((p0 - p1), (p0 - p2)), axis=2)
        alpha = np.linalg.norm(np.cross((intersect - p1), (intersect - p2)), axis=2)
        beta = np.linalg.norm(np.cross((intersect - p2), (intersect - p0)), axis=2)
        gamma = np.linalg.norm(np.cross((intersect - p0), (intersect - p1)), axis=2)
        diff = alpha + beta + gamma - tri_area

        # For now hard code the floating point tolerance.
        # A better way of handling floating point errors is needed.
        test = (diff < 1e-15) & (dist >= 0) & faces_mask
        m &= np.any(test, axis=0)
        hits[m] = np.argmax(test[:, m], axis=0)

        idx_rays = np.arange(len(m))
        X[m] = intersect[hits[m], idx_rays[m], :]
        profiler.stop('mesh_3 intersect')
        profiler.stop('mesh_intersect_check_3')
        return X, rays, hits

    def mesh_triangulate(self, ii):
        points = self.param['mesh_points']
        faces  = self.param['mesh_faces']
        
        # Find which points belong to the triangle face
        p1 = points[faces[ii,0],:]
        p2 = points[faces[ii,1],:]
        p3 = points[faces[ii,2],:]

        # Calculate the centerpoint and normal of the triangle face
        p0 = np.mean(np.array([p1, p2, p3]), 0)
        n  = np.cross((p1 - p2),(p3 - p2))
        n /= np.linalg.norm(n)
        
        #compact the triangle data into a dictionary for easy movement
        tri = dict()
        tri['center'] = p0
        tri['point1'] = p1
        tri['point2'] = p2
        tri['point3'] = p3
        tri['normal'] = n
        
        return tri

    def mesh_generate_normals(self, X, rays, hits):
        m = rays['mask']
        mesh_normals = self.param['mesh_normals']

        normals = np.zeros(rays['origin'].shape, dtype=np.float64)
        normals[m] = mesh_normals[hits[m],:]
        return normals

    def mesh_intersect_check_old(self, rays):
        """
        Calculate the intersections between the rays and the mesh.
        Also save the mesh faces that each ray hit.

        Programming Notes
        -----------------
          This is the original version that Eugene wrote. As far as
          I know this is accurate, but very inefficient.
        """
        profiler.start('mesh_intersect_check_old')
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X     = np.zeros(D.shape, dtype=np.float64)
        hits  = np.zeros(m.shape, dtype=np.int)

        # Loop over each triangular face to find which rays hit
        for ii in range(len(self.param['mesh_faces'])):
            intersect= np.zeros(O.shape, dtype=np.float64)
            distance = np.zeros(m.shape, dtype=np.float64)
            test     = np.zeros(m.shape, dtype=np.bool)
            
            # Query the triangle mesh grid
            tri = self.mesh_triangulate(ii)
            p0= tri['center']
            p1= tri['point1']
            p2= tri['point2']
            p3= tri['point3']
            n = tri['normal']

            # Find the intersection point between the rays and triangle plane
            distance     = np.dot((p0 - O), n) / np.dot(D, n)
            intersect[m] = O[m] + D[m] * distance[m,np.newaxis]
            
            # Test to see if the intersection is inside the triangle
            # uses barycentric coordinate math (compare parallelpiped areas)
            tri_area = np.linalg.norm(np.cross((p1 - p2),(p1 - p3)))
            alpha    = self.norm(np.cross((intersect - p2),(intersect - p3)))
            beta     = self.norm(np.cross((intersect - p3),(intersect - p1)))
            gamma    = self.norm(np.cross((intersect - p1),(intersect - p2)))

            # This test uses an explicit tolerance to account for
            # floating-point errors in the area calculations.
            #
            # It would be better if a test could be found that does not
            # require this explicit tolerance.
            test |= np.less_equal((alpha + beta + gamma - tri_area), 1e-15)
            test &= (distance >= 0)
            
            # Append the results to the global impacts arrays
            X[test]    = intersect[test]
            hits[test] = ii + 1
        
        #mask all the rays that missed all faces
        if self.param['do_miss_check'] is True:
            m[m] &= (hits[m] != 0)

        profiler.stop('mesh_intersect_check_old')

        return X, rays, hits

    def mesh_generate_normals_old(self, X, rays, hits):
        profiler.start('mesh_generate_normals_old')
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        for ii in range(len(self.param['mesh_faces'])):
            tri = self.mesh_triangulate(ii)
            test = np.equal(ii, (hits - 1))
            test &= m
            normals[test] = tri['normal']

        profiler.stop('mesh_generate_normals_old')
        return normals

    def make_image(self, rays):
        """
        Collect the rays that his this optic into a pixel array that can be used
        for further analysis or visualization.

        Programming Notes
        -----------------

        It is important thas this calculation is compatible with intersect_check
        in terms of floating point errors.  The simple way to achive this is
        to ensure that both use the same calculation method.
        """
        image = np.zeros((self.param['pixel_width'], self.param['pixel_height']))
        X = rays['origin']
        m = rays['mask'].copy()
        
        num_lines = np.sum(m)
        
        # Add the ray hits to the pixel array
        if num_lines > 0:
            # Transform the intersection coordinates from external coordinates
            # to local optical 'pixel' coordinates.
            point_loc = self.point_to_local(X[m])
            pix = point_loc / self.param['pixel_size']
                
            # Convert from pixels to channels.
            # The channel coordinate is defined from the *center* of the bottom left
            # pixel. The pixel coordinate is define from the geometrical center of
            # the detector (this could be in the middle of or in between pixels).
            channel = np.zeros(pix.shape)
            channel[:,0] = pix[:,0] + (self.param['pixel_width'] - 1)/2
            channel[:,1] = pix[:,1] + (self.param['pixel_height'] - 1)/2
            
            # Bin the channels into integer values so that we can use them as
            # indexes into the image. Keep in mind that channel coordinate
            # system is defined from the center of the pixel.
            channel = np.round(channel).astype(int)
            
            # Check for any hits that are outside of the image.
            # These are possible due to floating point calculations.
            m = np.ones(num_lines, dtype=bool)
            m &= channel[:,0] >= 0
            m &= channel[:,0] < self.param['pixel_width']
            m &= channel[:,1] >= 0
            m &= channel[:,1] < self.param['pixel_height']
            num_out = np.sum(~m)
            if num_out > 0:
                logging.warning('Rays found outside of pixel grid ({}).'.format(num_out))
            
            # I feel like there must be a faster way to do this than to loop over
            # every intersection.  This could be slow for large arrays.
            for ii in range(num_lines):
                if m[ii]:
                    image[channel[ii,0], channel[ii,1]] += 1

        return image
    
    def collect_rays(self, rays):
        """
        Perform ongoing collection into an internal image.
        """
        image = self.make_image(rays)
        self.image[:,:] += image
        
        return self.image[:,:] 
        
    def output_image(self, image_name, rotate=None):
        if rotate:
            out_array = np.rot90(self.image)
        else:
            out_array = self.image
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)
