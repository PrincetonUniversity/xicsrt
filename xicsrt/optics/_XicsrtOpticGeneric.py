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

        return config

    def check_config(self):
        super().check_config()

    def initialize(self):
        super().initialize()
        
        # Check the optic size compare to the meshgrid size.
        #
        # This is a temporary solution for plotting mesh intersections.
        # This check should eventually be removed. See todo file.
        """
        if self.config['use_meshgrid'] is True:
            mesh_loc = self.point_to_local(self.config['mesh_points'])
            
            # If any mesh points fall outside of the optic width, test fails.
            test = True
            test &= np.all(abs(mesh_loc[:,0]) <= (self.config['width']  / 2))
            test &= np.all(abs(mesh_loc[:,1]) <= (self.config['height'] / 2))
            if not test:
                raise Exception('Optic dimentions too small to contain meshgrid.')
        """
        # autofill pixel grid sizes
        if self.param['pixel_size'] is None:
            self.param['pixel_size'] = self.param['width']/100
        if self.param['pixel_width'] is None:
            self.param['pixel_width'] = int(np.ceil(self.param['width']  / self.param['pixel_size']))
        if self.param['pixel_height'] is None:
            self.param['pixel_height'] = int(np.ceil(self.param['height']  / self.param['pixel_size']))

        self.image        = np.zeros((self.param['pixel_width'], self.param['pixel_height']))
        self.photon_count = 0
        
    def normalize(self, vector):
        magnitude = self.norm(vector)
        vector_norm = vector / magnitude[:, np.newaxis]
        return vector_norm
        
    def norm(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        return magnitude
            
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
    
    def generate_optic_normals(self, X, rays):
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
    
    def mesh_intersect_check(self, rays):
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

        return X, rays, hits
    
    def light(self, rays):
        m = rays['mask']
        if self.param['use_meshgrid'] is False:
            distance = self.intersect(rays)
            X, rays  = self.intersect_check(rays, distance)
            self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0])) 
            normals  = self.generate_optic_normals(X, rays)
            rays     = self.reflect_vectors(X, rays, normals)
            self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
        else:
            X, rays, hits = self.mesh_intersect_check(rays)
            self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))  
            normals  = self.mesh_generate_optic_normals(X, rays, hits)
            rays     = self.reflect_vectors(X, rays, normals)
            self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
        return rays

    def mesh_generate_optic_normals(self, X, rays, hits):
            m = rays['mask']
            normals = np.zeros(X.shape, dtype=np.float64)
            for ii in range(len(self.param['mesh_faces'])):
                tri = self.mesh_triangulate(ii)
                test = np.equal(ii, (hits - 1))
                test &= m
                normals[test] = tri['normal']
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
