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

from xicsrt.util import profiler
from xicsrt.objects._GeometryObject import GeometryObject

class XicsrtOpticGeneric(GeometryObject):
    """
    A generic optical element. 
    Optical elements have a position and rotation in 3D space and a finite 
    extent. Additional properties, such as as crystal spacing, rocking curve, 
    and reflectivity, should be defined in derived classes.
    """
        
    def get_default_config(self):
        config = super().get_default_config()
        
        # boolean settings
        config['do_miss_check'] = True
        
        # spatial information
        config['width']          = 0.0
        config['height']         = 0.0
        config['depth']          = 0.0
        config['pixel_size']     = None
        config['pixel_width']    = None
        config['pixel_height']   = None

        return config

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

    def light(self, rays):
        """
        This is the main method that is called to perform ray-tracing
        for this optic.

        It may be convenient for some optics object to do raytracing
        in local coordinates rather than in global coordinates.
        that can achived by reimplementing this method as follows:

          self.ray_to_local(rays)
          super().light(rays)
          self.ray_to_external(rays)
          return rays
        """
        m = rays['mask']

        distance = self.intersect(rays)
        X, rays  = self.intersect_check(rays, distance)
        self.log.debug(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
        normals  = self.generate_normals(X, rays)
        rays     = self.reflect_vectors(X, rays, normals)
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
        distance[m] = (np.dot((self.param['origin'] - O[m]), self.param['zaxis'])
                       / np.dot(D[m], self.param['zaxis']))

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

    def reflect_vectors(self, X, rays, normals=None, mask=None):
        """
        Generic optic has no reflection, rays just pass through.
        """
        O = rays['origin']
        O[:]  = X[:]
        
        return rays

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
                self.log.warning('Rays found outside of pixel grid ({}).'.format(num_out))
            
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
