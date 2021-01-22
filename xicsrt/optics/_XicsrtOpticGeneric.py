# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>
    James Kring <jdk0026@tigermail.auburn.edu>
    Yevgeniy Yakusevich <eugenethree@gmail.com>

Contains the XicsrtOpticGeneric class.
"""
from PIL import Image
import numpy as np

from xicsrt.util import profiler
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.objects._GeometryObject import GeometryObject

@dochelper
class XicsrtOpticGeneric(GeometryObject):
    """
    A generic optical element. 
    Optical elements have a position and rotation in 3D space and a finite 
    extent. Additional properties, such as as crystal spacing, rocking curve, 
    and reflectivity, should be defined in derived classes.
    """

    def default_config(self):
        """
        xsize
          The size of this element along the xaxis direction.
          Typically corresponds to the 'width' of the optic.

        ysize
          The size of this element along the yaxis direction.
          Typically corresponds to the 'height' of the optic.

        zsize
          The size of this element along the zaxis direction.
          Typically not required, but if needed will correspond to the 'depth'
          of the optic.

        width
          The width of this element. Aligned with the x-axis.

        height
          The height of this element. Aligned with the y-axis.

        depth:
          The depth of this element. Aligned with the z-axis.

        pixel_size: float (None)
          The pixel size, used for binning rays into images.
          This is currently a single number signifying square pixels.

        do_trace_local: bool (False)
          If true: transform rays to optic local coordinates before raytracing,
          do raytracing in local coordinates, then transform back to global
          coordinates.

          The default is 'false' as most built-in optics can perform raytracing
          in global coordinates. This option is convenient for optics with
          complex geometry for which intersection and reflection equations
          are easier or more clear to program in fixed local coordinates.

        do_miss_check: bool (true)
          Perform a check for whether the rays intersect the optic within the
          defined bounds (usually defined by 'width' and 'height'). If set to
          `False` all rays with a defined reflection/transmission condition
          will be traced.
        """
        config = super().default_config()
        
        # spatial information
        config['xsize']          = 0.0
        config['ysize']          = 0.0
        config['zsize']          = 0.0
        config['pixel_size']     = None

        # boolean settings
        config['do_trace_local'] = False
        config['do_miss_check'] = True

        return config

    def initialize(self):
        super().initialize()

        # autofill pixel grid sizes
        if self.param['pixel_size'] is None:
            self.param['pixel_size'] = self.param['xsize']/100

        # Determine the number of pixels on the detector.
        # For now assume that the user set the width of the detector to be
        # a multiple of the pixel size.
        #
        # Except for the detector there is really no reason that this would
        # always be true, so for now only make this a warning. I need to think
        # about how to handle this better.
        pixel_width = self.param['xsize'] / self.param['pixel_size']
        pixel_height = self.param['ysize'] / self.param['pixel_size']
        try:
            np.testing.assert_almost_equal(pixel_width, np.round(pixel_width))
            np.testing.assert_almost_equal(pixel_height, np.round(pixel_height))
        except AssertionError:
            self.log.warning(f"Optic width ({self.param['xsize']:0.4f}x{self.param['ysize']:0.4f})"
                             f"is not a multiple of the pixel_size ({self.param['pixel_size']:0.4f})."
                             f"May lead to truncation of output image."
                             )

        self.param['pixel_width'] = int(np.round(pixel_width))
        self.param['pixel_height'] = int(np.round(pixel_height))
        self.log.debug(f"Pixel grid size: {self.param['pixel_width']} x {self.param['pixel_height']}")

        self.image = np.zeros((self.param['pixel_width'], self.param['pixel_height']))
        self.photon_count = 0

    def trace_global(self, rays):
        """
        This is method that is called by the dispacher to perform
        ray-tracing for this optic. Rays into and out of this method
        are always in global coordinates.

        It may be convenient for some optics object to do raytracing
        in local coordinates rather than in global coordinates. This
        method facilitates this by implementing the 'trace_local'
        configuration option.
        """

        if self.param['do_trace_local']:
            self.log.debug('Converting to local coordinates.')
            rays = self.ray_to_local(rays, copy=False)

        rays = self.trace(rays)

        if self.param['do_trace_local']:
            self.log.debug('Converting to external coordinates.')
            rays = self.ray_to_external(rays, copy=False)
        return rays

    def trace(self, rays):
        """
        The main method that performs raytracing for this optic.

        Raytracing here may be done in global or local coordinates
        depending on the how the optic is designed and the value
        of the configuration option: 'do_trace_local'.

        This method can be re-implemented by indiviual optics.
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
            m[m] &= (np.abs(X_local[m,0]) < self.param['xsize'] / 2)
            m[m] &= (np.abs(X_local[m,1]) < self.param['ysize'] / 2)

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
