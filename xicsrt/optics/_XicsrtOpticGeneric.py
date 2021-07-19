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

from xicsrt.tools import xicsrt_aperture

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

        pixel_size: float (None)
          The pixel size, used for binning rays into images.
          This is currently a single number signifying square pixels.

        aperture: dict or array (None)
          Define one or more apertures to to apply to this optic.
          Each aperture is defined as a dictionary with the following keys:
          shape, size, origin, logic. The origin and logic field keys are
          optional. The interpretation of size will depend on the provided
          shape.

        trace_local: bool (False)
          If true: transform rays to optic local coordinates before raytracing,
          do raytracing in local coordinates, then transform back to global
          coordinates.

          The default is 'false' as most built-in optics can perform raytracing
          in global coordinates. This option is convenient for optics with
          complex geometry for which intersection and reflection equations
          are easier or more clear to program in fixed local coordinates.

        check_size: bool (true)
          Perform a check for whether the rays intersect the optic within the
          defined bounds (usually defined by 'xsize' and 'ysize'). If set to
          `False` all rays with a defined reflection/transmission condition
          will be traced if an intersection can be determined.

        check_aperture: bool (true)
          Perform a check for whether the rays intersect the optic within the
          defined bounds (usually defined by 'xsize' an 'ysize'). If set to
          `False` all rays with a defined reflection/transmission condition
          will be traced if an intersection can be determined.
        """
        config = super().default_config()
        
        # spatial information
        config['xsize']          = None
        config['ysize']          = None
        config['zsize']          = None
        config['pixel_size']     = None

        # boolean settings
        config['trace_local'] = False
        config['check_size'] = True
        config['check_aperture'] = True
        
        #aperture info
        config['aperture'] = None

        return config

    def initialize(self):
        super().initialize()

        if self.param['xsize'] and self.param['ysize']:
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
            pixel_xsize = self.param['xsize'] / self.param['pixel_size']
            pixel_ysize = self.param['ysize'] / self.param['pixel_size']
            try:
                np.testing.assert_almost_equal(pixel_xsize, np.round(pixel_xsize))
                np.testing.assert_almost_equal(pixel_ysize, np.round(pixel_ysize))
            except AssertionError:
                self.log.warning(f"Optic width ({self.param['xsize']:0.4f}x{self.param['ysize']:0.4f})"
                                 f"is not a multiple of the pixel_size ({self.param['pixel_size']:0.4f})."
                                 f"May lead to truncation of output image."
                                 )

            self.param['pixel_xsize'] = int(np.round(pixel_xsize))
            self.param['pixel_ysize'] = int(np.round(pixel_ysize))
            self.log.debug(f"Pixel grid size: {self.param['pixel_xsize']} x {self.param['pixel_ysize']}")

            self.image = np.zeros((self.param['pixel_xsize'], self.param['pixel_ysize']))
            self.param['enable_image'] = True
        else:
            self.param['enable_image'] = False

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

        if self.param['trace_local']:
            self.log.debug('Converting to local coordinates.')
            rays = self.ray_to_local(rays, copy=False)

        rays = self.trace(rays)

        if self.param['trace_local']:
            self.log.debug('Converting to external coordinates.')
            rays = self.ray_to_external(rays, copy=False)
        return rays

    def trace(self, rays):
        """
        The main method that performs raytracing for this optic.

        Raytracing here may be done in global or local coordinates
        depending on the how the optic is designed and the value
        of the configuration option: 'trace_local'.

        This method can be re-implemented by indiviual optics.
        """
        m = rays['mask']

        rays, X = self.intersect(rays)
        rays = self.check_bounds(rays, X)

        self.log.debug(' Rays on   {}:   {:6.4e}'.format(self.name, m[m].shape[0]))
        normals  = self.generate_normals(rays, X)
        rays     = self.reflect_vectors(rays, X, normals)
        self.log.debug(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))

        return rays

    def intersect(self, rays):
        """
        Calculate the intersection between the incoming rays and this optic.

        Programming Notes
        -----------------

        When creating an new optic one can choose to reimplement either
        this method or `intersect_distance`, whichever is easier.
        """

        rays, distance = self.intersect_distance(rays)

        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        # X is the 3D point where the ray intersects the optic.
        # It may be better, from a memory perspective, to modify O in place
        # instead of creating a new array. If this is done we need to be sure to
        # set O to np.nan where no intersection exists.
        X = np.full(O.shape, np.nan, dtype=np.float64)
        X[m] = O[m] + D[m] * distance[m, np.newaxis]

        return rays, X

    def intersect_distance(self, rays):
        """
        Calculate the distance to an intersection with a plane.
        """

        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        if self.param['trace_local']:
            distance = np.zeros(m.shape, dtype=np.float64)
            distance[m] = (np.dot((0 - O[m]), np.array([ 0., 0., 1.]))
                           / np.dot(D[m], np.array([ 0., 0., 1.])))
        else:
            distance = np.zeros(m.shape, dtype=np.float64)
            distance[m] = (np.dot((self.param['origin'] - O[m]), self.param['zaxis'])
                           / np.dot(D[m], self.param['zaxis']))

        # Update the mask to only include positive distances.
        m &= (distance >= 0)

        return rays, distance

    def check_bounds(self, rays, X):
        m = rays['mask']

        if self.param['trace_local']:
            X_local = X
        else:
            X_local = np.zeros(X.shape, dtype=np.float64)
            X_local[m] = self.point_to_local(X[m])

        rays = self.check_size(rays, X_local)
        rays = self.check_aperture(rays, X_local)

        return rays

    def check_size(self, rays, X_local):
        """
        Check if the ray intersection is within the optic bounds as set
        by the xsize, ysize and zsize config options.

        Note:
            This method expects to be given the ray intersections in local
            coordinates. Generally this method should not be called directly
            instead use `check_bounds`.
        """
        m = rays['mask']

        if self.param['check_size']:
            if self.param['xsize'] is not None:
                m[m] &= (np.abs(X_local[m,0]) < self.param['xsize'] / 2)
            if self.param['ysize'] is not None:
                m[m] &= (np.abs(X_local[m,1]) < self.param['ysize'] / 2)
            if self.param['zsize'] is not None:
                m[m] &= (np.abs(X_local[m,2]) < self.param['zsize'] / 2)
        
        return rays
    
    def check_aperture(self, rays, X_local):
        """
        Check if the ray intersection is within the aperture as set
        by the 'aperture' config option.

        Note:
            This method expects to be given the ray intersections in local
            coordinates. Generally this method should not be called directly
            instead use `check_bounds`.
        """
        m = rays['mask']

        if self.param['check_aperture']:
            m_aperture = xicsrt_aperture.aperture_mask(X_local, m, self.param['aperture'])
            m[m] = m_aperture[m]
        
        return rays

    def generate_normals(self, rays, X):
        """
        The Generic optic is flat, so the normal direction is always the zaxis.
        """
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        normals[m] = self.param['zaxis']
        return normals

    def reflect_vectors(self, rays, X, normals=None, mask=None):
        """
        Generic optic has no reflection, rays just pass through.
        """
        O = rays['origin']

        # The following line does two things:
        # 1. Set the origin for rays (lost or found) with an intersection.
        # 2. Set the rays without an intersection to nan.
        O[:] = X[:]
        
        return rays

    def make_image(self, rays):
        """
        Collect the rays that intersect with this optic into a pixel array that
        can be used to generate an intersection image.

        Programming Notes
        -----------------

        It is important that this calculation is compatible with intersect_check
        in terms of floating point errors.  The simple way to achieve this is
        to ensure that both use the same calculation method.
        """

        if not self.param['enable_image']:
            return None

        image = np.zeros((self.param['pixel_xsize'], self.param['pixel_ysize']))
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
            channel[:,0] = pix[:,0] + (self.param['pixel_xsize'] - 1)/2
            channel[:,1] = pix[:,1] + (self.param['pixel_ysize'] - 1)/2
            
            # Bin the channels into integer values so that we can use them as
            # indexes into the image. Keep in mind that channel coordinate
            # system is defined from the center of the pixel.
            channel = np.round(channel).astype(int)
            
            # Check for any hits that are outside of the image.
            # These are possible due to floating point calculations.
            m = np.ones(num_lines, dtype=bool)
            m &= channel[:,0] >= 0
            m &= channel[:,0] < self.param['pixel_xsize']
            m &= channel[:,1] >= 0
            m &= channel[:,1] < self.param['pixel_ysize']
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
        if not self.param['enable_image']:
            raise Exception('Images are not enabled. Check config parameters.')

        image = self.make_image(rays)
        self.image[:,:] += image
        
        return self.image[:,:] 
        
    def save_image(self, image_name, rotate=None):

        if not self.param['enable_image']:
            raise Exception('Images are not enabled. Check config parameters.')

        if rotate:
            out_array = np.rot90(self.image)
        else:
            out_array = self.image
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)
