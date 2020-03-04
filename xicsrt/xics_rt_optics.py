# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:05:57 2017
Edited on Fri Sep 06 11:06:00 2019

@author: James
@editor: Eugene

Description
-----------
The spherical quartz crystal and Highly Oriented Pyrolytic Graphite film 
that reflect X-rays that satisfy the Bragg condition. Optical elements have a
position and rotation in 3D space, optical properties such as crystal spacing,
rocking curve, and reflectivity, as well as a height and width.
"""
from PIL import Image
import numpy as np

from xicsrt.xics_rt_objects import TraceObject

class GenericOptic(TraceObject):
    def __init__(self, optic_input):
        super().__init__(
            optic_input['position']
            ,optic_input['normal']
            ,optic_input['orientation'])
        
        #spatial information
        self.position       = optic_input['position']
        self.normal         = optic_input['normal']
        self.xorientation   = optic_input['orientation']
        self.yorientation   = np.cross(self.normal, self.xorientation)
        self.yorientation  /= np.linalg.norm(self.yorientation)
        self.width          = optic_input['width']
        self.height         = optic_input['height']
        self.depth          = 0.0
        #mesh information
        self.use_meshgrid   = optic_input['use_meshgrid']
        self.mesh_points    = optic_input['mesh_points']
        self.mesh_faces     = optic_input['mesh_faces']
        #xray optical information and polarization information
        self.crystal_spacing= optic_input['spacing']
        self.rocking_curve  = optic_input['rocking_curve']
        self.reflectivity   = optic_input['reflectivity']
        self.sigma_data     = optic_input['sigma_data']
        self.pi_data        = optic_input['pi_data']
        self.mix_factor     = optic_input['mix_factor']
        #image information
        self.pixel_size     = self.width / optic_input['pixel_scaling']
        self.pixel_width    = int(round(self.width  / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))
        self.pixel_array    = np.zeros((self.pixel_width, self.pixel_height))
        self.photon_count   = 0
        #boolean settings
        self.bragg_checks   = optic_input['do_bragg_checks']
        self.miss_checks    = optic_input['do_miss_checks']
        self.rocking_type   = str.lower(optic_input['rocking_curve_type'])

        self.check_inputs()
            
    def check_inputs(self):

        # Check the optic size compare to the meshgrid size.
        #
        # This is a temporary solution for plotting mesh intersections.
        # This check should eventually be removed. See todo file.
        if self.use_meshgrid is True:
            mesh_loc = self.point_to_local(self.mesh_points)

            # If any mesh points fall outside of the optic width, test fails.
            test = True
            test |= np.all(abs(mesh_loc[:,0]) > self.width / 2)
            test |= np.all(abs(mesh_loc[:,1]) < self.height / 2)
            
            if not test:
                raise Exception('Optic dimentions too small to contain meshgrid.')

        ## Check if the pixel array and optic size match.
        test  = True
        test &= (self.pixel_width  == int(round(self.width  / self.pixel_size)))
        test &= (self.pixel_height == int(round(self.height / self.pixel_size)))
        if not test:
            raise Exception('pixel_width/pixel_height and width/height inputs do not match.')

    def normalize(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        vector_norm = vector / magnitude[:, np.newaxis]
        return vector_norm
        
    def norm(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        return magnitude
    
    def mesh_triangulate(self, ii):
        points = self.mesh_points
        faces  = self.mesh_faces
        #find which points belong to the triangle face
        p1 = points[faces[ii,0],:]
        p2 = points[faces[ii,1],:]
        p3 = points[faces[ii,2],:]

        #calculate the centerpoint and normal of the triangle face
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

    def rocking_curve_filter(self, incident_angle, bragg_angle):
        if "step" in self.rocking_type:
            # Step Function
            mask = (abs(incident_angle - bragg_angle) <= self.rocking_curve)
            
        elif "gauss" in self.rocking_type:
            # Convert from FWHM to sigma.
            sigma = self.rocking_curve / np.sqrt(2 * np.log(2)) / 2
            
            # evaluate rocking curve reflectivity value for each ray
            p = np.exp(-np.power(incident_angle - bragg_angle, 2.) / (2 * sigma**2))
            
            # give each ray a random number and compare that number to the reflectivity 
            # curves to decide whether the ray reflects or not.         
            test = np.random.uniform(0.0, 1.0, len(incident_angle))
            mask = p.flatten() > test
            
        elif "file" in self.rocking_type:
            # read datafiles and extract points
            sigma_data  = np.loadtxt(self.sigma_data, dtype = np.float64)
            pi_data     = np.loadtxt(self.pi_data, dtype = np.float64)
            
            # convert data from arcsec to rad
            sigma_data[:,0] *= np.pi / (180 * 3600)
            pi_data[:,0]    *= np.pi / (180 * 3600)
            
            # evaluate rocking curve reflectivity value for each incident ray
            sigma_curve = np.zeros(len(incident_angle), dtype = np.float64)
            pi_curve    = np.zeros(len(incident_angle), dtype = np.float64)
            
            sigma_curve = np.interp((incident_angle - bragg_angle), 
                                    sigma_data[:,0], sigma_data[:,1], 
                                    left = 0.0, right = 0.0)
            
            pi_curve    = np.interp((incident_angle - bragg_angle), 
                                    pi_data[:,0], pi_data[:,1], 
                                    left = 0.0, right = 0.0)
            
            # give each ray a random number and compare that number to the reflectivity 
            # curves to decide whether the ray reflects or not. Use mix factor.
            test = np.random.uniform(0.0, 1.0, len(incident_angle))
            mask = self.mix_factor * sigma_curve + (1 - self.mix_factor) * pi_curve >= test
            
        else:
            raise Exception('Rocking curve type not understood: {}'.format(self.rocking_curve))
        return mask  
    
    def intersect_check(self, rays, distance):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X = np.zeros(O.shape, dtype=np.float64)
        X_local = np.zeros(O.shape, dtype=np.float64)
        
        # X is the 3D point where the ray intersects the optic
        X[m] = O[m] + D[m] * distance[m,np.newaxis]

        X_local[m] = self.point_to_local(X[m])
        
        #find which rays hit the optic, update mask to remove misses
        if self.miss_checks is True:
            m[m] &= (np.abs(X_local[m,0]) < self.width / 2)
            m[m] &= (np.abs(X_local[m,1]) < self.height / 2)

        return X, rays
    
    def mesh_intersect_check(self, rays):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X     = np.zeros(D.shape, dtype=np.float64)
        hits  = np.zeros(m.shape, dtype=np.int)

        # Loop over each triangular face to find which rays hit
        for ii in range(len(self.mesh_faces)):
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
            
            #append the results to the global impacts arrays
            X[test]    = intersect[test]
            hits[test] = ii + 1
        
        #mask all the rays that missed all faces
        if self.miss_checks is True:
            m[m] &= (hits[m] != 0)
        return X, rays, hits
    
    def angle_check(self, rays, normals):
        D = rays['direction']
        W = rays['wavelength']
        m = rays['mask']
        
        bragg_angle = np.zeros(m.shape, dtype=np.float64)
        dot = np.zeros(m.shape, dtype=np.float64)
        incident_angle = np.zeros(m.shape, dtype=np.float64)
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the optic
        bragg_angle[m] = np.arcsin( W[m] / (2 * self.crystal_spacing))
        dot[m] = np.abs(np.einsum('ij,ij->i',D[m], -1 * normals[m]))
        incident_angle[m] = (np.pi / 2) - np.arccos(dot[m] / self.norm(D[m]))
        
        #check which rays satisfy bragg, update mask to remove those that don't
        if self.bragg_checks is True:
            m[m] &= self.rocking_curve_filter(bragg_angle[m], incident_angle[m])
        return rays, normals
    
    def reflect_vectors(self, X, rays, normals):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        # Check which vectors meet the Bragg condition (with rocking curve)
        rays, normals = self.angle_check(rays, normals)
        
        # Perform reflection around normal vector, creating new rays with new
        # origin O = X and new direction D
        O[:]  = X[:]
        D[m] -= 2 * (np.einsum('ij,ij->i', D[m], normals[m])[:, np.newaxis]
                     * normals[m])
        
        return rays
    
    def light(self, rays):
        m = rays['mask']
        if self.use_meshgrid is False:
            distance = self.optic_intersect(rays)
            X, rays  = self.intersect_check(rays, distance)
            # print(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0])) 
            normals  = self.generate_optic_normals(X, rays)
            rays     = self.reflect_vectors(X, rays, normals)
            # print(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
        else:
            X, rays, hits = self.mesh_intersect_check(rays)
            # print(' Rays on {}:   {:6.4e}'.format(self.name, m[m].shape[0]))  
            normals  = self.mesh_generate_optic_normals(X, rays, hits)
            rays     = self.reflect_vectors(X, rays, normals)
            # print(' Rays from {}: {:6.4e}'.format(self.name, m[m].shape[0]))
        return rays

    def collect_rays(self, rays):
        """
        Collect the rays that his this optic into a pixel array that can be used
        for further analysis or visualization.

        Programming Notes
        -----------------

        It is important thas this calculation is compatible with intersect_check
        in terms of floating point errors.  The simple way to achive this is
        to ensure that both use the same calculation method.
        """
        X = rays['origin']
        m = rays['mask'].copy()
        
        num_lines = np.sum(m)
        self.photon_count = num_lines
        
        # Add the ray hits to the pixel array
        if num_lines > 0:
            # Transform the intersection coordinates from external coordinates
            # to local optical coordinates.
            point_loc = self.point_to_local(X[m])
            
            # Bin the intersections into pixels using integer math.
            pix = np.zeros([num_lines, 3], dtype = int)
            pix = np.floor(point_loc / self.pixel_size).astype(int)
            
            # Check to ascertain if origin pixel is even or odd
            if (self.pixel_width % 2) == 0:
                pix_min_x = self.pixel_width//2
            else:
                pix_min_x = (self.pixel_width + 1)//2
                
            if (self.pixel_height % 2) == 0:
                pix_min_y = self.pixel_height//2
            else:
                pix_min_y = (self.pixel_height + 1)//2
            
            pix_min = np.array([pix_min_x, pix_min_y, 0], dtype = int)
            
            # Convert from pixels, which are centered around the origin, to
            # channels, which start from the corner of the optic.
            channel    = np.zeros(pix.shape, dtype = int)
            channel[:] = pix[:] + pix_min
            
            # I feel like there must be a faster way to do this than to loop over
            # every intersection.  This could be slow for large arrays.
            for ii in range(len(channel)):
                self.pixel_array[channel[ii,0], channel[ii,1]] += 1
        
        return self.pixel_array
        
    def output_image(self, image_name, rotate=None):
        if rotate:
            out_array = np.rot90(self.pixel_array)
        else:
            out_array = self.pixel_array
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)

      
class SphericalCrystal(GenericOptic):
    def __init__(self, crystal_input):
        super().__init__(crystal_input)
        
        self.radius         = crystal_input['curvature']
        self.center         = self.radius * self.normal + self.position
        
    def optic_intersect(self, rays):
        """
        Calulate the distance to the intersection of the rays with the
        spherical optic.

        This calculation is copied from:
        https://www.scratchapixel.com/lessons/3d-basic-rendering/
                minimal-ray-tracer-rendering-simple-shapes/
                ray-sphere-intersection
        """
        #setup variables
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        d        = np.zeros(m.shape, dtype=np.float64)
        t_hc     = np.zeros(m.shape, dtype=np.float64)
        t_0      = np.zeros(m.shape, dtype=np.float64)
        t_1      = np.zeros(m.shape, dtype=np.float64)

        # L is the destance from the ray origin to the center of the sphere.
        # t_ca is the projection of this distance along the ray direction.
        L     = self.center - O
        t_ca  = np.einsum('ij,ij->i', L, D)
        
        # If t_ca is less than zero, then there is no intersection in the
        # the direction of the ray (there might be an intersection behind.)
        # Use mask to only perform calculations on rays that hit the crystal
        # m[m] &= (t_ca[m] >= 0)
        
        # d is the minimum distance between a ray and center of curvature.
        d[m] = np.sqrt(np.einsum('ij,ij->i',L[m] ,L[m]) - t_ca[m]**2)

        # If d is larger than the radius, the ray misses the sphere.
        m[m] &= (d[m] <= self.radius)
        
        # t_hc is the distance from d to the intersection points.
        t_hc[m] = np.sqrt(self.radius**2 - d[m]**2)
        
        t_0[m] = t_ca[m] - t_hc[m]
        t_1[m] = t_ca[m] + t_hc[m]

        # Distance traveled by the ray before hitting the optic
        distance[m] = np.where(t_0[m] > t_1[m], t_0[m], t_1[m])
        return distance
    
    def generate_optic_normals(self, X, rays):
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        normals[m] = self.normalize(self.center - X[m])
        return normals
    
    def mesh_generate_optic_normals(self, X, rays, hits):
        m = rays['mask']
        normals = np.zeros(X.shape, dtype=np.float64)
        for ii in range(len(self.mesh_faces)):
            tri   = self.mesh_triangulate(ii)
            test  = np.equal(ii, (hits - 1))
            test &= m
            normals[test] = tri['normal']
            
        return normals
    

class MosaicGraphite(GenericOptic):
    def __init__(self, graphite_input):
        super().__init__(graphite_input)
        
        self.__name__       = 'MosaicGraphite'
        self.mosaic_spread  = graphite_input['mosaic_spread']

    def optic_intersect(self, rays):
        """
        Calulate the distance to the intersection of the rays with an 
        infinite plane.
        """
        
        #test to see if a ray intersects the mirror plane
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m]  = np.dot((self.position - O[m]), self.normal)
        distance[m] /= np.dot(D[m], self.normal)

        # Update the mask to count only intersections with the plain.
        m[m] &= (distance[m] > 0)
        
        return distance
    
    def generate_optic_normals(self, X, rays):
        # Pulled from Novi's FocusedExtendedSource
        # Generates a list of crystallite norms normally distributed around the
        # average graphite mirror norm
        def f(theta, number):
            output = np.empty((number, 3))
            
            z   = np.random.uniform(np.cos(theta),1, number)
            phi = np.random.uniform(0, np.pi * 2, number)
            
            output[:,0]   = np.sqrt(1-z**2) * np.cos(phi)
            output[:,1]   = np.sqrt(1-z**2) * np.sin(phi)
            output[:,2]   = z
            return output
        
        O = rays['origin']
        m = rays['mask']
        
        normals = np.zeros(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, len(m))

        # Create two vectors perpendicular to the surface normal,
        # it doesn't matter how they are oriented otherwise.
        norm_surf = np.ones(O.shape) * self.normal
        o_1     = np.zeros(O.shape)
        o_1[m]  = np.cross(norm_surf[m], [0,0,1])
        o_1[m] /= np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
        o_2     = np.zeros(O.shape)
        o_2[m]  = np.cross(norm_surf[m], o_1[m])
        o_2[m] /= np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
        
        R = np.empty((len(m), 3, 3))
        # We could mask this with m, but I don't know if that will
        # improve speed or actually make it worse.
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = norm_orig

        normals[m] = np.einsum('ij,ijk->ik', dir_local[m], R[m])
        return normals
    
    def mesh_generate_optic_normals(self, X, rays, hits):
        # Pulled from Novi's FocusedExtendedSource
        # Generates a list of crystallite norms normally distributed around the
        # average graphite mirror norm
        def f(theta, number):
            output = np.empty((number, 3))
            
            z   = np.random.uniform(np.cos(theta),1, number)
            phi = np.random.uniform(0, np.pi * 2, number)
            
            output[:,0]   = np.sqrt(1-z**2) * np.cos(phi)
            output[:,1]   = np.sqrt(1-z**2) * np.sin(phi)
            output[:,2]   = z
            return output
        
        O = rays['origin']
        m = rays['mask']
        
        normals = np.zeros(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, len(m))

        for ii in range(len(self.mesh_faces)):
            tri   = self.mesh_triangulate(ii)
            test  = np.equal(ii, (hits - 1))
            test &= m
            
            norm_surf  = np.ones(O.shape) * tri['normal']
            o_1     = np.zeros(O.shape)
            o_1[test]  = np.cross(norm_surf[test], [0,0,1])
            o_1[test] /= np.linalg.norm(o_1[test], axis=1)[:, np.newaxis]
            o_2     = np.zeros(O.shape)
            o_2[test]  = np.cross(norm_surf[test], o_1[test])
            o_2[test] /= np.linalg.norm(o_2[test], axis=1)[:, np.newaxis]
            
            R = np.empty((len(m), 3, 3))
            # We could mask this with test, but I don't know if that will
            # improve speed or actually make it worse.
            R[:,0,:] = o_1
            R[:,1,:] = o_2
            R[:,2,:] = normal
            
            normals[test] = np.einsum('ij,ijk->ik', dir_local[test], R[test])
            
        return normals
