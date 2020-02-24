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
            
    def pixel_array_size_check(self):
        ## Before loading anything up, check if the pixel array is large enough
        failure  = False
        
        if self.use_meshgrid is True:
            mesh_loc = self.point_to_local(self.mesh_points)
            mesh_loc = np.round(mesh_loc, decimals = 6)

            #if any mesh points fall outside of the pixel array, it fails
            failure |= np.any(mesh_loc[:,0] < -self.width / 2)
            failure |= np.any(mesh_loc[:,1] < -self.height/ 2)
            failure |= np.any(mesh_loc[:,0] > self.width / 2)
            failure |= np.any(mesh_loc[:,1] > self.height/ 2)
            
            if failure:
                print('{} pixel array is too small'.format(self.__name__))
                print('Meshgrid will not fit within its extent')
                print('Please increase {} width/height'.format(self.__name__))
                raise Exception
        
        ## Before loading anything up, check if the pixel array is mishapen
        failure |= (self.pixel_width  != int(round(self.width  / self.pixel_size)))
        failure |= (self.pixel_height != int(round(self.height / self.pixel_size)))
        
        if failure:
            print('{} pixel array is mishapen'.format(self.__name__))
            print('Pixel array width/height and optic width/height are disproportionate')
            print('Please check {} code'.format(self.__name__))
            
            print('{} width  = {}'.format(self.__name__, self.width))
            print('{} height = {}'.format(self.__name__, self.height))
            print('{} pixel width  = {}'.format(self.__name__, self.pixel_width))
            print('{} pixel height = {}'.format(self.__name__, self.pixel_height))
            raise Exception

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
        xproj = np.zeros(m.shape, dtype=np.float64)
        yproj = np.zeros(m.shape, dtype=np.float64)
        
        #X is the 3D point where the ray intersects the optic
        X[m] = O[m] + D[m] * distance[m,np.newaxis]
        
        #find which rays hit the optic, update mask to remove misses
        xproj[m] = abs(np.dot(X[m] - self.position, self.xorientation))
        yproj[m] = abs(np.dot(X[m] - self.position, self.yorientation))
        if self.miss_checks is True:
            m[m] &= ((xproj[m] <= self.width / 2) & (yproj[m] <= self.height / 2))
            
        return X, rays
    
    def mesh_intersect_check(self, rays):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        X     = np.zeros(D.shape, dtype=np.float64)
        hits  = np.zeros(m.shape, dtype=np.int)

        #loop over each triangular face to find which rays hit
        for ii in range(len(self.mesh_faces)):
            intersect= np.zeros(O.shape, dtype=np.float64)
            distance = np.zeros(m.shape, dtype=np.float64)
            test     = np.zeros(m.shape, dtype=np.bool)
            
            #query the triangle mesh grid
            tri = self.mesh_triangulate(ii)
            p0= tri['center']
            p1= tri['point1']
            p2= tri['point2']
            p3= tri['point3']
            n = tri['normal']

            #find the intersection point between the rays and triangle plane
            distance     = np.dot((p0 - O), n) / np.dot(D, n)
            intersect[m] = O[m] + D[m] * distance[m,np.newaxis]
            
            #test to see if the intersection is inside the triangle
            #'test' starts as 0 and flips to 1 for each successful hit
            #uses barycentric coordinate math (compare parallelpiped areas)
            tri_area = np.linalg.norm(np.cross((p1 - p2),(p1 - p3)))
            alpha    = self.norm(np.cross((intersect - p2),(intersect - p3)))
            beta     = self.norm(np.cross((intersect - p3),(intersect - p1)))
            gamma    = self.norm(np.cross((intersect - p1),(intersect - p2)))
            
            alpha /= tri_area
            beta  /= tri_area
            gamma /= tri_area
            
            test |= np.isclose((alpha + beta + gamma), 1.0)
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
            m &= self.rocking_curve_filter(bragg_angle,incident_angle)
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
            print(' Rays on {}:   {:6.4e}'.format(self.__name__, m[m].shape[0])) 
            normals  = self.generate_optic_normals(X, rays)
            rays     = self.reflect_vectors(X, rays, normals)
            print(' Rays from {}: {:6.4e}'.format(self.__name__, m[m].shape[0]))
        else:
            X, rays, hits = self.mesh_intersect_check(rays)
            print(' Rays on {}:   {:6.4e}'.format(self.__name__, m[m].shape[0]))  
            normals  = self.mesh_generate_optic_normals(X, rays, hits)
            rays     = self.reflect_vectors(X, rays, normals)
            print(' Rays from {}: {:6.4e}'.format(self.__name__, m[m].shape[0]))
        return rays

    def collect_rays(self, rays):
        X = rays['origin']
        m = rays['mask'].copy()
        
        num_lines = np.sum(m)
        self.photon_count = num_lines
        
        ##Add the ray hits to the pixel array
        if num_lines > 0:
            # Transform the intersection coordinates from external coordinates
            # to local optical coordinates.
            point_loc = self.point_to_local(X[m])
            
            # Bin the intersections into pixels using integer math.
            pix = np.zeros([num_lines, 3], dtype = int)
            pix = np.round(point_loc / self.pixel_size).astype(int)
            
            #check to ascertain if origin pixel is even or odd
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
                try:
                    self.pixel_array[channel[ii,0], channel[ii,1]] += 1
                except:
                    pass
                
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
        
        self.__name__       = 'SphericalCrystal'
        self.radius         = crystal_input['curvature']
        self.center         = self.radius * self.normal + self.position
        
        self.pixel_array_size_check()
        
    def optic_intersect(self, rays):
        """
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
        
        #distance traveled by the ray before hitting the optic
        #this calculation is performed for all rays, mask regardless
        L     = self.center - O
        t_ca  = np.einsum('ij,ij->i', L, D)
        
        #If t_ca is less than zero, then there is no intersection
        #Use mask to only perform calculations on rays that hit the crystal        
        #d is the impact parameter between a ray and center of curvature
        
        d[m]    = np.sqrt(np.einsum('ij,ij->i',L[m] ,L[m])
                                 - t_ca[m]**2)
        t_hc[m] = np.sqrt(self.radius**2 - d[m]**2)
        
        t_0[m] = t_ca[m] - t_hc[m]
        t_1[m] = t_ca[m] + t_hc[m]
        
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
            tri  = self.mesh_triangulate(ii)
            test = np.equal(ii, (hits - 1))
            test&= m
            normals[test] = tri['normal']
            
        return normals
    

class MosaicGraphite(GenericOptic):
    def __init__(self, graphite_input):
        super().__init__(graphite_input)
        
        self.__name__       = 'MosaicGraphite'
        self.mosaic_spread  = graphite_input['mosaic_spread']
        
        self.pixel_array_size_check()

    def optic_intersect(self, rays):
        #test to see if a ray intersects the mirror plane
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m]  = np.dot((self.position - O[m]), self.normal)
        distance[m] /= np.dot(D[m], self.normal)
        
        test = (distance > 0) & (distance < 10)
        distance = np.where(test, distance, 0)
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
        normals = np.empty(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, len(m))
        
        normal = np.ones(O.shape) * self.normal
        o_1     = np.cross(normal, [0,0,1])
        o_1[m] /= np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
        o_2     = np.cross(normal, o_1)
        o_2[m] /= np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
        
        R = np.empty((len(m), 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal
        
        normals = np.einsum('ij,ijk->ik', dir_local, R)
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
        normals = np.empty(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, len(m))
        
        for ii in range(len(self.mesh_faces)):
            tri  = self.mesh_triangulate(ii)
            test = np.equal(ii, (hits - 1))
            test&= m
            
            normal  = np.ones(O.shape) * tri['normal']
            o_1     = np.cross(normal, [0,0,1])
            o_1[m] /= np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
            o_2     = np.cross(normal, o_1)
            o_2[m] /= np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
            
            R = np.empty((len(m), 3, 3))
            R[:,0,:] = o_1
            R[:,1,:] = o_2
            R[:,2,:] = normal
            
            normals[test] = np.einsum('ij,ijk->ik', dir_local, R)[test]
        return normals
