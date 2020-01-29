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
        super(GenericOptic, self).__init__(
            optic_input['position']
            ,optic_input['normal']
            ,optic_input['orientation'])

        self.position       = optic_input['position']
        self.normal         = optic_input['normal']
        self.xorientation   = optic_input['orientation']
        self.yorientation   = np.cross(self.normal, self.xorientation)
        self.yorientation  /= np.linalg.norm(self.yorientation)
        self.spacing        = optic_input['spacing']
        self.rocking_curve  = optic_input['rocking_curve']
        self.reflectivity   = optic_input['reflectivity']
        self.width          = optic_input['width']
        self.height         = optic_input['height']
        self.depth          = 0.0
        self.pixel_size     = self.width / optic_input['pixel_scaling']
        self.pixel_width    = int(round(self.width  / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))
        self.pixel_array    = np.zeros((self.pixel_width, self.pixel_height))
        self.sigma_data     = optic_input['sigma_data']
        self.pi_data        = optic_input['pi_data']
        self.mix_factor     = optic_input['mix_factor']
        self.bragg_checks   = optic_input['do_bragg_checks']
        self.miss_checks    = optic_input['do_miss_checks']
        self.rocking_type   = str.lower(optic_input['rocking_curve_type'])

    def normalize(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        vector_norm = vector / magnitude[:, np.newaxis]
        return vector_norm
        
    def norm(self, vector):
        magnitude = np.einsum('ij,ij->i', vector, vector) ** .5
        return magnitude 

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
    
    def angle_check(self, X, rays, norm):
        D = rays['direction']
        W = rays['wavelength']
#       w = rays['weight']
        m = rays['mask']
        
        bragg_angle = np.zeros(m.shape, dtype=np.float64)
        dot = np.zeros(m.shape, dtype=np.float64)
        incident_angle = np.zeros(m.shape, dtype=np.float64)
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the optic
        bragg_angle[m] = np.arcsin( W[m] / (2 * self.crystal_spacing))
        dot[m] = np.abs(np.einsum('ij,ij->i',D[m], -1 * norm[m]))
        incident_angle[m] = (np.pi / 2) - np.arccos(dot[m] / self.norm(D[m]))
        #check which rays satisfy bragg, update mask to remove those that don't
        if self.bragg_checks is True:
            m[m] &= self.rocking_curve_filter(bragg_angle[m], incident_angle[m])
        return rays, norm
    
    def reflect_vectors(self, X, rays, norm):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        # Check which vectors meet the Bragg condition (with rocking curve)
        rays, norm = self.angle_check(X, rays, norm)
        
        # Perform reflection around normal vector, creating new rays with new
        # origin O = X and new direction D
        O[:]  = X[:]
        D[m] -= 2 * np.einsum('ij,ij->i', D[m], norm[m])[:, np.newaxis] * norm[m]
        
        return rays

    def collect_rays(self, rays):
        X = rays['origin']
        m = rays['mask'].copy()
        
        num_lines = np.sum(m)
        if num_lines > 0:
            # Transform the intersection coordinates from external coordinates
            # to local optical coordinates.
            point_loc = self.point_to_local(X[m])
            
            # Bin the intersections into pixels using integer math.
            pix = np.zeros([num_lines, 3], dtype = int)
            pix = np.round(point_loc / self.pixel_size).astype(int)
            
            # Check to ascertain if origin pixel is even or odd
            if (self.pixel_width % 2) == 0:
                pix_min_x = self.pixel_width//2
            else:
                pix_min_x = (self.pixel_width - 1)//2
                
            if (self.pixel_height % 2) == 0:
                pix_min_y = self.pixel_height//2
            else:
                pix_min_y = (self.pixel_height - 1)//2
            
            pix_min = np.array([pix_min_x, pix_min_y, 0], dtype = int)
            
            # Convert from pixels, which are centered around the origin, to
            # channels, which start from the corner of the optic.
            channel    = np.zeros([num_lines, 3], dtype = int)
            channel[:] = pix[:] - pix_min
            
            # I feel like there must be a faster way to do this than to loop over
            # every intersection.  This could be slow for large arrays.
            for ii in range(len(channel)):
                self.pixel_array[channel[ii,0], channel[ii,1]] += 1
        
        self.photon_count = len(m[m])
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
        super(SphericalCrystal, self).__init__(crystal_input)
        
        self.__name__       = 'SphericalCrystal'
        self.radius         = crystal_input['curvature']
        self.position       = crystal_input['position']
        self.normal         = crystal_input['normal']
        self.xorientation   = crystal_input['orientation']
        self.yorientation   = np.cross(self.normal, self.xorientation)
        self.yorientation  /= np.linalg.norm(self.yorientation)
        self.crystal_spacing= crystal_input['spacing']
        self.rocking_curve  = crystal_input['rocking_curve']
        self.reflectivity   = crystal_input['reflectivity']
        self.width          = crystal_input['width']
        self.height         = crystal_input['height']
        self.center         = self.radius * self.normal + self.position
        self.pixel_size     = self.width / crystal_input['pixel_scaling']
        self.pixel_width    = int(round(self.width / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))
        self.pixel_array    = np.zeros((self.pixel_width, self.pixel_height))
        self.sigma_data     = crystal_input['sigma_data']
        self.pi_data        = crystal_input['pi_data']
        self.mix_factor     = crystal_input['mix_factor']
        
    def spherical_intersect(self, rays):
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
        
        d[m]    = np.sqrt(np.abs(np.einsum('ij,ij->i',L[m] ,L[m]) - t_ca[m]**2))
        t_hc[m] = np.sqrt(np.abs(self.radius**2 - d[m]**2))
        
        t_0[m] = t_ca[m] - t_hc[m]
        t_1[m] = t_ca[m] + t_hc[m]
        
        distance[m] = np.where(t_0[m] > t_1[m], t_0[m], t_1[m])
        return distance
    
    def spherical_norm_generate(self, X, rays):
        m = rays['mask']
        norm = np.zeros(X.shape, dtype=np.float64)
        norm[m] = self.normalize(self.center - X[m])
        return norm

    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.spherical_intersect(rays))
        print(' Rays on Crystal:   {:6.4e}'.format(D[m].shape[0]))        
        rays = self.reflect_vectors(X, rays, self.spherical_norm_generate(X, rays))
        print(' Rays from Crystal: {:6.4e}'.format(D[m].shape[0]))     
        return rays      

class MosaicGraphite(GenericOptic):
    def __init__(self, graphite_input):
        super(MosaicGraphite, self).__init__(graphite_input)
        
        self.__name__       = 'MosaicGraphite'
        self.position       = graphite_input['position']
        self.normal         = graphite_input['normal']
        self.xorientation   = graphite_input['orientation']
        self.yorientation   = np.cross(self.normal, self.xorientation) 
        self.yorientation  /= np.linalg.norm(self.yorientation)
        self.crystal_spacing= graphite_input['spacing']
        self.rocking_curve  = graphite_input['rocking_curve']
        self.reflectivity   = graphite_input['reflectivity']
        self.mosaic_spread  = graphite_input['mosaic_spread']
        self.width          = graphite_input['width']
        self.height         = graphite_input['height']
        self.pixel_size     = self.width / graphite_input['pixel_scaling'] 
        self.pixel_width    = int(round(self.width / self.pixel_size))
        self.pixel_height   = int(round(self.height / self.pixel_size))
        self.pixel_array    = np.zeros((self.pixel_width, self.pixel_height))
        self.center         = self.position
        self.sigma_data     = graphite_input['sigma_data']
        self.pi_data        = graphite_input['pi_data']
        self.mix_factor     = graphite_input['mix_factor']

    def planar_intersect(self, rays):
        #test to see if a ray intersects the mirror plane
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        distance = np.zeros(m.shape, dtype=np.float64)
        distance[m] = np.dot((self.position - O[m]), self.normal)/np.dot(D[m], self.normal)
        
        test = (distance > 0) & (distance < 10)
        distance = np.where(test, distance, 0)
        return distance
    
    def mosaic_norm_generate(self, rays):
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
        normal = np.ones(O.shape) * self.normal
        length = len(m)
        norm = np.empty(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, length)
        
        o_1     = np.cross(normal, [0,0,1])
        o_1[m] /= np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
        o_2     = np.cross(normal, o_1)
        o_2[m] /= np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
        
        R = np.empty((length, 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal
        
        norm = np.einsum('ij,ijk->ik', dir_local, R)
        return norm
    
    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays = self.intersect_check(rays, self.planar_intersect(rays))
        print(' Rays on Graphite:  {:6.4e}'.format(D[m].shape[0]))        
        rays = self.reflect_vectors(X, rays, self.mosaic_norm_generate(rays))
        print(' Rays from Graphite:{:6.4e}'.format(D[m].shape[0]))     
        return rays
    
    
class MosaicGraphiteMesh(TraceObject):
    def __init__(self, graphite_input):
        super(MosaicGraphiteMesh, self).__init__(
             graphite_input['position']
            ,graphite_input['normal']
            ,graphite_input['orientation'])
        
        self.__name__       = 'MosaicGraphite'
        self.mesh_points    = graphite_input['mesh_points']
        self.mesh_faces     = graphite_input['mesh_faces']
        
        self.crystal_spacing= graphite_input['spacing']
        self.rocking_curve  = graphite_input['rocking_curve']
        self.reflectivity   = graphite_input['reflectivity']
        self.mosaic_spread  = graphite_input['mosaic_spread']
        self.sigma_data     = graphite_input['sigma_data']
        self.pi_data        = graphite_input['pi_data']
        self.mix_factor     = graphite_input['mix_factor']
        
        self.bragg_checks   = graphite_input['do_bragg_checks']
        self.miss_checks    = graphite_input['do_miss_checks']
        self.rocking_type   = str.lower(graphite_input['rocking_curve_type'])

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
            #uses barycentric coordinate technique (compute and compare parallelpiped areas)
            tri_area = np.linalg.norm(np.cross((p1 - p2),(p1 - p3)))
            alpha    = self.norm(np.cross((intersect - p2),(intersect - p3))) / tri_area
            beta     = self.norm(np.cross((intersect - p3),(intersect - p1))) / tri_area
            gamma    = self.norm(np.cross((intersect - p1),(intersect - p2))) / tri_area
            
            test |= np.round((alpha + beta + gamma), decimals = 6) == 1.000000
            test &= (distance >= 0)
            
            #append the results to the global impacts arrays
            X[test]    = intersect[test]
            hits[test] = ii + 1
        
        #mask all the rays that missed all faces
        if self.miss_checks is True:
            m[m] &= (hits[m] != 0)
        
        return X, rays, hits
    
    def angle_check(self, X, rays, norm):
        D = rays['direction']
        W = rays['wavelength']
#       w = rays['weight']
        m = rays['mask']
        
        bragg_angle = np.zeros(m.shape, dtype=np.float64)
        dot = np.zeros(m.shape, dtype=np.float64)
        incident_angle = np.zeros(m.shape, dtype=np.float64)
        
        # returns vectors that satisfy the bragg condition
        # only perform check on rays that have intersected the optic
        bragg_angle[m] = np.arcsin( W[m] / (2 * self.crystal_spacing))
        dot[m] = np.abs(np.einsum('ij,ij->i',D[m], -1 * norm[m]))
        incident_angle[m] = (np.pi / 2) - np.arccos(dot[m] / self.norm(D[m]))
        #check which rays satisfy bragg, update mask to remove those that don't
        if self.bragg_checks is True:
            m[m] &= self.rocking_curve_filter(bragg_angle[m], incident_angle[m])
        return rays
    
    def mesh_reflect_vectors(self, X, rays, hits):
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']
        
        norm = np.zeros(O.shape, dtype=np.float64)
        for ii in range(len(self.mesh_faces)):
            #query the triangle mesh grid
            tri  = self.mesh_triangulate(ii)
            test = np.equal(ii, (hits - 1))
            
            #generate the normal vector for each impact location and append it
            norm[test] = self.mesh_mosaic_norm_generate(rays, tri)[test]
            
        # Check which vectors meet the Bragg condition (with rocking curve)
        rays = self.angle_check(X, rays, norm)
        
        # Perform reflection around normal vector, creating new rays with new
        # origin O = X and new direction D
        O[:]  = X[:]
        D[m] -= 2 * np.einsum('ij,ij->i', D[m], norm[m])[:, np.newaxis] * norm[m]
        
        return rays
    
    def mesh_mosaic_norm_generate(self, rays, tri):
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
        normal = np.ones(O.shape) * tri['normal']
        length = len(m)
        norm = np.empty(O.shape)
        rad_spread = np.radians(self.mosaic_spread)
        dir_local = f(rad_spread, length)
        
        o_1     = np.cross(normal, [0,0,1])
        o_1[m] /= np.linalg.norm(o_1[m], axis=1)[:, np.newaxis]
        o_2     = np.cross(normal, o_1)
        o_2[m] /= np.linalg.norm(o_2[m], axis=1)[:, np.newaxis]
        
        R = np.empty((length, 3, 3))
        R[:,0,:] = o_1
        R[:,1,:] = o_2
        R[:,2,:] = normal
        
        norm = np.einsum('ij,ijk->ik', dir_local, R)
        return norm
    
    def light(self, rays):
        D = rays['direction']
        m = rays['mask']
        X, rays, hits = self.mesh_intersect_check(rays)
        print(' Rays on Graphite:  {:6.4e}'.format(D[m].shape[0]))       
        rays = self.mesh_reflect_vectors(X, rays, hits)
        print(' Rays from Graphite:{:6.4e}'.format(D[m].shape[0]))
        return rays

    def collect_rays(self, rays):
        X = rays['origin']
        m = rays['mask'].copy()
        self.photon_count = len(m[m])
        
        """
        num_lines = np.sum(m)
        if num_lines > 0:
            # Transform the intersection coordinates from external coordinates
            # to local optical coordinates.
            point_loc = self.point_to_local(X[m])
            
            # Bin the intersections into pixels using integer math.
            pix = np.zeros([num_lines, 3], dtype = int)
            pix = np.round(point_loc / self.pixel_size).astype(int)
            
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
            channel    = np.zeros([num_lines, 3], dtype = int)
            channel[:] = pix[:] - pix_min
            
            # I feel like there must be a faster way to do this than to loop over
            # every intersection.  This could be slow for large arrays.
            for ii in range(len(channel)):
                self.pixel_array[channel[ii,0], channel[ii,1]] += 1
        
        self.photon_count = len(m[m])
        return self.pixel_array  
        """
        
    def output_image(self, image_name, rotate=None):
        """
        if rotate:
            out_array = np.rot90(self.pixel_array)
        else:
            out_array = self.pixel_array
            
        generated_image = Image.fromarray(out_array)
        generated_image.save(image_name)
        """
        