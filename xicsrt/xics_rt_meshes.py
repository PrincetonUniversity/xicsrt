# -*- coding: utf-8 -*-
"""
Created on Tue Feb 4 13:48:00 2020

Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
A standalone script for generating and visualizing advanced lens geometries
"""
import numpy as np
from scipy.spatial import Delaunay

"""
manfred_input = {}
manfred_input['horz_resolution'] = 11
manfred_input['vert_resolution'] = 11
manfred_input['base_radius'] = 0.300
manfred_input['base_height'] = 0.01
manfred_input['spiral_parameter'] = 0.34
manfred_input['crystal_spacing'] = 2.82868 / 2
"""

def generate_sinusoidal_spiral(manfred_input):
    ## Initial Setup
    #unpack variables
    m   = manfred_input['horz_resolution']
    n   = manfred_input['vert_resolution']
    r_0 = manfred_input['base_radius']
    h_0 = manfred_input['base_height']
    b   = manfred_input['spiral_parameter']
    d   = manfred_input['crystal_spacing']
    
    #constants for unit conversion
    conv1 = 12.398425
    conv2 = np.pi/180
    
    ## Solve the Sinusoidal Spiral
    energy     = np.linspace(9.750, 10.750, m)
    #energy     = np.linspace(9.0, 100.0, m)
    wavelength = conv1 / energy
    theta      = np.arcsin(wavelength / (2 * d))
    
    #b is the sinusoidal spiral constant; it shouldn't equal 1
    a = b - 1.0
    c = 1.0 / a
    
    #solve for the sinusoidal spiral in polar coordinates
    phi   = (theta - theta[0]) / a
    alpha = theta + phi
    r     = r_0 * np.power((np.sin(theta) / np.sin(theta[0])), c)
    
    #convert polar to cartesian coordinates
    spiral      = np.zeros([m,3])
    spiral[:,0] = r * np.cos(phi)
    spiral[:,1] = r * np.sin(phi)
    spiral[:,2] = 0.0
    
    ## Solve the Lens Metadata
    #calculate crystal length by adding up the lengths of individual segments
    c_length = 0.0
    for j in range(m - 1):
        c_length += r[j] * (phi[j+1] - phi[j]) / np.sin(theta[j])
    
    #calculate crystal radius of curvature rho and centers of curvature
    rho = r / ((a + 1) * np.sin(theta))
    
    center      = np.zeros([m,3])
    center[:,0] = spiral[:,0] - rho * np.sin(theta)
    center[:,1] = spiral[:,1] + rho * np.cos(theta)
    center[:,2] = 0.0
    
    #calculate detector points and detector length
    detector_points = np.zeros([m,3], dtype = np.float64)
    detector_points[:,0] = rho[:] * np.sin(theta[:]) * np.cos(alpha[:] + theta[:])
    detector_points[:,1] = rho[:] * np.sin(theta[:]) * np.sin(alpha[:] + theta[:])
    detector_points[:,2] = 0.0
    detector_points += spiral
    
    d_length = np.linalg.norm(detector_points[-1,:] - detector_points[0,:])
    
    #create vectors from the curvature centers to the lens
    normal = spiral - center
    
    # Solve for Mesh Geometry
    #use the normal vectors to calculate the mesh geometry
    height    = np.zeros([m,n], dtype = np.float64)
    height[:] = np.linspace(-h_0/2, h_0/2, n)
    theta     = np.repeat(theta[:,np.newaxis], n, axis = 1)
    rho       = np.repeat(rho[:,np.newaxis], n, axis = 1)
    beta      = np.arcsin(height / rho)
    
    #calculate mesh points
    mesh_points = np.zeros([m,n,3], dtype = np.float64)
    mesh_params = np.zeros([m,n,2], dtype = np.float64)
    mesh_params[:,:,0] = theta
    mesh_params[:,:,1] = beta
    
    z_vector    = np.array([0.0,0.0,1.0])
    for ii in range(m):
        for jj in range(n):
            mesh_points[ii,jj,:] = (normal[ii,:] * np.cos(beta[ii,jj]) +
                                    z_vector[:]  * np.sin(beta[ii,jj]) +
                                    center[ii])      
    
    #calculate mesh faces from the 2D (theta, beta) parameter space
    #this works because there is a 1:1 mapping from 2D parameter space 
    #to 3D point space, and since Delaunay works better in 2D
    mesh_points = mesh_points.reshape((m*n),3)
    mesh_params = mesh_params.reshape((m*n),2)
    mesh_faces  = Delaunay(mesh_params).simplices
    
    #calculate the normal of each mesh face
    p1 = mesh_points[mesh_faces[:,0],:]
    p2 = mesh_points[mesh_faces[:,1],:]
    p3 = mesh_points[mesh_faces[:,2],:]
    mesh_normals  = np.cross((p1 - p2),(p3 - p2), axis = 1)
    mesh_normals /= np.linalg.norm(mesh_normals , axis = 0)
    
    ## Pack Up the Variables
    manfred_output = {}
    manfred_output['mesh_points']     = mesh_points
    manfred_output['mesh_faces']      = mesh_faces
    manfred_output['mesh_normals']    = mesh_normals
    manfred_output['detector_points'] = detector_points
    manfred_output['c_length']        = c_length
    manfred_output['d_length']        = d_length
    
    return manfred_output
