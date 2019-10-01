# -*- coding: utf-8 -*-
"""
Created on Tue Oct 01 13:33:28 2019

Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>
  
Description
-----------
This script runs a greatly reduced but highly detailed analysis of the current
XICS detector geometry using a heavily modified raytrace algorithm.
"""

from xicsrt.util import profiler
import numpy as np
from copy import deepcopy

def analytical_model(source, crystal, graphite, pilatus, 
                     source_input, graphite_input, crystal_input,
                     detector_input, general_input):
    profiler.start('Analytical Model')
    print('')
    print('Running Analytical Model')
    
    """
    Analytical model uses fewer rays, but calculates more details about each
    ray, and targets each ray to a specific location.
    """
    ## Initial variable setup
    rays_history = []
    
    position = np.zeros([4,3], dtype = np.float64)
    normal   = np.zeros([4,3], dtype = np.float64)
    orient_x = np.zeros([4,3], dtype = np.float64)
    orient_y = np.zeros([4,3], dtype = np.float64)
    width    = np.zeros([4], dtype = np.float64)[:,np.newaxis]
    height   = np.zeros([4], dtype = np.float64)[:,np.newaxis]
    
    targets = np.zeros([4,9,3], dtype = np.float64)
    
    ## Pipe in properties of each optical element
    #for slicing puposes, each optical element now has a number
    #uses same geometrical math as visualizer
    #source = 0, graphite = 1, crystal = 2, detector = 3
    
    #position[Optical Element Number, 3D Coordinates]
    position[0,:] = source_input['source_position']
    position[1,:] = graphite_input['graphite_position']
    position[2,:] = crystal_input['crystal_position']
    position[3,:] = detector_input['detector_position']
    #normal[Optical Element Number, 3D Coordinates]
    normal[0,:] = source_input['source_normal']
    normal[1,:] = graphite_input['graphite_normal']
    normal[2,:] = crystal_input['crystal_normal']
    normal[3,:] = detector_input['detector_normal']
    #orient_x[Optical Element Number, 3D Coordinates]
    orient_x[0,:] = source_input['source_orientation']
    orient_x[1,:] = graphite_input['graphite_orientation']
    orient_x[2,:] = crystal_input['crystal_orientation']
    orient_x[3,:] = detector_input['detector_orientation']
    #orient_y[Optical Element Number, 3D Coordinates]
    orient_y[0,:] = np.cross(normal[0,:], orient_x[0,:]) 
    orient_y[1,:] = np.cross(normal[1,:], orient_x[1,:]) 
    orient_y[2,:] = np.cross(normal[2,:], orient_x[2,:]) 
    orient_y[3,:] = np.cross(normal[3,:], orient_x[3,:])
    #width[Optical Element Number]
    width[0] = source_input['source_width']
    width[1] = graphite_input['graphite_width'] 
    width[2] = crystal_input['crystal_width']
    width[3] = detector_input['pixel_size'] * detector_input['horizontal_pixels']
    #height[Optical Element Number]
    height[0] = source_input['source_height']
    height[1] = graphite_input['graphite_height']
    height[2] = crystal_input['crystal_height']
    height[3] = detector_input['pixel_size'] * detector_input['vertical_pixels']
    
    ## Generate initial target points for each optical element
    #target 0 is the center, 1-4 are  edges, 5-8 are corners
    #targets[Optical Element Number, Point Number, 3D Coordinates]
    targets[:,0,:] = position[:,:]
    targets[:,1,:] = position[:,:] - (width[:] * orient_x[:,:] / 2)
    targets[:,2,:] = position[:,:] + (width[:] * orient_x[:,:] / 2)
    targets[:,3,:] = position[:,:] - (height[:] * orient_y[:,:] / 2)
    targets[:,4,:] = position[:,:] + (height[:] * orient_y[:,:] / 2)
    targets[:,5,:] = (position[:,:] - (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    targets[:,6,:] = (position[:,:] + (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    targets[:,7,:] = (position[:,:] + (width[:] * orient_x[:,:] / 2) - (height[:] * orient_y[:,:] / 2))  
    targets[:,8,:] = (position[:,:] - (width[:] * orient_x[:,:] / 2) - (height[:] * orient_y[:,:] / 2))
    
    ## Generate rays from source and aim them at the targets on the first element
    #rays are generated from a point source regardless of source dimensions
    O = np.ones([9,3], dtype = np.float64)
    D = np.zeros([9,3], dtype = np.float64)
    W = np.ones([9], dtype=np.float64)
    w = np.ones([9], dtype=np.float64)
    m = np.ones([9], dtype=np.bool)
    
    O *= position[0,:]
    W *= source_input['source_wavelength']
    
    if general_input['backwards_raytrace'] is True:
        D = targets[2,:,:] - O[:,:]
    if general_input['backwards_raytrace'] is False:
        D = targets[1,:,:] - O[:,:]
    D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
    
    rays = {'origin': O, 'direction': D, 'wavelength': W, 'weight': w, 'mask': m}
    rays_history.append(deepcopy(rays))
    
    # launch the rays at the first optical element
    if general_input['backwards_raytrace'] is True:
        rays = crystal.light(rays)
        norm = crystal.normalize(crystal.center - targets[2,:,:])
    if general_input['backwards_raytrace'] is False:
        rays = graphite.light(rays)
        norm = np.ones([9,3], dtype = np.float64) * graphite.normal
    dot = np.einsum('ij,ij->i', rays['direction'], -1 * norm)
    incident_angle = -(np.pi / 2) + np.arccos(dot)[:,np.newaxis]
    
    rays_history.append(deepcopy(rays))

    disp = rays_history[1]['origin'] - rays_history[0]['origin']
    dist = np.linalg.norm(disp, axis = 1)[:,np.newaxis]
    
    print(dist)
    print(incident_angle)
    
    
    
    return rays_history