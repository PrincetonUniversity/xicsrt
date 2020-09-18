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

def generate_rays(origin, targets, wavelength):
    O = np.ones([9,3], dtype = np.float64)
    D = np.ones([9,3], dtype = np.float64)
    W = np.ones([9], dtype=np.float64)
    w = np.ones([9], dtype=np.float64)
    m = np.ones([9], dtype=np.bool)
    
    O *= origin
    W *= wavelength
    D  = targets - O
    D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
    
    rays = {'origin': O, 'direction': D, 'wavelength': W, 'weight': w, 'mask': m}
    return rays

def compute_metadata(norm, rays, rays_history):
    dot = np.einsum('ij,ij->i', rays['direction'], -1 * norm)
    incident_angle = -(np.pi / 2) + np.arccos(dot)[:,np.newaxis]
    disp = rays_history[-1]['origin'] - rays_history[-2]['origin']
    dist = np.linalg.norm(disp, axis = 1)[:,np.newaxis]
    
    metadata = {'incident_angle': incident_angle, 'distance': dist}
    return metadata

def analytical_model(source, crystal, graphite, detector, 
                     source_input, graphite_input, crystal_input,
                     detector_input, general_input):
    profiler.start('Analytical Model')
    print('')
    print('Running Analytical Model')
    
    """
    Analytical model uses fewer rays, but calculates more details about each
    ray, and targets each ray to a specific location.
    
    Set full_propagate to True to make the system conserve rays and propagate 
    the initial 9 generated rays through the whole system. Set it to False to
    make the system generate a new set of 9 rays for every optical element.
    """
    ## Initial variable setup
    rays_history = []
    rays_metadata= []
    full_propagate = True
    
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
    position[0,:] = source_input['position']
    position[1,:] = graphite_input['position']
    position[2,:] = crystal_input['position']
    position[3,:] = detector_input['position']
    #normal[Optical Element Number, 3D Coordinates]
    normal[0,:] = source_input['normal']
    normal[1,:] = graphite_input['normal']
    normal[2,:] = crystal_input['normal']
    normal[3,:] = detector_input['normal']
    #orient_x[Optical Element Number, 3D Coordinates]
    orient_x[0,:] = source_input['orientation']
    orient_x[1,:] = graphite_input['orientation']
    orient_x[2,:] = crystal_input['orientation']
    orient_x[3,:] = detector_input['orientation']
    #orient_y[Optical Element Number, 3D Coordinates]
    orient_y[0,:] = np.cross(normal[0,:], orient_x[0,:]) 
    orient_y[1,:] = np.cross(normal[1,:], orient_x[1,:]) 
    orient_y[2,:] = np.cross(normal[2,:], orient_x[2,:]) 
    orient_y[3,:] = np.cross(normal[3,:], orient_x[3,:])
    #width[Optical Element Number]
    width[0] = source_input['width'] * 0.8
    width[1] = graphite_input['width'] * 0.8
    width[2] = crystal_input['width'] * 0.8
    width[3] = detector_input['pixel_size'] * detector_input['horizontal_pixels'] * 0.8
    #height[Optical Element Number]
    height[0] = source_input['height'] * 0.8
    height[1] = graphite_input['height'] * 0.8
    height[2] = crystal_input['height'] * 0.8
    height[3] = detector_input['pixel_size'] * detector_input['vertical_pixels'] * 0.8
    
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
    if general_input['backwards_raytrace'] is True:
        #launch the rays from source to crystal
        rays = generate_rays(position[0,:], targets[2,:,:], 
                             source_input['wavelength'])
    if general_input['backwards_raytrace'] is False:
        #launch the rays from source to graphite
        rays = generate_rays(position[0,:], targets[1,:,:], 
                             source_input['wavelength'])
    rays_history.append(deepcopy(rays))
    
    # launch the rays at the first optical element
    if general_input['backwards_raytrace'] is True:
        rays = crystal.trace(rays)
        crystal.collect_rays(rays)
        norm = crystal.normalize(crystal.center - targets[2,:,:])
    if general_input['backwards_raytrace'] is False:
        rays = graphite.trace(rays)
        graphite.collect_rays(rays)
        norm = np.ones([9,3], dtype = np.float64) * graphite.normal
    
    # update ray metadata after the first impact
    rays_history.append(deepcopy(rays))
    rays_metadata.append(compute_metadata(norm, rays, rays_history))
    
    ## Generate rays from first element and aim them at the targets on the second element
    #rays are generated from a point source regardless of source dimensions
    if full_propagate is False:
        if general_input['backwards_raytrace'] is True:
            #launch the rays from crystal to graphite
            rays = generate_rays(position[2,:], targets[1,:,:], 
                                 source_input['wavelength'])
        if general_input['backwards_raytrace'] is False:
            #launch the rays from graphite to crystal
            rays = generate_rays(position[1,:], targets[2,:,:], 
                                 source_input['wavelength'])
        rays_history.append(deepcopy(rays))
        rays_metadata.append(compute_metadata(norm, rays, rays_history))
    
    # launch the rays at the second optical element
    if general_input['backwards_raytrace'] is True:
        rays = graphite.trace(rays)
        graphite.collect_rays(rays)
        norm = np.ones([9,3], dtype = np.float64) * graphite.normal
    if general_input['backwards_raytrace'] is False:
        rays = crystal.trace(rays)
        crystal.collect_rays(rays)
        norm = crystal.normalize(crystal.center - targets[2,:,:])
    
    # update ray metadata after the second impact
    rays_history.append(deepcopy(rays))
    rays_metadata.append(compute_metadata(norm, rays, rays_history))
    
    ## Generate rays from second element and aim them at the targets on the detector
    #rays are generated from a point source regardless of source dimensions
    if full_propagate is False:
        if general_input['backwards_raytrace'] is True:
            #launch the rays from graphite to detector
            rays = generate_rays(position[1,:], targets[3,:,:], 
                                 source_input['wavelength'])
        if general_input['backwards_raytrace'] is False:
            #launch the rays from crystal to detector
            rays = generate_rays(position[2,:], targets[3,:,:], 
                                 source_input['wavelength'])
        norm = np.ones([9,3], dtype = np.float64) * detector.normal
        rays_history.append(deepcopy(rays))
        rays_metadata.append(compute_metadata(norm, rays, rays_history))

    detector.collect_rays(rays)
    norm = np.ones([9,3], dtype = np.float64) * detector.normal
    rays_metadata.append(compute_metadata(norm, rays, rays_history))

    return rays_history, rays_metadata
