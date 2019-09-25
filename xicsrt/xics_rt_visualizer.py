# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:55:01 2019

Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
This script takes in all of the input parameters from the raytracer and makes
a 3D visualization of the X-Ray optics setup
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_layout(general_input, source_input, graphite_input, crystal_input,
                     detector_input):
    ## Setup plot and axes
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    """
    ax.set_xlim(0,3.5)
    ax.set_ylim(-2, 2)
    """
    ax.set_zlim(-1, 1)
    plt.title("X-Ray Optics Layout")
    
    ## Setup variables, described below
    position = np.zeros([3,4], dtype = np.float64)
    normal   = np.zeros([3,4], dtype = np.float64)
    orient_x = np.zeros([3,4], dtype = np.float64)
    orient_y = np.zeros([3,4], dtype = np.float64)
    width    = np.zeros([4], dtype = np.float64)
    height   = np.zeros([4], dtype = np.float64)
    
    corners = np.zeros([3,5,4], dtype = np.float64)
    
    beamline = np.zeros([3,4], dtype = np.float64)
    
    circle_points   = np.linspace(0, np.pi * 2, 36)[:,np.newaxis]
    crystal_circle  = np.zeros([3,36], dtype = np.float64)
    rowland_circle  = np.zeros([3,36], dtype = np.float64)
    
    meridi_line = np.zeros([3,2], dtype = np.float64)
    saggit_line = np.zeros([3,2], dtype = np.float64)
    
    ## Define variables
    #for slicing puposes, each optical element now has a number
    #source = 0, graphite = 1, crystal = 2, detector = 3
    
    #position[3D Coordinates, Optical Element Number]
    position[:,0] = source_input['source_position']
    position[:,1] = graphite_input['graphite_position']
    position[:,2] = crystal_input['crystal_position']
    position[:,3] = detector_input['detector_position']
    #normal[3D Coordinates, Optical Element Number]
    normal[:,0] = source_input['source_normal']
    normal[:,1] = graphite_input['graphite_normal']
    normal[:,2] = crystal_input['crystal_normal']
    normal[:,3] = detector_input['detector_normal']
    #orient_x[3D Coordinates, Optical Element Number]
    orient_x[:,0] = source_input['source_orientation']
    orient_x[:,1] = graphite_input['graphite_orientation']
    orient_x[:,2] = crystal_input['crystal_orientation']
    orient_x[:,3] = detector_input['detector_orientation']
    #orient_y[3D Coordinates, Optical Element Number]
    orient_y[:,0] = np.cross(normal[:,0], orient_x[:,0]) 
    orient_y[:,1] = np.cross(normal[:,1], orient_x[:,1]) 
    orient_y[:,2] = np.cross(normal[:,2], orient_x[:,2]) 
    orient_y[:,3] = np.cross(normal[:,3], orient_x[:,3])
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
    #crystal optical properties [Float64]
    crystal_bragg = crystal_input['crystal_bragg']
    meridi_focus  = crystal_input['meridi_focus']
    saggit_focus  = crystal_input['saggit_focus']
    
    ## Create Bounding Boxes
    #3D coordinates of the four corners of each optical element 
    #The 5th corner is a duplicate of the 1st, it closes the bounding box
    #corners[3D Coodrinates, Corner Number, Optical Element Number]
    corners[:,0,:] = (position[:,:] - (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    corners[:,1,:] = (position[:,:] + (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    corners[:,2,:] = (position[:,:] + (width[:] * orient_x[:,:] / 2) - (height[:] * orient_y[:,:] / 2))  
    corners[:,3,:] = (position[:,:] - (width[:] * orient_x[:,:] / 2) - (height[:] * orient_y[:,:] / 2))  
    corners[:,4,:] = (position[:,:] - (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    
    ## The line connecting the centers of all optical elements
    #beamline[3D Coodrinates, Optical Element Number]
    if general_input['backwards_raytrace'] is False:
        beamline[:,:] = position[:,:]
    
    elif general_input['backwards_raytrace'] is True:
        beamline[:,0] = position[:,3]
        beamline[:,1] = position[:,1]
        beamline[:,2] = position[:,2]
        beamline[:,3] = position[:,0]
        
    ## The crystal's radius of curvature and Rowland circle
    #crystal_center[3D Coodrinates]
    crystal_center  =(crystal_input['crystal_curvature'] 
                    * crystal_input['crystal_normal']
                    + crystal_input['crystal_position'])
    #crystal_circle[3D Coodrinates, Point Number], 36 evenly-spaced points
    crystal_circle  = crystal_input['crystal_curvature'] * (
            (orient_y[:,2] * np.cos(circle_points)) + (normal[:,2] * np.sin(circle_points)))
    crystal_circle += crystal_center
    
    rowland_circle  = crystal_input['crystal_curvature'] * np.cos(crystal_bragg) * (
            (orient_y[:,2] * np.cos(circle_points)) + (normal[:,2] * np.sin(circle_points)))
    rowland_circle += crystal_center
    
    ## The crystal's saggital and meridional foci
    inbound_vector = position[:,1] - position[:,2]
    inbound_vector/= np.linalg.norm(inbound_vector)
    #meridi_line[3D Coodinates, Point Number], 2 points (one above, one below)
    meridi_line[:,0] = position[:,2] + meridi_focus * inbound_vector + 0.1 * orient_x[:,2]
    meridi_line[:,1] = position[:,2] + meridi_focus * inbound_vector - 0.1 * orient_x[:,2]
    saggit_line[:,0] = position[:,2] + saggit_focus * inbound_vector + 0.1 *   normal[:,2]
    saggit_line[:,1] = position[:,2] + saggit_focus * inbound_vector - 0.1 *   normal[:,2]
    
    ## Plot everything
    #position points
    ax.scatter(position[0,0], position[1,0], position[2,0], color = "yellow")
    ax.scatter(position[0,1], position[1,1], position[2,1], color = "grey")
    ax.scatter(position[0,2], position[1,2], position[2,2], color = "cyan")
    ax.scatter(position[0,3], position[1,3], position[2,3], color = "red")
    
    #normal vectors
    ax.quiver(position[0,0], position[1,0], position[2,0],
              normal[0,0]  , normal[1,0]  , normal[2,0]  ,
              color = "yellow", length = 0.1, arrow_length_ratio = 0.1)
    ax.quiver(position[0,1], position[1,1], position[2,1],
              normal[0,1]  , normal[1,1]  , normal[2,1]  ,
              color = "grey", length = 0.1, arrow_length_ratio = 0.1)
    ax.quiver(position[0,2], position[1,2], position[2,2],
              normal[0,2]  , normal[1,2]  , normal[2,2]  ,
              color = "cyan", length = 0.1, arrow_length_ratio = 0.1)
    ax.quiver(position[0,3], position[1,3], position[2,3],
              normal[0,3]  , normal[1,3]  , normal[2,3]  ,
              color = "red", length = 0.1 , arrow_length_ratio = 0.1)
    
    #beamline
    ax.plot3D(beamline[0,:], beamline[1,:], beamline[2,:], "black")
    
    #bounding boxes
    ax.plot3D(corners[0,:,0], corners[1,:,0], corners[2,:,0], color = "yellow")
    ax.plot3D(corners[0,:,1], corners[1,:,1], corners[2,:,1], color = "grey")
    ax.plot3D(corners[0,:,2], corners[1,:,2], corners[2,:,2], color = "cyan")
    ax.plot3D(corners[0,:,3], corners[1,:,3], corners[2,:,3], color = "red")
    
    #circles (NOTE: the circle arrays are sliced differently than other arrays)
    ax.plot3D(crystal_circle[:,0], crystal_circle[:,1], crystal_circle[:,2], color = "blue")
    ax.plot3D(rowland_circle[:,0], rowland_circle[:,1], rowland_circle[:,2], color = "blue")
    
    #foci
    ax.plot3D(meridi_line[0,:], meridi_line[1,:], meridi_line[2,:], color = "blue")
    ax.plot3D(saggit_line[0,:], saggit_line[1,:], saggit_line[2,:], color = "blue")
    
    return plt, ax
    
def visualize_vectors(output, general_input, source_input, graphite_input,
                      crystal_input, detector_input):
    ## Do all of the steps as before, but also add the output rays
    origin = output['origin']
    direct = output['direction']
    m      = output['mask']
    
    print(len(m[m]))
    #to avoid plotting too many rays, randomly cull rays until there are 1000
    if len(origin[m]) > 1000:
        m[m] &= (np.random.randint(0, len(origin[m])) < 1000)
    print(len(m[m]))
    
    plt, ax = visualize_layout(general_input, source_input, graphite_input, 
                               crystal_input, detector_input)
    plt.title("X-Ray Raytracing Results")    
    
    ax.quiver(origin[m,0], origin[m,1], origin[m,2],
              direct[m,0], direct[m,1], direct[m,2],
              length = 1.0, arrow_length_ratio = 0.1 , color = "green",
              normalize = True)
    
    return plt, ax