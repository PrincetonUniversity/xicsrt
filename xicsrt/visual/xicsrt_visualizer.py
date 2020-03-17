# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:55:01 2019

Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
This script takes in all of the input parameters from the raytracer and makes
a 3D visualization of the X-Ray optics setup using matplotlib Axes3D
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from PIL import Image

from xicsrt.xicsrt_math import bragg_angle
import stelltools

def visualize_layout(config):
    ## Setup plot and axes
    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    
    plt.title("X-Ray Optics Layout")
    
    ## Setup variables, described below
    origin          = np.zeros([5,3], dtype = np.float64)
    normal          = np.zeros([5,3], dtype = np.float64)
    orient_x        = np.zeros([5,3], dtype = np.float64)
    orient_y        = np.zeros([5,3], dtype = np.float64)
    width           = np.zeros([5], dtype = np.float64)[:,np.newaxis]
    height          = np.zeros([5], dtype = np.float64)[:,np.newaxis]
    corners         = np.zeros([5,5,3], dtype = np.float64)
    
    ## Define variables
    scenario        = config['general_input']['scenario']
    graphite_mesh   = config['graphite_input']['use_meshgrid']
    crystal_mesh    = config['crystal_input']['use_meshgrid']
    #for slicing puposes, each optical element now has a number
    #source = 0, graphite = 1, crystal = 2, detector = 3, plasma = 4
    #origin[Optical Element Number, 3D Coordinates]
    origin[0,:]   = config['source_input']['origin']
    origin[1,:]   = config['graphite_input']['origin']
    origin[2,:]   = config['crystal_input']['origin']
    origin[3,:]   = config['detector_input']['origin']
    origin[4,:]   = config['plasma_input']['origin']
    #zaxis[Optical Element Number, 3D Coordinates]
    normal[0,:]     = config['source_input']['zaxis']
    normal[1,:]     = config['graphite_input']['zaxis']
    normal[2,:]     = config['crystal_input']['zaxis']
    normal[3,:]     = config['detector_input']['zaxis']
    normal[4,:]     = config['plasma_input']['zaxis']
    #orient_x[Optical Element Number, 3D Coordinates]
    orient_x[0,:]   = config['source_input']['xaxis']
    orient_x[1,:]   = config['graphite_input']['xaxis']
    orient_x[2,:]   = config['crystal_input']['xaxis']
    orient_x[3,:]   = config['detector_input']['xaxis']
    orient_x[4,:]   = config['plasma_input']['xaxis']
    #orient_y[Optical Element Number, 3D Coordinates]
    orient_y[0,:]   = np.cross(normal[0,:], orient_x[0,:]) 
    orient_y[1,:]   = np.cross(normal[1,:], orient_x[1,:]) 
    orient_y[2,:]   = np.cross(normal[2,:], orient_x[2,:]) 
    orient_y[3,:]   = np.cross(normal[3,:], orient_x[3,:])
    orient_y[4,:]   = np.cross(normal[4,:], orient_x[4,:])
    
    orient_y[0,:]  /= np.linalg.norm(orient_y[0,:])
    orient_y[1,:]  /= np.linalg.norm(orient_y[1,:])
    orient_y[2,:]  /= np.linalg.norm(orient_y[2,:])
    orient_y[3,:]  /= np.linalg.norm(orient_y[3,:])
    orient_y[4,:]  /= np.linalg.norm(orient_y[4,:])
    #width[Optical Element Number]
    width[0]        = config['source_input']['width']
    width[1]        = config['graphite_input']['width'] 
    width[2]        = config['crystal_input']['width']
    width[3]        = config['detector_input']['width']
    width[4]        = config['plasma_input']['width']
    #height[Optical Element Number]
    height[0]       = config['source_input']['height']
    height[1]       = config['graphite_input']['height']
    height[2]       = config['crystal_input']['height']
    height[3]       = config['detector_input']['height']
    height[4]       = config['plasma_input']['height']
    #corners[Optical Element Number, Corner Number, 3D Coordinates]
    corners[:,0,:]  = (origin[:,:] - (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    corners[:,1,:]  = (origin[:,:] + (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))   
    corners[:,2,:]  = (origin[:,:] + (width[:] * orient_x[:,:] / 2) - (height[:] * orient_y[:,:] / 2))  
    corners[:,3,:]  = (origin[:,:] - (width[:] * orient_x[:,:] / 2) - (height[:] * orient_y[:,:] / 2))  
    corners[:,4,:]  = (origin[:,:] - (width[:] * orient_x[:,:] / 2) + (height[:] * orient_y[:,:] / 2))
    #The 5th corner is a duplicate of the 1st, it closes the bounding box
    
    ## Compactify variables into a dictionary
    config_vis = {'origin':origin,'normal':normal,'orient_x':orient_x,
                  'orient_y':orient_y,'width':width,'height':height,
                  'corners':corners}

    ## Plot everything
    if scenario == "REAL" or scenario == "PLASMA":
        #resize and recenter axes on graphite
        view_center = 1
        
        #plasma sightline[3D Coordinates]
        plasma_sight = config['filter_input']['origin'] + 7 * config['filter_input']['direction']
        
        #draw beamline
        beamline = np.zeros([4,3], dtype = np.float64)
        if config['general_input']['backwards_raytrace'] is False:
            beamline[0,:] = plasma_sight
            beamline[1,:] = origin[1,:]
            beamline[2,:] = origin[2,:]
            beamline[3,:] = origin[3,:]
            
        if config['general_input']['backwards_raytrace'] is True:
            beamline[0,:] = origin[3,:]
            beamline[1,:] = origin[2,:]
            beamline[2,:] = origin[1,:]
            beamline[3,:] = plasma_sight
                    
        #draw plasma, graphite, crystal, detector
        draw_flux(config, ax)
        draw_graphite(config, config_vis, graphite_mesh, ax)
        draw_crystal(config, config_vis, crystal_mesh, True, ax)
        draw_detector(config, config_vis, ax)
    
    elif scenario == "THROUGHPUT":
        #resize and recenter axes on plasma
        view_center = 4
        
        #draw beamline
        beamline = np.zeros([4,3], dtype = np.float64)
        if config['general_input']['backwards_raytrace'] is False:
            beamline[0,:] = origin[4,:]
            beamline[1,:] = origin[1,:]
            beamline[2,:] = origin[2,:]
            beamline[3,:] = origin[3,:]
            
        if config['general_input']['backwards_raytrace'] is True:
            beamline[0,:] = origin[3,:]
            beamline[1,:] = origin[2,:]
            beamline[2,:] = origin[1,:]
            beamline[3,:] = origin[4,:]
                    
        #draw plasma, graphite, crystal, detector
        draw_torus(config, config_vis, ax)
        draw_graphite(config, config_vis, graphite_mesh, ax)
        draw_crystal(config, config_vis, crystal_mesh, True, ax)
        draw_detector(config, config_vis, ax)
        
    elif scenario == "BEAM" or scenario == "MODEL":
        #resize and recenter axes on source
        view_center = 1
       
        #draw beamline
        beamline = np.zeros([4,3], dtype = np.float64)
        if config['general_input']['backwards_raytrace'] is False:
            beamline[0,:] = origin[0,:]
            beamline[1,:] = origin[1,:]
            beamline[2,:] = origin[2,:]
            beamline[3,:] = origin[3,:]
            
        if config['general_input']['backwards_raytrace'] is True:
            beamline[0,:] = origin[3,:]
            beamline[1,:] = origin[2,:]
            beamline[2,:] = origin[1,:]
            beamline[3,:] = origin[0,:]
                    
        #draw source, graphite, crystal, detector
        draw_source(config, config_vis, ax)
        draw_graphite(config, config_vis, graphite_mesh, ax)
        draw_crystal(config, config_vis, crystal_mesh, True, ax)
        draw_detector(config, config_vis, ax)
        
    elif scenario == "MANFRED":
        #resize and recenter axes on crystal
        view_center = 2
        
        #draw beamline
        beamline = np.zeros([3,3], dtype = np.float64)
        beamline[0,:] = origin[0,:]
        beamline[1,:] = origin[2,:]
        beamline[2,:] = origin[3,:]
        
        #draw plasma, graphite, crystal, detector
        draw_source(config, config_vis, ax)
        draw_crystal(config, config_vis, crystal_mesh, False, ax)
        draw_detector(config, config_vis, ax)
        
    elif scenario == "CRYSTAL":
        #resize and recenter axes on crystal
        view_center = 2
        
        #draw beamline
        beamline = np.zeros([3,3], dtype = np.float64)
        beamline[0,:] = origin[0,:]
        beamline[1,:] = origin[2,:]
        beamline[2,:] = origin[3,:]
        
        #draw plasma, graphite, crystal, detector
        draw_source(config, config_vis, ax)
        draw_crystal(config, config_vis, crystal_mesh, True, ax)
        draw_detector(config, config_vis, ax)
    
    elif scenario == "GRAPHITE":
        #resize and recenter axes on graphite
        view_center = 1     
        
        #draw beamline
        beamline = np.zeros([3,3], dtype = np.float64)
        beamline[0,:] = origin[0,:]
        beamline[1,:] = origin[1,:]
        beamline[2,:] = origin[3,:]

        #draw source, graphite, detector
        draw_source(config, config_vis, ax)
        draw_graphite(config, config_vis, graphite_mesh, ax)
        draw_detector(config, config_vis, ax)
        
    elif scenario == "SOURCE":
        #resize and recenter axes on source
        view_center = 0     
        
        #draw beamline
        beamline = np.zeros([2,3], dtype = np.float64)
        beamline[0,:] = origin[0,:]
        beamline[1,:] = origin[3,:]
        
        #draw source, detector
        draw_source(config, config_vis, ax)
        draw_detector(config, config_vis, ax)   
    
    scale = 10
    ax.plot3D(beamline[:,0], beamline[:,1], beamline[:,2], "black", zorder = 5)
    ax.set_xlim(origin[view_center,0] - scale, origin[view_center,0] + scale)
    ax.set_ylim(origin[view_center,1] - scale, origin[view_center,1] + scale)
    ax.set_zlim(origin[view_center,2] - scale, origin[view_center,2] + scale)

    return plt, ax

def draw_flux(config, ax):
    stelltools.initialize_from_wout(config['plasma_input']['wout_file'])
    
    #flux toroid resolution (number of points)
    num_r = 1
    num_m = 25
    num_n = 25

    num_points = num_r * num_m * num_n
    flux_points = np.empty((num_points, 3))
    
    #build the flux surface
    for nn in range(num_n):
        for mm in range(num_m):
            for rr in range(num_r):
                index = rr + (mm * num_r) + (nn * num_r * num_m)
                
                flux_points[index, 0] = 1.0
                flux_points[index, 1] = 2 * np.pi / (num_m - 1) * mm
                flux_points[index, 2] = np.pi / (num_n - 1)* nn
    
    #convert to cartesian coordinates from flux coordinates
    mesh_points = np.empty(flux_points.shape)
    for ii in range(flux_points.shape[0]):
        mesh_points[ii,:] = stelltools.car_from_flx(flux_points[ii,:])
        
    #build the plasma mesh grid
    x_array = mesh_points[:,0]
    y_array = mesh_points[:,1]
    z_array = mesh_points[:,2]
    
    mesh_faces  = Delaunay(flux_points[:,1:]).simplices
    
    triangles = tri.Triangulation(x_array, y_array, mesh_faces)
    ax.plot_trisurf(triangles, z_array, color = 'yellow', alpha = 0.25, zorder = 0)
    
    #plot the plasma bounding box
    plasma_corners = np.zeros([8,3], dtype = np.float64)
    dx = config['plasma_input']['width']  / 2
    dy = config['plasma_input']['height'] / 2
    dz = config['plasma_input']['depth']  / 2
    
    plasma_corners[0,:] = [ dx, dy, dz]
    plasma_corners[1,:] = [-dx, dy, dz]
    plasma_corners[2,:] = [ dx,-dy, dz]
    plasma_corners[3,:] = [-dx,-dy, dz]
    plasma_corners[4,:] = [ dx, dy,-dz]
    plasma_corners[5,:] = [-dx, dy,-dz]
    plasma_corners[6,:] = [ dx,-dy,-dz]
    plasma_corners[7,:] = [-dx,-dy,-dz]
    plasma_corners     += config['plasma_input']['origin']
    
    ax.scatter(plasma_corners[:,0], plasma_corners[:,1], plasma_corners[:,2], color = "yellow")
    
    return ax

def draw_torus(config, ax):
    major_radius = config['plasma_input']['major_radius']
    minor_radius = config['plasma_input']['minor_radius']
    angle = np.linspace(0, np.pi * 2, 36)
    theta, phi = np.meshgrid(angle, angle)
    
    x_array = (major_radius + (minor_radius * np.cos(theta))) * np.cos(phi)
    y_array = (major_radius + (minor_radius * np.cos(theta))) * np.sin(phi)    
    z_array = minor_radius * np.sin(theta)
    
    ax.plot_surface(x_array, y_array, z_array, color = 'yellow', alpha = 0.25, zorder = 0)
    
    return ax

def draw_source(config, config_vis, ax):
    #unpack variables
    origin   = config_vis['origin']
    normal   = config_vis['normal']
    corners  = config_vis['corners']
    
    #draw source origin dot, normal vector, and bounding box
    ax.scatter(origin[0,0], origin[0,1], origin[0,2], color = "yellow")
    ax.quiver(origin[0,0], origin[0,1], origin[0,2],
              normal[0,0]  , normal[0,1]  , normal[0,2]  ,
              color = "yellow", length = 0.1, arrow_length_ratio = 0.1)
    ax.plot3D(corners[0,:,0], corners[0,:,1], corners[0,:,2], color = "yellow")
    
    return ax

def draw_graphite(config, config_vis, graphite_mesh, ax):
    #unpack variables
    origin      = config_vis['origin']
    normal      = config_vis['normal']
    corners     = config_vis['corners']

    mesh_points = config['graphite_input']['mesh_points']
    mesh_faces  = config['graphite_input']['mesh_faces']
    
    if graphite_mesh is True:
        x_array = mesh_points[:,0]
        y_array = mesh_points[:,1]
        z_array = mesh_points[:,2]
        triangles = tri.Triangulation(x_array, y_array, mesh_faces)
        ax.plot_trisurf(triangles, z_array, color = 'grey', zorder = 10)
    else:
        #draw graphite origin dot, normal vector, and bounding box
        ax.scatter(origin[1,0], origin[1,1], origin[1,2], color = "grey", zorder = 10)
        ax.quiver(origin[1,0], origin[1,1], origin[1,2],
                  normal[1,0]  , normal[1,1]  , normal[1,2]  ,
                  color = "grey", length = 0.1, arrow_length_ratio = 0.1, zorder = 10)
        ax.plot3D(corners[1,:,0], corners[1,:,1], corners[1,:,2], color = "grey", zorder = 10)
    
    return ax

def draw_crystal(config, config_vis, crystal_mesh, crystal_guides, ax):
    #unpack variables
    scenario        = config['general_input']['scenario']
    mesh_points     = config['crystal_input']['mesh_points']
    mesh_faces      = config['crystal_input']['mesh_faces']
    
    origin          = config_vis['origin']
    normal          = config_vis['normal']
    orient_x        = config_vis['orient_x']
    orient_y        = config_vis['orient_y']
    corners         = config_vis['corners']
    
    circle_points   = np.linspace(0, np.pi * 2, 36)[:,np.newaxis]
    crystal_circle  = np.zeros([36,3], dtype = np.float64)
    rowland_circle  = np.zeros([36,3], dtype = np.float64)
    
    meridi_line     = np.zeros([2,3], dtype = np.float64)
    saggit_line     = np.zeros([2,3], dtype = np.float64)
    
    #crystal optical properties [Float64]
    crystal_bragg   = bragg_angle(config['source_input']['wavelength'],
                                  config['crystal_input']['crystal_spacing'])
    meridi_focus    = (config['crystal_input']['radius']
                        * np.sin(crystal_bragg))
    sagitt_focus    = - meridi_focus / np.cos(2 * crystal_bragg)
    
    ## The crystal's radius of curvature and Rowland circle
    #crystal_center[3D Coodrinates]
    crystal_center  =(config['crystal_input']['radius'] 
                    * config['crystal_input']['zaxis']
                    + config['crystal_input']['origin'])
    
    rowland_center  =(config['crystal_input']['radius'] / 2
                    * config['crystal_input']['zaxis']
                    + config['crystal_input']['origin'])
    
    #crystal_circle[Point Number, 3D Coordinates], 36 evenly-spaced points
    crystal_circle  = config['crystal_input']['radius'] * (
            (orient_y[2,:] * np.cos(circle_points)) + (normal[2,:] * np.sin(circle_points)))
    crystal_circle += crystal_center
    
    tangent_circle    = config['crystal_input']['radius'] * np.cos(crystal_bragg) * (
            (orient_y[2,:] * np.cos(circle_points)) + (normal[2,:] * np.sin(circle_points)))
    tangent_circle   += crystal_center
    
    rowland_circle  = config['crystal_input']['radius'] * 0.5 * (
            (orient_y[2,:] * np.cos(circle_points)) + (normal[2,:] * np.sin(circle_points)))
    rowland_circle += rowland_center
    
    ## The crystal's saggital and meridional foci
    inbound_vector   = np.zeros([3], dtype = np.float64)
    if scenario == "REAL" or scenario == "PLASMA" or scenario == "BEAM":
        inbound_vector = origin[1,:] - origin[2,:]
        inbound_vector/= np.linalg.norm(inbound_vector)
        
    if scenario == "THROUGHPUT" or scenario == "CRYSTAL":
        inbound_vector = origin[0,:] - origin[2,:]
        inbound_vector/= np.linalg.norm(inbound_vector)
        
    #meridi_line[Point Number, 3D Coordinates], 2 points (one above, one below)
    meridi_line[0,:] = origin[2,:] + meridi_focus * inbound_vector + 0.1 * orient_x[2,:]
    meridi_line[1,:] = origin[2,:] + meridi_focus * inbound_vector - 0.1 * orient_x[2,:]
    saggit_line[0,:] = origin[2,:] + sagitt_focus * inbound_vector + 0.1 *   normal[2,:]
    saggit_line[1,:] = origin[2,:] + sagitt_focus * inbound_vector - 0.1 *   normal[2,:]
    
    if crystal_mesh is True:
        x_array = mesh_points[:,0]
        y_array = mesh_points[:,1]
        z_array = mesh_points[:,2]
        triangles = tri.Triangulation(x_array, y_array, mesh_faces)
        ax.plot_trisurf(triangles, z_array, color = 'cyan', zorder = 10)
    else:
        #draw crystal origin dot, normal vector, center point, and bounding box
        ax.scatter(origin[2,0], origin[2,1], origin[2,2], color = "cyan", zorder = 10)
        ax.quiver(origin[2,0], origin[2,1], origin[2,2],
                  normal[2,0]  , normal[2,1]  , normal[2,2]  ,
                  color = "cyan", length = 0.1, arrow_length_ratio = 0.1, zorder = 10)
        ax.scatter(crystal_center[0], crystal_center[1], crystal_center[2], color = "blue")
        ax.plot3D(corners[2,:,0], corners[2,:,1], corners[2,:,2], color = "cyan", zorder = 10)
        
    if crystal_guides is True:
        #draw crystal foci and circles
        ax.plot3D(crystal_circle[:,0], crystal_circle[:,1], crystal_circle[:,2], color = "blue", zorder = 5)
        ax.plot3D(tangent_circle[:,0], tangent_circle[:,1], tangent_circle[:,2], color = "blue", zorder = 5)
        ax.plot3D(rowland_circle[:,0], rowland_circle[:,1], rowland_circle[:,2], color = "blue", zorder = 5)
        ax.plot3D(meridi_line[:,0], meridi_line[:,1], meridi_line[:,2], color = "blue", zorder = 5)
        ax.plot3D(saggit_line[:,0], saggit_line[:,1], saggit_line[:,2], color = "blue", zorder = 5)
    
    return ax

def draw_detector(config, config_vis, ax):
    #unpack variables
    origin  = config_vis['origin']
    normal  = config_vis['normal']
    corners = config_vis['corners']
    #draw detector origin dot, normal vector, and bounding box
    ax.scatter(origin[3,0], origin[3,1], origin[3,2], color = "red", zorder = 10)
    ax.quiver(origin[3,0], origin[3,1], origin[3,2],
              normal[3,0]  , normal[3,1]  , normal[3,2]  ,
              color = "red", length = 0.1 , arrow_length_ratio = 0.1, zorder = 10)    
    ax.plot3D(corners[3,:,0], corners[3,:,1], corners[3,:,2], color = "red", zorder = 10)
    
    
    return ax

def visualize_vectors(config, output, ii):
    ## Do all of the steps as before, but also add the output rays
    origin = output['lost'][ii]['origin']
    direct = output['lost'][ii]['direction']
    m      = output['lost'][ii]['mask']
    
    #to avoid plotting too many rays, randomly cull rays until there are 1000
    if len(m[m]) > 1000:
        cutter = np.random.randint(0, len(m[m]), len(m))
        m[m] &= (cutter[m] < 1000)
    
    plt, ax = visualize_layout(config)
    plt.title("X-Ray Raytracing Results")
    
    ax.quiver(origin[m,0], origin[m,1], origin[m,2],
              direct[m,0], direct[m,1], direct[m,2],
              length = 1.0, arrow_length_ratio = 0.01, 
              color = "green", alpha = 0.1, normalize = True)
    
    return plt, ax

def visualize_bundles(config, output):
    ## Do all of the steps as before, but also add the plasma bundles
    origin = output['lost'][0]['origin']
    m      = output['lost'][0]['mask']
    
    #to avoid plotting too many bundles, randomly cull rays until there are 1000
    if len(m[m]) > 1000:
        cutter = np.random.randint(0, len(m[m]), len(m))
        m[m] &= (cutter[m] < 1000)
    
    plt, ax = visualize_layout(config)
    plt.title("X-Ray Bundle Generation Results")

    ax.scatter(origin[m,0], origin[m,1], origin[m,2],
              color = "green", alpha = 0.1)
    
    return plt, ax
