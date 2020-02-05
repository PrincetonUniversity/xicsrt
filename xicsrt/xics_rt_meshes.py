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
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

#constants for unit conversion
conv1 = 12.398425
conv2 = np.pi/180

#input mesh horizontal and vertical resolutions here
m = 100
n = 10

#input base radius and base height here (meters)
r_0 = 0.300
h_0 = 0.01

"""
multi-toroidal mirror designed to reflect x-rays with energies 9.750 - 10.560 keV
intended for Ge[400] crystal mirror with inter-atomic distance 2d = 2.82868 A
"""

## Solve the Sinusoidal Spiral
energy = np.linspace(9.750, 10.560, m)
#energy      = np.linspace(9.0, 100.0, m)
wavelength  = conv1 / energy
spacing     = 2.82868
theta_rad   = np.arcsin(wavelength / spacing)
theta_0_rad = theta_rad[0]

#b is the sinusoidal spiral constant; it shouldn't equal 1
b = 2.0
a = b - 1.0
c = 1.0 / a

#solve for the sinusoidal spiral in polar coordinates
phi = (theta_rad - theta_0_rad) / a
alpha_rad = theta_rad + phi
r = r_0 * np.power((np.sin(theta_rad) / np.sin(theta_0_rad)), c)

#convert polar to cartesian coordinates
spiral      = np.zeros([m,3])
spiral[:,0] = r * np.cos(phi)
spiral[:,1] = r * np.sin(phi)
spiral[:,2] = 0.0

## Solve the Lens Metadata
#calculate crystal length by adding up the lengths of individual segments
length = 0
for j in range(m - 1):
    length += r[j] * (phi[j+1] - phi[j] / np.sin(theta_rad[j]))
    
#calculate crystal radius of curvature rho
rho = r / ((a + 1) * np.sin(theta_rad))

#calculate coordinates of curvature centers
center      = np.zeros([m,3])
center[:,0] = spiral[:,0] - rho * np.sin(theta_rad)
center[:,1] = spiral[:,1] + rho * np.cos(theta_rad)
center[:,2] = 0.0

#create vectors from the curvature centers to the lens
normal = spiral - center

#use the normal vectors to calculate the mesh geometry
height    = np.zeros([m,n], dtype = np.float64)
height[:] = np.linspace(-h_0/2, h_0/2, n)
theta_rad = np.repeat(theta_rad[:,np.newaxis], n, axis = 1)
rho       = np.repeat(rho[:,np.newaxis], n, axis = 1)
beta_rad  = np.arcsin(height / rho)

#calculate mesh points
mesh_points = np.zeros([m,n,3], dtype = np.float64)
z_vector    = np.array([0.0,0.0,1.0])
for ii in range(m):
    for jj in range(n):
        mesh_points[ii,jj,:] = (normal[ii,:] * np.cos(beta_rad[ii,jj]) +
                                z_vector[:]  * np.sin(beta_rad[ii,jj]))
mesh_points = mesh_points.reshape((m*n),3)
mesh_faces  = Delaunay(mesh_points).simplices
mesh_faces  = mesh_faces[:,1:4]
#visualize the mesh
plt.plot(spiral[:,0], spiral[:,1])
plt.plot(center[:,0], center[:,1])

fig = plt.figure()
ax  = fig.gca(projection='3d')
ax.scatter(mesh_points[:,0], mesh_points[:,1], mesh_points[:,2], color = "cyan")
#triangles = tri.Triangulation(mesh_points[:,0], mesh_points[:,1], mesh_faces)
#ax.plot_trisurf(triangles, mesh_points[:,2], color = 'cyan', zorder = 0)
fig.show()
