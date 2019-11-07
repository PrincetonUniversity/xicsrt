#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:29:13 2019

Authors
-------
  - Yevgeniy Yakusevich <eugenethree@gmail.com>

Description
-----------
A standalone script that reads the xicsrt_input.json file and offsets + tilts
the individual optical elements. Useful for misalignment analysis, but not 
vital to the core ray tracer functionality.
"""
import json
import numpy as np
from xicsrt.xics_rt_math import vector_rotate

## Manual input
g_offset = np.array([0,0,0], dtype = np.float64)
g_tilt   = np.array([0,0,0], dtype = np.float64)
c_offset = np.array([0,0,0], dtype = np.float64)
c_tilt   = np.array([0,0,0], dtype = np.float64)
d_offset = np.array([0,0,0], dtype = np.float64)
d_tilt   = np.array([0,0,0], dtype = np.float64)

config_num = 0

## Read the xicsrt_input.json file
try:
    with open('xicsrt_input.json', 'r') as input_file:
        xicsrt_input = json.load(input_file)
except FileNotFoundError:
    print('Input file not found.')
    exit()

#convert all lists into numpy arrays
for element in xicsrt_input[config_num]:
    for key in xicsrt_input[config_num][element]:
        if type(xicsrt_input[config_num][element][key]) is list:
            xicsrt_input[config_num][element][key] = np.array(
                    xicsrt_input[config_num][element][key])

## Create coordinate bases for each optical element
g_x_vector  = xicsrt_input[config_num]['graphite_input']['orientation']
g_z_vector  = xicsrt_input[config_num]['graphite_input']['normal']
g_y_vector  = np.cross(g_z_vector, g_x_vector)
g_y_vector /= np.linalg.norm(g_y_vector)   

c_x_vector  = xicsrt_input[config_num]['crystal_input']['orientation']
c_z_vector  = xicsrt_input[config_num]['crystal_input']['normal']
c_y_vector  = np.cross(c_z_vector, c_x_vector)
c_y_vector /= np.linalg.norm(c_y_vector)

d_x_vector  = xicsrt_input[config_num]['detector_input']['orientation']
d_z_vector  = xicsrt_input[config_num]['detector_input']['normal']
d_y_vector  = np.cross(d_z_vector, d_x_vector)
d_y_vector /= np.linalg.norm(d_y_vector)

g_basis     = np.transpose(np.array([g_x_vector, g_y_vector, g_z_vector]))
c_basis     = np.transpose(np.array([c_x_vector, c_y_vector, c_z_vector]))
d_basis     = np.transpose(np.array([d_x_vector, d_y_vector, d_z_vector]))

## Perform offset math
g_position  = xicsrt_input[config_num]['graphite_input']['position']
c_position  = xicsrt_input[config_num]['crystal_input']['position']
d_position  = xicsrt_input[config_num]['detector_input']['position']

g_position += g_basis.dot(np.transpose(g_offset))
c_position += c_basis.dot(np.transpose(c_offset))
d_position += d_basis.dot(np.transpose(d_offset))

## Perform tilt math
#rotate around the X axis
g_x_new     = vector_rotate(g_x_vector, g_x_vector, g_tilt[0])
g_y_new     = vector_rotate(g_y_vector, g_x_vector, g_tilt[0])
g_z_new     = vector_rotate(g_z_vector, g_x_vector, g_tilt[0])

c_x_new     = vector_rotate(c_x_vector, c_x_vector, c_tilt[0])
c_y_new     = vector_rotate(c_y_vector, c_x_vector, c_tilt[0])
c_z_new     = vector_rotate(c_z_vector, c_x_vector, c_tilt[0])

d_x_new     = vector_rotate(d_x_vector, d_x_vector, d_tilt[0])
d_y_new     = vector_rotate(d_y_vector, d_x_vector, d_tilt[0])
d_z_new     = vector_rotate(d_z_vector, d_x_vector, d_tilt[0])

#rotate around the Y axis
g_x_new     = vector_rotate(g_x_vector, g_y_vector, g_tilt[1])
g_y_new     = vector_rotate(g_y_vector, g_y_vector, g_tilt[1])
g_z_new     = vector_rotate(g_z_vector, g_y_vector, g_tilt[1])

c_x_new     = vector_rotate(c_x_vector, c_y_vector, c_tilt[1])
c_y_new     = vector_rotate(c_y_vector, c_y_vector, c_tilt[1])
c_z_new     = vector_rotate(c_z_vector, c_y_vector, c_tilt[1])

d_x_new     = vector_rotate(d_x_vector, d_y_vector, d_tilt[1])
d_y_new     = vector_rotate(d_y_vector, d_y_vector, d_tilt[1])
d_z_new     = vector_rotate(d_z_vector, d_y_vector, d_tilt[1])

#rotate around the Z axis
g_x_new     = vector_rotate(g_x_vector, g_z_vector, g_tilt[2])
g_y_new     = vector_rotate(g_y_vector, g_z_vector, g_tilt[2])
g_z_new     = vector_rotate(g_z_vector, g_z_vector, g_tilt[2])

c_x_new     = vector_rotate(c_x_vector, c_z_vector, c_tilt[2])
c_y_new     = vector_rotate(c_y_vector, c_z_vector, c_tilt[2])
c_z_new     = vector_rotate(c_z_vector, c_z_vector, c_tilt[2])

d_x_new     = vector_rotate(d_x_vector, d_z_vector, d_tilt[2])
d_y_new     = vector_rotate(d_y_vector, d_z_vector, d_tilt[2])
d_z_new     = vector_rotate(d_z_vector, d_z_vector, d_tilt[2])

## Write the misaligned coordinates to the input file
#re-pack variables
xicsrt_input[config_num]['graphite_input']['position']      = g_position
xicsrt_input[config_num]['crystal_input']['position']       = c_position
xicsrt_input[config_num]['detector_input']['position']      = d_position

xicsrt_input[config_num]['graphite_input']['orientation']   = g_x_new
xicsrt_input[config_num]['graphite_input']['normal']        = g_z_new
xicsrt_input[config_num]['crystal_input']['orientation']    = c_x_new
xicsrt_input[config_num]['crystal_input']['normal']         = c_z_new
xicsrt_input[config_num]['detector_input']['orientation']   = d_x_new
xicsrt_input[config_num]['detector_input']['normal']        = d_z_new

# Convert all numpy arrays into json-recognizable lists
for element in xicsrt_input[config_num]:
    for key in xicsrt_input[config_num][element]:
        if type(xicsrt_input[config_num][element][key]) is np.ndarray:
            xicsrt_input[config_num][element][key] = (
                    xicsrt_input[config_num][element][key].tolist())

#save the file
with open('xicsrt_input.json', 'w') as input_file:
        json.dump(xicsrt_input ,input_file, indent = 1, sort_keys = True)
print('xicsrt_input.json saved!')