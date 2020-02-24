#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:18:02 2020

@author: Eugene

Read the .TIF files located in ./xicsrt/results and perform various operations.
Run this code once to initialize, then type the necessary functions.
"""
#Imports
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

#Inputs
results_path = '../results/'

#Read the results directory and list all .TIF images
results_list = os.listdir(results_path)
tif_list = list()

for file in range(len(results_list)):
    if results_list[file].endswith('.tif'):
        tif_list.append(results_list[file])
        
def colorize_png(tif_list):
    #Convert the .TIF images into colormapped .PNG images and save them
    for file in range(len(tif_list)):
        image_array = np.array(Image.open(results_path + tif_list[file]))
        image = plt.imshow(image_array, interpolation = None)
        image.set_cmap('nipy_spectral')
        image.axes.get_xaxis().set_visible(False)
        image.axes.get_yaxis().set_visible(False)
        
        png_name = results_path + tif_list[file][:-4] + '.png'
        plt.savefig(png_name, bbox_inches = 'tight')
        print(png_name + ' saved!')
        
def histogram_analysis(tif_list):
    #Read the .TIF images and plot two histograms and a spectrogram
    for file in range(len(tif_list)):
        image_array = np.array(Image.open(results_path + tif_list[file]))
        fig, ax = plt.subplots(nrows = 2, ncols = 2)
        
        #Plot vertical histograms
        v_x = np.linspace(0, image_array.shape[1], num = image_array.shape[1])
        v_y = np.sum(image_array, axis = 0, dtype = int)
        ax[0,0].bar(v_x, v_y, width = 1.0, color = 'gray')
        
        #Plot horizontal histograms
        h_x = np.linspace(image_array.shape[0], 0, num = image_array.shape[0])
        h_y = np.sum(image_array, axis = 1, dtype = int)
        ax[1,1].barh(h_x, h_y, height = 1.0, color = 'gray')
        
        #Plot numpy arrays as images with logarithmic grayscale colormap
        ax[1,0].imshow(image_array, cmap = 'gray')
        ax[1,0].axis('off')
        
        #Plot Spectrogram
        image_max = int(np.amax(image_array))
        ax[0,1].hist(image_array.ravel(), bins = image_max, 
                     range = (0, image_max), density = True, color = 'gray')
        
        #Save Images
        hist_name = results_path + tif_list[file][:-4] + '_histogram.png'
        fig.savefig(hist_name, bbox_inches = 'tight')
        print(hist_name + ' saved!')
        
def tif_overlay(tif_list):
    #Overlay the .TIF images to create composite .TIF images
    g_list = list()
    c_list = list()
    d_list = list()
    
    for file in range(len(tif_list)): 
        if tif_list[file].startswith('xicsrt_graphite_'):
            g_list.append(tif_list[file])
        if tif_list[file].startswith('xicsrt_crystal_'):
            c_list.append(tif_list[file])
        if tif_list[file].startswith('xicsrt_detector_'):
            d_list.append(tif_list[file])
    
    g_array = None
    c_array = None
    d_array = None
    
    for file in range(len(g_list)):
        image_array = np.array(Image.open(results_path + g_list[file]))
        if g_array is None:
            g_array  = image_array
        else:
            g_array += image_array
            
    for file in range(len(c_list)):
        image_array = np.array(Image.open(results_path + c_list[file]))
        if c_array is None:
            c_array  = image_array
        else:
            c_array += image_array
            
    for file in range(len(d_list)):
        image_array = np.array(Image.open(results_path + d_list[file]))
        if d_array is None:
            d_array  = image_array
        else:
            d_array += image_array
            
    g_image = Image.fromarray(g_array)
    c_image = Image.fromarray(c_array)
    d_image = Image.fromarray(d_array)
    
    g_image.save(results_path + 'xicsrt_graphite_total.tif')
    print('xicsrt_graphite_total.tif saved!')
    c_image.save(results_path + 'xicsrt_crystal_total.tif')
    print('xicsrt_crystal_total.tif saved!')
    d_image.save(results_path + 'xicsrt_detector_total.tif')
    print('xicsrt_detector_total.tif saved!')
    
def tif_subtract(tif_1, tif_2):
    #Open two .TIF images, subtract them, and output the difference
    image_array_1 = np.array(Image.open(results_path + tif_1))
    image_array_2 = np.array(Image.open(results_path + tif_2))
    image_array_s = np.abs(image_array_1 - image_array_2)
    
    image = plt.imshow(image_array_s, interpolation = None)
    image.set_cmap('nipy_spectral')
    image.axes.get_xaxis().set_visible(False)
    image.axes.get_yaxis().set_visible(False)
    
    png_name = results_path + 'tif_difference' + '.png'
    plt.savefig(png_name, bbox_inches = 'tight')
    print(png_name + ' saved!')

"""
colorize_png(tif_list)
histogram_analysis(tif_list)
tif_overlay(tif_list)
"""