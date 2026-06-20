import numpy as np
import xicsrt
import xicsrt.visual.xicsrt_3d__plotly as xicsrt_3d
from matplotlib import pyplot
import xicsrt.visual.xicsrt_2d__matplotlib as xicsrt_2d
intensity = 1e5
L_incident = 0.33
wavelength = 1.542
L_slit_receive_1 = 0.18
L_receive = 0.33
omega = 13.25
phi = 13.25

def Configure_Diffracted(omega, two_theta, L_incident,L_receive):

    phi = two_theta-omega
    config = dict()

    config['general'] = {}
    config['general']['number_of_iter'] = 5
    config['general']['save_images'] = False

    config['sources'] = {}
    config['sources']['source'] = {}
    config['sources']['source']['class_name'] = 'XicsrtSourceDirected'
    config['sources']['source']['intensity'] = intensity
    config['sources']['source']['wavelength'] = wavelength
    config['sources']['source']['angular_dist'] = 'flat_xy'
    config['sources']['source']['spread'] = [np.radians(5.0),np.radians(0.01)]
    config['sources']['source']['xsize'] = 0.012
    config['sources']['source']['ysize'] = 0.001
    config['sources']['source']['zsize'] = 0.0
    config['sources']['source']['zaxis'] = [0.0, np.cos(np.radians(omega)), -np.sin(np.radians(omega))]
    config['sources']['source']['origin'] = [0.0, -L_incident*np.cos(np.radians(omega)), L_incident*np.sin(np.radians(omega))]
    config['sources']['source']['direction'] = [0.0, np.cos(np.radians(omega)), -np.sin(np.radians(omega))]

    config['optics'] = {}
    config['optics']['aperture'] = {}
    config['optics']['aperture']['class_name'] = 'XicsrtOpticAperture'
    config['optics']['aperture']['origin'] = [0.0, -L_slit_receive_1*np.cos(np.radians(omega)), L_slit_receive_1*np.sin(np.radians(omega))]
    config['optics']['aperture']['zaxis'] = [0.0, np.cos(np.radians(omega)), -np.sin(np.radians(omega))]
    config['optics']['aperture']['aperture']=[
        {'shape':'rectangle', 'size':[0.002, 0.01], 'origin':[0.0, 0.0],  'logic':'and'},
        ]
    
    config['optics']['crystal'] = {}
    config['optics']['crystal']['class_name'] = 'XicsrtOpticPlanarMosaicCrystal'
    config['optics']['crystal']['mosaic_spread'] = np.radians(0.81)
    config['optics']['crystal']['mosaic_depth'] = 15
    config['optics']['crystal']['crystal_spacing'] = 3.357
    config['optics']['crystal']['reflectivity'] = 0.25
    config['optics']['crystal']['mosaic_absorption'] = 0.05
    config['optics']['crystal']['rocking_fwhm'] = np.radians(0.01)
    config['optics']['crystal']['xsize'] = 0.01
    config['optics']['crystal']['ysize'] = 0.01
    config['optics']['crystal']['origin'] = [0.0, 0.0, 0.0]
    config['optics']['crystal']['zaxis'] = [0.0, 0.0, 1.0]

    

    config['optics']['detector'] = {}
    config['optics']['detector']['class_name'] = 'XicsrtOpticDetector'
    config['optics']['detector']['origin'] = [0.0, L_receive*np.cos(np.radians(phi)), L_receive*np.sin(np.radians(phi))]
    config['optics']['detector']['zaxis']  = [0.0, -np.cos(np.radians(phi)), -np.sin(np.radians(phi))]
    config['optics']['detector']['xsize']  = 0.02274
    config['optics']['detector']['ysize']  = 0.02274

    return config

if __name__ == "__main__":
    two_theta = omega+phi
    config = Configure_Diffracted(omega,two_theta,L_incident,L_receive)
    result = xicsrt.raytrace(config)
    fig = xicsrt_3d.figure()
    xicsrt_3d.add_rays(result)
    xicsrt_3d.add_optics(result['config'])
    xicsrt_3d.add_sources(result['config'])
    xicsrt_3d.show()
    fig = xicsrt_2d.plot_intersect(result, 'detector',units = 'm')
    pyplot.show()