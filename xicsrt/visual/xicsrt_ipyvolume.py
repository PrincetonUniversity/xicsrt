
import ipyvolume as ipv
import numpy as np

import matplotlib

from xicsrt import xicsrt_config
from xicsrt.xicsrt_dispatch import XicsrtDispatcher

def truncate_mask(mask, max_num):
    num_mask = np.sum(mask)
    if num_mask > max_num:
        w = np.where(mask)[0]
        np.random.shuffle(w)
        mask[w[max_num:]] = False

    return mask

def figure():
    fig = ipv.figure(width=900, height=900)
    return fig

def add_rays(results, config):
    _add_ray_history(results['found']['history'], config, lost=False)
    _add_ray_history(results['lost']['history'], config, lost=True)

def _add_ray_history(history, config, lost=None):

    if lost is False:
        color = [1.0, 0.0, 0.0, 0.1]
    elif lost is True:
        color = [0.0, 0.0, 1.0, 0.1]
    else:
        color = [0.0, 0.0, 0.0, 0.1]
        
    num_elem = len(history)
    key_list = list(history.keys())
    
    for ii in range(num_elem-1):

        # All rays leaving this optic element.
        mask = history[key_list[ii]]['mask']
        num_mask = np.sum(mask)

        if num_mask == 0:
            continue
        
        x0 = history[key_list[ii]]['origin'][mask, 0]
        y0 = history[key_list[ii]]['origin'][mask, 1]
        z0 = history[key_list[ii]]['origin'][mask, 2]
        x1 = history[key_list[ii+1]]['origin'][mask, 0]
        y1 = history[key_list[ii+1]]['origin'][mask, 1]
        z1 = history[key_list[ii+1]]['origin'][mask, 2]

        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.concatenate((z0, z1))

        lines = np.zeros((num_mask, 2), dtype=int)
        lines[:, 0] = np.arange(0, num_mask)
        lines[:, 1] = np.arange(num_mask, num_mask * 2)
        
        obj = ipv.plot_trisurf(x, y, z, lines=lines, color=color)
        obj.line_material.transparent = True
        obj.line_material.linewidth = 10.0
    

def generate_colors(history, config):
    """
    This is some old code to generate colors for the ray plotting
    in _add_ray_history.

    For the moment I am just saving the old code here, but this
    will not work with the new history dictionary. (Only minor 
    adjustments are neccessary to make it work, but the 
    implementation was not very general.)
    """
    raise NotImplementedError()


    color_masks = []
    color_list = []

    # This is a good pink color for un-reflected rays.
    # color_unref = [1.0, 0.5, 0.5, 0.2]

    if False:
        color_masks.append(output[0]['mask'].copy())
        color_list.append((1.0, 0.0, 0.0, 0.5))

    # Color by vertical position in Plasma.
    if False:
        norm = matplotlib.colors.Normalize(0.0, 1.0)
        cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='gist_rainbow')

        color_masks.append((output[0]['origin'][:, 2] > 0.1))
        color_list.append(cm.to_rgba(0.0, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > 0.05) & (output[0]['origin'][:, 2] < 0.1))
        color_list.append(cm.to_rgba(1 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > 0.0) & (output[0]['origin'][:, 2] < 0.05))
        color_list.append(cm.to_rgba(2 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > -0.05) & (output[0]['origin'][:, 2] < 0))
        color_list.append(cm.to_rgba(3 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] > -0.1) & (output[0]['origin'][:, 2] < -0.05))
        color_list.append(cm.to_rgba(4 / 5, alpha=0.5))

        color_masks.append((output[0]['origin'][:, 2] < -0.1))
        color_list.append(cm.to_rgba(5 / 5, alpha=0.5))

    if True:
        norm = matplotlib.colors.Normalize(0.0, 1.0)
        cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='gist_rainbow')

        delta = 0.005
        num_wave = 5
        for ii in range(num_wave):
            wave_0 = inputs['source_input']['wavelength'] - 1 * delta * (num_wave - 1) / 2 + delta * ii
            wave_min = wave_0 - delta / 2
            wave_max = wave_0 + delta / 2
            mask_temp = (output[0]['wavelength'][:] > wave_min) & (output[0]['wavelength'][:] < wave_max)
            print('{:7.4f} {:7.4f} {:7.4f} {}'.format(wave_0, wave_min, wave_max, np.sum(mask_temp)))
            color_masks.append(mask_temp)
            color_list.append(cm.to_rgba(ii / num_wave, alpha=0.5))

def add_surf(obj):
    w = obj.param['width'] / 2.0
    h = obj.param['height'] / 2.0
    d = obj.param['depth'] / 2.0

    points = np.zeros((8, 3))
    points[0, :] = [w, h, d]
    points[1, :] = [w, -h, d]
    points[2, :] = [-w, h, d]
    points[3, :] = [-w, -h, d]

    points[4, :] = [w, h, -d]
    points[5, :] = [w, -h, -d]
    points[6, :] = [-w, h, -d]
    points[7, :] = [-w, -h, -d]

    points_ext = obj.point_to_external(points)

    x = points_ext[:, 0]
    y = points_ext[:, 1]
    z = points_ext[:, 2]

    # I am sure there is a way to automate this using a meshgrid,
    # but at the moment this is faster.
    if d > 0:
        triangles = (
            (0, 3, 1)
            , (0, 3, 2)

            , (4, 7, 5)
            , (4, 7, 6)

            , (0, 6, 2)
            , (0, 6, 4)

            , (0, 5, 1)
            , (0, 5, 4)

            , (3, 6, 2)
            , (3, 6, 7)

            , (3, 5, 1)
            , (3, 5, 7)
        )
    else:
        triangles = (
            (0, 3, 1)
            , (0, 3, 2)
        )

    ipv_obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[1.0, 1.0, 0.0, 0.2])
    ipv_obj.material.transparent = True
    
def add_optics(config):

    for key_opt in config['optics']:
        config_opt = config['optics'][key_opt]

        if True:
            w = config_opt['width'] / 2.0
            h = config_opt['height'] / 2.0
            cx = config_opt['xaxis']
            cy = np.cross(config_opt['xaxis'], config_opt['zaxis'])

            point0 = w * cx + h * cy + config_opt['origin']
            point1 = h * cy + config_opt['origin']
            point2 = -1 * w * cx + h * cy + config_opt['origin']

            point3 = w * cx + config_opt['origin']
            point4 = config_opt['origin']
            point5 = -1 * w * cx + config_opt['origin']

            point6 = w * cx - h * cy + config_opt['origin']
            point7 = -1 * h * cy + config_opt['origin']
            point8 = -1 * w * cx - h * cy + config_opt['origin']

            points = np.array([
                point0
                , point1
                , point2
                , point3
                , point4
                , point5
                , point6
                , point7
                , point8
            ])

            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            # I am sure there is a way to automate this using a meshgrid,
            # but at the moment this is faster.
            triangles = (
                (4, 0, 2)
                , (4, 2, 8)
                , (4, 6, 8)
                , (4, 6, 0)
            )

            obj = ipv.plot_trisurf(x, y, z, triangles=triangles, color=[0.5, 0.5, 0.5, 0.5])
            obj.material.transparent = True

        if 'crystal' in str.lower(key_opt):
            # This code is incomplete.
            # Everything is plotted a constant z plain, instead of being tilted
            # with the crystal.
            #
            # What should be done here instead is to plot these circle in local
            # crystal coordinates, and then transform them into machine coordinates.
            
            crystal_center = config_opt['origin'] + config_opt['zaxis'] * config_opt['radius']
            x = np.array([crystal_center[0]])
            y = np.array([crystal_center[1]])
            z = np.array([crystal_center[2]])
            ipv.scatter(x, y, z, size=2, marker="sphere")

            # Plot the crystal circle.
            num = 1000
            crystal_radius = config_opt['radius']
            x = np.sin(np.linspace(0.0, np.pi * 2, num)) * config_opt['radius'] + crystal_center[0]
            y = np.cos(np.linspace(0.0, np.pi * 2, num)) * config_opt['radius'] + crystal_center[1]
            z = np.zeros(num) + crystal_center[2]
            lines = np.zeros((num, 2), dtype=int)
            lines[:, 0] = np.arange(num)
            lines[:, 1] = np.roll(lines[:, 0], 1)
            obj = ipv.plot_trisurf(x, y, z, lines=lines, color=[0.0, 0.0, 0.0, 0.5])

            rowland_center = config_opt['origin'] + config_opt['zaxis'] * config_opt['radius'] / 2
            rowland_radius = crystal_radius / 2
            x = np.sin(np.linspace(0.0, np.pi * 2, num)) * rowland_radius + rowland_center[0]
            y = np.cos(np.linspace(0.0, np.pi * 2, num)) * rowland_radius + rowland_center[1]
            z = np.zeros(num) + crystal_center[2]
            lines = np.zeros((num, 2), dtype=int)
            lines[:, 0] = np.arange(num)
            lines[:, 1] = np.roll(lines[:, 0], 1)
            obj = ipv.plot_trisurf(x, y, z, lines=lines, color=[0.0, 0.0, 0.0, 0.5])
            
def add_sources(config):
    # Update the default config with the user config.
    config = xicsrt_config.get_config(config)
    
    # Combine the user and default object pathlists.
    pathlist = []
    pathlist.extend(config['general']['pathlist_objects'])
    pathlist.extend(config['general']['pathlist_default'])
    
    sources = XicsrtDispatcher(config['sources'], pathlist)
    sources.instantiate_objects()
    sources.initialize()

    for key in config['sources']:
        obj = sources.get_object(key)
        
    add_surf(obj)

def show():
    view = [0,0,0]
    ipv.xlim(view[1] - 0.5, view[1] + 0.5)
    ipv.ylim(view[1] - 0.5, view[1] + 0.5)
    ipv.zlim(view[2] - 0.5, view[2] + 0.5)

    #ipv.style.axes_off()
    #ipv.style.box_off()

    ipv.show()
