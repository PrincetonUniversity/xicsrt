# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    James Kring <jdk0026@tigermail.auburn.edu>

A set of tools for 2d visualization of the XICSRT results
"""

import numpy as np
import logging

from xicsrt.util import mirplot
from xicsrt import xicsrt_public
from xicsrt.tools import xicsrt_aperture

def plot_intersect(
        results,
        name=None,
        section=None,

        alpha=None,

        lost=None,
        lost_color=None,
        lost_alpha=None,

        found=None,
        found_color=None,
        found_alpha=None,

        xbound=None,
        ybound=None,
        scale=None,
        aspect=None,
        plot_bounds=None,
        plot_aperture=None,

        plotlist=None,
        plot_to_screen=True,
        ):
    """
    Plot the intersection of rays with the given optic.

    Parameters
    ----------
    results
      The restults dictionary from `raytrace()` that include the ray history.

    Keywords
    --------
    name : string (None)
      The name of the optic or source for which to plot intersections. The name
      refers to the key of the entry in the config dictionary. For example
      the name 'detector' will refer to config['optics']['detector'].

    section : string (None)
      [Optional] The name of the config section in which to search for `name`.
      This should typically be either 'optics' or 'sources'. If no section is
      given then then 'optics' will be searched first, then 'sources'.

    Returns
    -------
    plotlist
      Will return a mirplot.PlotList with the full plot definition.
    """
    if name is None: name = 'detector'
    if found is None: found = True
    if lost is None: lost = True
    if aspect is None: aspect = 1
    if lost_color is None: lost_color = 'royalblue'
    if found_color is None: found_color = 'red'
    if plot_bounds is None: plot_bounds = True
    if plot_aperture is None: plot_aperture = True

    if (alpha is None) and (lost_alpha is None):
        lost_alpha = 0.1
    elif (lost_alpha is None):
        lost_alpha = alpha

    if (alpha is None) and (found_alpha is None):
        found_alpha = 0.5
    elif (found_alpha is None):
        found_alpha = alpha

    if scale is None: scale = 1.0

    # Create a plot list.
    if plotlist is None:
        plotlist = mirplot.PlotList()

    config = results['config']

    # Get the crystal object from the dispatcher.
    obj = xicsrt_public.get_element(config, name, section)

    if plot_bounds:
        plotlist.extend(_get_bounds_plotlist(obj, scale))
    if plot_aperture:
        plotlist.extend(_get_aperture_plotlist(obj, scale))

    if xbound is None:
        xbound = np.array([-1*obj.param['xsize']/2, obj.param['xsize']/2])*1.2
    if ybound is None:
        ybound = np.array([-1*obj.param['ysize']/2, obj.param['ysize']/2])*1.2

    if lost:
        # Lets plot the 'lost' rays.
        origin_ext = results['lost']['history'][name]['origin']
        origin_loc = obj.point_to_local(origin_ext)

        # We only want to plot the rays that intersected this optic.
        # This mask will likely need to be changed in the future once I
        # improve the handling of lost rays.
        mask = np.all(origin_ext[:, :] != np.nan, axis=1)
        if np.sum(mask) > 0:
            plotlist.append({
                'name': '0'
                , 'type': 'scatter'
                , 'x': origin_loc[mask, 0]*scale
                , 'y': origin_loc[mask, 1]*scale
                , 'xbound': xbound*scale
                , 'ybound': ybound*scale
                , 'aspect': aspect
                , 'alpha': lost_alpha
                , 'color': lost_color
            })

    if found:
        # Lets plot the 'found' rays.
        origin_ext = results['found']['history'][name]['origin']
        origin_loc = obj.point_to_local(origin_ext)
        mask = results['found']['history'][name]['mask']
        if np.sum(mask) > 0:
            plotlist.append({
                'name': '0'
                , 'type': 'scatter'
                , 'x': origin_loc[mask, 0]*scale
                , 'y': origin_loc[mask, 1]*scale
                , 'xbound': xbound*scale
                , 'ybound': ybound*scale
                , 'aspect': aspect
                , 'alpha': found_alpha
                , 'color': found_color
            })

    if plot_to_screen:
        p = plotlist.plotToScreen()

    return plotlist

def _get_bounds_plotlist(obj, scale=None):
    if scale is None: scale=1.0
    plotlist = mirplot.PlotList()

    # Plot the optic extent as taken from the xsize and ysize.
    opt_x = obj.param['xsize']/2*scale
    opt_y = obj.param['ysize']/2*scale
    plotlist.append({
        'name': '0',
        'x': [-1*opt_x, opt_x, opt_x, -1*opt_x, -1*opt_x],
        'y': [opt_y, opt_y, -1*opt_y, -1*opt_y, opt_y],
        'linestyle': '--',
        'color': 'gray',
    })

    return plotlist

def _get_aperture_plotlist(obj, scale=None):
    if scale is None: scale=1.0
    plotlist = mirplot.PlotList()

    if (not 'aperture' in obj.param) or (obj.param['aperture'] is None):
        return []

    for apt in obj.param['aperture']:
        apt = xicsrt_aperture._aperture_defaults(apt)
        shape = apt['shape']
        if shape == 'square':
            size = apt['size'][0]/2*scale
            origin = apt['origin'].copy()*scale
            x = np.array([-1, 1,  1, -1, -1])*size + origin[0]
            y = np.array([ 1, 1, -1, -1,  1])*size + origin[1]

        if shape == 'rectangle':
            size = apt['size'].copy()/2*scale
            origin = apt['origin'].copy()*scale
            x = np.array([-1, 1,  1, -1, -1])*size[0] + origin[0]
            y = np.array([ 1, 1, -1, -1,  1])*size[1] + origin[1]

        if shape == 'triangle':
            vert = apt['vertices'].copy()*scale
            origin = apt['origin'].copy()*scale
            x = np.array([vert[0,0], vert[1,0], vert[2,0], vert[0,0]])+origin[0]
            y = np.array([vert[0,1], vert[1,1], vert[2,1], vert[0,1]])+origin[0]
        else:
            logging.warning(f'Plotting of {shape} aperture not yet implemented.')
            return []

        plotlist.append({
            'name': '0',
            'x': x,
            'y': y,
            'linestyle': '-',
            'color': 'black',
        })
    return plotlist