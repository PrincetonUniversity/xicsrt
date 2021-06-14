# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    James Kring <jdk0026@tigermail.auburn.edu>

A set of tools for 2d visualization of the XICSRT results
"""

import numpy as np
from xicsrt.util import mirplot
from xicsrt.objects._Dispatcher import Dispatcher

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
        optic_bounds=None,

        plotlist=None,
        plot_to_screen=True,
        ):
    """
    Plot the intersection of rays with the given optic.
    """
    if section is None: section = 'optics'
    if name is None: name = 'detector'
    if found is None: found = True
    if lost is None: lost = True
    if aspect is None: aspect = 1
    if lost_color is None: lost_color = 'royalblue'
    if found_color is None: found_color = 'red'
    if optic_bounds is None: optic_bounds = True

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

    # Use the dispatcher in instantiate and initialize objects.
    optics = Dispatcher(config, section)
    optics.instantiate([name])
    optics.setup()
    optics.initialize()

    # Get the crystal object from the dispatcher.
    obj = optics.get_object(name)

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

    if optic_bounds:
        # Plot the optic extent as taken from the xsize and ysize.
        # In the future we should also plot the aperture (if defined).
        opt_x = obj.param['xsize']/2*scale
        opt_y = obj.param['ysize']/2*scale
        plotlist.append({
            'name': '0',
            'x': [-1*opt_x, opt_x, opt_x, -1*opt_x, -1*opt_x],
            'y': [opt_y, opt_y, -1*opt_y, -1*opt_y, opt_y],
            'linestyle': '--',
            'color': 'black',
        })

    if plot_to_screen:
        p = plotlist.plotToScreen()

    return plotlist