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
        aspect=None,
        alpha_lost=None,
        alpha_found=None,
        xbound=None,
        ybound=None,
        ):
    """
    Plot the intersection of rays with the given optic.
    """
    if section is None: section = 'optics'
    if name is None: name = 'detector'
    if aspect is None: aspect = 1
    if alpha_lost is None: alpha_lost = 0.1
    if alpha_found is None: alpha_found = 0.5

    # Create a plot list.
    plotlist = mirplot.PlotList()

    config = results['config']

    print(name, section)
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

    if True:
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
                , 'x': origin_loc[mask, 0]
                , 'y': origin_loc[mask, 1]
                , 'xbound': xbound
                , 'ybound': ybound
                , 'aspect': aspect
                , 'alpha': alpha_lost
                , 'color': 'royalblue'
            })

    if True:
        # Lets plot the 'found' rays.
        origin_ext = results['found']['history'][name]['origin']
        origin_loc = obj.point_to_local(origin_ext)
        mask = results['found']['history'][name]['mask']
        if np.sum(mask) > 0:
            plotlist.append({
                'name': '0'
                , 'type': 'scatter'
                , 'x': origin_loc[mask, 0]
                , 'y': origin_loc[mask, 1]
                , 'xbound': xbound
                , 'ybound': ybound
                , 'aspect': aspect
                , 'alpha': alpha_found
                , 'color': 'red'
            })

    if True:
        # Plot the optic extent as taken from the xsize and ysize.
        # In the future we should also plot the aperture (if defined).
        opt_x = obj.param['xsize']/2
        opt_y = obj.param['ysize']/2
        plotlist.append({
            'name': '0',
            'x': [-1*opt_x, opt_x, opt_x, -1*opt_x, -1*opt_x],
            'y': [opt_y, opt_y, -1*opt_y, -1*opt_y, opt_y],
            'linestyle': '--',
            'color': 'black',
        })

    p = plotlist.plotToScreen()
