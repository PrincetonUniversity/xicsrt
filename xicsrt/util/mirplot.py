# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir pablant <npablant@pppl.gov>

An interface to matplotlib that allows specification of complex plots
though a list of parameter dictionaries.

Example
-------

The simplest example:

.. code::

  import numpy as np
  import mirplot

  x = np.arange(10)
  y = x
  plotlist = [{'x':x, 'y':y}]
  fig = mirplot.plot_to_screen(plotlist)

Any supported plot properties can be added to the plot dictionary:

.. code:

    plotlist = [{
        'x':x,
        'y':y,
        'xbound':[0,1],
        'ybound':[0,1],
        'xtitle':'This is the x-axis',
        'ytitle':'This is the y-axis',
        }]
    fig = mirplot.plot_to_screen(plotlist)

To add multiple plots to a single figure add parameter dicts ta the plotlist:

.. code:

    plotlist = [
        {'x':x1, 'y':y1},
        {'x':x2, 'y':y2},
        ]
    fig = mirplot.plot_to_screen(plotlist)

If axes names are provided then plots will be added to separate subfigures
(stacked vertically). Each unique axes name will result in a new subfigure.

.. code:

    plotlist = [
        {'axes':'plot 1', 'x':x1, 'y':y1},
        {'axes':'plot 2', 'x':x2, 'y':y2},
        ]
    fig = mirplot.plot_to_screen(plotlist)

mirplot can also be used with predifined axes. For this purpose the axes must
be placed into a dictionary and passed to `plot_to_axes`.

.. code:

    fig, axs = plt.subplots(1, 2)

    axesdict = {
        'plot 1':axs[0],
        'plot 2':axs[1],
        }
    plotlist = [
        {'axes':'plot 1', 'x':x1, 'y':y1},
        {'axes':'plot 2', 'x':x2, 'y':y2},
        ]

    fig = mirplot2.plot_to_axes(plotlist, axesdict)

mirplot properties
------------------

A set of unique plot and axes properties are defined by mirplot to enable
a complete dictionary definition.

type : str ('line')
  Allowed Values: line, errorbar, scatter, fill_between, hline, vline, hspan,
  vspan.

legend : bool (false)
  Set to true to show the legend in this subplot.

matplotlib properties
---------------------

Any matplotlib plot or axes property that can be set using a simple
`set_prop(value)` method is supported. Certain properties requiring
a more complex set call are also supported.

"""

import logging
import copy

import matplotlib
import numpy as np

m_log = logging.getLogger(__name__.split('.')[-1])
m_log.setLevel(logging.INFO)

__version__ = '2.0.0'

def plot_to_screen(plotlist, show=True):
    matplotlib.pyplot.ioff()

    namelist = _autoname_plots(plotlist)
    fig = _make_figure(namelist)
    axesdict = _make_axes(namelist, fig)
    plot_to_axes(plotlist, axesdict)

    matplotlib.pyplot.ion()
    if show:
        fig.show()

    return fig


def plot_to_file(plotlist, filename):

    fig = plot_to_screen(plotlist, show=False)
    fig.savefig(filename)
    m_log.info('Saved figure to file: {}'.format(filename))


def plot_to_axes(plotlist, axesdict):

    # The order of these calls is important so we need to use multiple loops

    axeslist = []
    for plot in plotlist:
        if plot.get('type') == 'figure':
            axes = list(axesdict.values())[0]
        else:
            axes = plot['axes']

        if isinstance(axes, str):
            if axes in axesdict:
                axes = axesdict[axes]
            else:
                raise Exception(f'Named axes {axes} not found.')
        axeslist.append(axes)

    for plot in plotlist:
        _set_plot_defaults(plot)
        _clean_plot_prop(plot)

    for ii, plot in enumerate(plotlist):
        _apply_plot_prop(plot, axeslist[ii])

    for ii, plot in enumerate(plotlist):
        _apply_axes_prop(plot, axeslist[ii])

    for ii, plot in enumerate(plotlist):
        _apply_fig_prop(plot, axeslist[ii])


def _set_plot_defaults(prop):
    prop.setdefault('type', 'line')

    if prop.get('type') == 'figure':
        return
    if prop.get('type') == 'axes':
        return

    prop.setdefault('x')
    prop.setdefault('y')
    prop.setdefault('xerr')
    prop.setdefault('yerr')
    prop.setdefault('s', 15)
    prop.setdefault('legend_fontsize', 12.0)
    prop.setdefault('legend_framealpha', 0.7)

    if prop['type'] == 'image':
        if (prop['x'] is not None) and (prop['y'] is not None):
            prop['extent'] = [min(prop['x']), max(prop['x']), min(prop['y']), max(prop['y'])]
    elif prop['type'] == 'hline':
        if prop['y'] is None:
            prop['y'] = [0]
    elif prop['type'] == 'vline':
        if prop['x'] is None:
            prop['x'] = [0]
    else:
        if prop['x'] is None:
            prop['x'] = np.arange(len(prop['y']))
        #if not 'ybound' in prop:
        #    yrange = np.array([np.nanmin(prop['y']), np.nanmax(prop['y'])])
        #    prop['ybound'] = yrange + np.array([-0.1, 0.1]) * (yrange[1] - yrange[0])

def _clean_plot_prop(prop):
    """
    Check the plot properties and cleanup or provides errors.
    """
    if prop.get('type') == 'figure':
        return
    if prop.get('type') == 'axes':
        return

    if 'x' in prop and prop['x'] is not None:
        if np.isscalar(prop['x']):
            prop['x'] = np.asarray([prop['x']])
        else:
            prop['x'] = np.asarray(prop['x'])

    if 'y' in prop and prop['y'] is not None:
        if np.isscalar(prop['y']):
            prop['y'] = np.asarray([prop['y']])
        else:
            prop['y'] = np.asarray(prop['y'])


def _apply_plot_prop(prop, axes):
    if prop.get('type') == 'figure':
        return
    if prop.get('type') == 'axes':
        return

    if prop['type'] == 'line':
        plotobj, = axes.plot(prop['x'], prop['y'])
    elif prop['type'] == 'errorbar':
        plotobj = axes.errorbar(
            prop['x']
            , prop['y']
            , xerr=prop['xerr']
            , yerr=prop['yerr']
            , fmt='none'
            , capsize=prop['capsize'])
    elif prop['type'] == 'scatter':
        plotobj = axes.scatter(prop['x'], prop['y'], s=prop['s'], marker=prop.get('marker', None))
    elif prop['type'] == 'fill_between' or prop['type'] == 'fillbetween':
        plotobj = axes.fill_between(prop['x'], prop['y'], prop['y1'])
    elif prop['type'] == 'hline':
        plotobj = axes.axhline(prop['y'][0])
    elif prop['type'] == 'vline':
        plotobj = axes.axvline(prop['x'][0])
    elif prop['type'] == 'hspan':
        plotobj = axes.axhspan(prop['y'][0], prop['y'][1])
    elif prop['type'] == 'vspan':
        plotobj = axes.axvspan(prop['x'][0], prop['x'][1])
    elif prop['type'] == 'image':
        # Some value (any value) must be given for the extent.
        plotobj = axes.imshow(prop['z'], aspect='auto', extent=prop['extent'])
    else:
        raise Exception('Plot type unknown: {}'.format(prop['type']))

    # Certain plot types actually consist of collections of line objects.
    if prop['type'] == 'errorbar':
        obj_list = []
        obj_list.extend(plotobj[1])
        obj_list.extend(plotobj[2])
    else:
        obj_list = [plotobj]

    # Loop through all objects and set the appropriate properties.
    for key in prop:
        for obj in obj_list:
            obj_dir = dir(obj)
            if key == 'markersize':
                if prop['type'] == 'scatter':
                    sizes = obj.get_sizes()
                    sizes[:] = prop['markersize']
                    obj.set_sizes(sizes)
                else:
                    obj.set_markersize(prop['markersize'])
            else:
                # This will catch any properties can can be simply set with a
                # plot.set_prop(value) function.
                funcname = 'set_' + key
                if funcname in obj_dir:
                    if prop[key] is not None:
                        getattr(obj, funcname)(prop[key])


def _apply_axes_prop(prop, axes):
    if prop.get('type') == 'figure':
        return

    prop = copy.copy(prop)

    # In some cases the order of these statements is important.
    # (For example xscale needs to come before xbound I think.)

    axes.tick_params('both'
                     , direction='in'
                     , which='both'
                     , top='on'
                     , bottom='on'
                     , left='on'
                     , right='on')

    ax_dir = dir(axes)
    for key in prop:
        if key == 'xscale':
            if prop['xscale'] == 'log':
                nonpositive = 'clip'
            else:
                nonpositive = None
            axes.set_xscale(prop['xscale'], nonpositive=nonpositive)
        elif key == 'yscale':
            if prop['yscale'] == 'log':
                nonpositive = 'clip'
            else:
                nonposisitve = None
            axes.set_yscale(prop['yscale'], nonpositive=nonpositive)
        elif key == 'legend':
            axes.legend(loc=prop.get('legend_location')
                        , fontsize=prop.get('legend_fontsize')
                        , framealpha=prop.get('legend_framealpha')
                        )
        elif key == 'label_outer':
            axes.label_outer()
        else:
            # This will catch any properties can can be simply set with a
            # axes.set_prop(value) function.
            funcname = 'set_'+key
            if funcname in ax_dir:
                if prop[key] is not None:
                    getattr(axes, funcname)(prop[key])


def _apply_fig_prop(prop, ax):
    if not prop.get('type') == 'figure':
        return

    fig = ax.figure
    fig_dir = dir(fig)
    for key in prop:
        if key == 'suptitle':
            x = prop.get('suptitle_x', 0.02)
            y = prop.get('suptitle_y', 0.98)
            ha = prop.get('suptitle_ha', 'left')
            weight = prop.get('suptitle_weight')
            fig.suptitle(prop['suptitle'], x=x, y=y, ha=ha, weight=weight)
        else:
            # This will catch any properties can can be simply set with a
            # axes.set_prop(value) function.
            funcname = 'set_' + key
            if funcname in fig_dir:
                if prop[key] is not None:
                    getattr(fig, funcname)(prop[key])


def _make_figure(namelist):
    size = _get_figure_size(len(namelist))
    fig = matplotlib.pyplot.figure(figsize=size)
    return fig


def _make_axes(namelist, fig):

    numaxes = len(namelist)
    axesdict = {}
    for ii, name in enumerate(namelist):
        axesdict[name] = fig.add_subplot(numaxes, 1, ii + 1)

    fig.axesdict = axesdict

    return axesdict


def _autoname_plots(plotlist, sequential=False):
    """
    Automatically name any plots that were not given a name by the user.
    """
    namelist = []

    count = 0
    for plot in plotlist:
        if plot.get('type') == 'figure':
            continue
        name = plot.get('axes')
        if name is None:
            if sequential:
                num = count
            else:
                num = 0
            name = '_autoname_{:02d}'.format(num)
            count += 1
        plot['axes'] = name
        namelist.append(name)

    # Extract the unique names in order.
    namelist = list(dict.fromkeys(namelist))
    return namelist


def _get_figure_size(numaxes):
    """
    Return the default figure size.
    Width: 8 units
    Height: 3 units for every subplot or max 9 units
    Return
    ------

    (width, height)
      The figure size in inches.
    """

    figure_width = 8
    figure_height = max(6, min(numaxes * 3, 10))

    return (figure_width, figure_height)
