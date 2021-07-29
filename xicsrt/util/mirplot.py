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

If axis names are provided then plots will be added to separate subfigures
(stacked vertically). Each unique axis name will result in a new subfigure.

.. code:

    plotlist = [
        {'axis':'plot 1', 'x':x1, 'y':y1},
        {'axis':'plot 2', 'x':x2, 'y':y2},
        ]
    fig = mirplot.plot_to_screen(plotlist)

mirplot can also be used with predifined axes. For this purpose the axes must
be placed into a dictionary and passed to `plot_to_axes`.

.. code:

    fig, axs = plt.subplots(1, 2)

    axes = {
        'plot 1':axs[0],
        'plot 2':axs[1],
        }
    plotlist = [
        {'axis':'plot 1', 'x':x1, 'y':y1},
        {'axis':'plot 2', 'x':x2, 'y':y2},
        ]

    fig = mirplot2.plot_to_axes(plotlist, axes)

mirplot properties
------------------

A set of unique plot and axis properties are defined by mirplot to enable
a complete dictionary definition.

type : str ('line')
  Allowed Values: line, errorbar, scatter, fill_between, hline, vline, hspan,
  vspan.

legend : bool (false)
  Set to true to show the legend in this subplot.

matplotlib properties
---------------------

Any matplotlib plot or axis property that can be set using a simple
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
    axes = _make_axes(namelist, fig)
    plot_to_axes(plotlist, axes)

    matplotlib.pyplot.ion()
    if show:
        fig.show()

    return fig


def plot_to_file(plotlist, filename):

    fig = plot_to_screen(plotlist, show=False)
    fig.savefig(filename)
    m_log.info('Saved figure to file: {}'.format(filename))


def plot_to_axes(plotlist, axes):

    for plot in plotlist:
        _set_plot_defaults(plot)
        _clean_plot_prop(plot)

        if plot.get('type') == 'figure':
            axis = list(axes.values())[0]
            _apply_fig_prop(plot, axis)
            continue

        axis = plot['axis']
        
        if isinstance(axis, str):
            if axis in axes:
                axis = axes[axis]
            else:
                raise Exception(f'Named axis {axis} not found.')

        _apply_plot_prop(plot, axis)
        _apply_axis_prop(plot, axis)
        _apply_fig_prop(plot, axis)


def _set_plot_defaults(prop):
    prop.setdefault('type', 'line')

    if prop['type'] == 'figure':
        return
    if prop['type'] == 'axis':
        return

    prop.setdefault('x')
    prop.setdefault('y')
    prop.setdefault('xerr')
    prop.setdefault('yerr')

    if prop['x'] is None:
        prop['x'] = np.arange(len(prop['y']))

    prop.setdefault('s', 15)
    prop.setdefault('legend_fontsize', 12.0)
    prop.setdefault('legend_framealpha', 0.7)

    if not 'ybound' in prop:
        yrange = np.array([np.nanmin(prop['y']), np.nanmax(prop['y'])])
        prop['ybound'] = yrange + np.array([-0.1, 0.1]) * (yrange[1] - yrange[0])

def _clean_plot_prop(prop):
    """
    Check the plot properties and cleanup or provides errors.
    """
    if prop['type'] == 'figure':
        return
    if prop['type'] == 'axis':
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


def _apply_plot_prop(prop, axis):
    if prop['type'] == 'figure':
        return
    if prop['type'] == 'axis':
        return

    if prop['type'] == 'line':
        plotobj, = axis.plot(prop['x'], prop['y'])
    elif prop['type'] == 'errorbar':
        plotobj = axis.errorbar(
            prop['x']
            , prop['y']
            , xerr=prop['xerr']
            , yerr=prop['yerr']
            , fmt='none'
            , capsize=prop['capsize'])
    elif prop['type'] == 'scatter':
        plotobj = axis.scatter(prop['x'], prop['y'], s=prop['s'], marker=prop.get('marker', None))
    elif prop['type'] == 'fill_between' or prop['type'] == 'fillbetween':
        plotobj = axis.fill_between(prop['x'], prop['y'], prop['y1'])
    elif prop['type'] == 'hline':
        plotobj = axis.axhline(prop['y'][0])
    elif prop['type'] == 'vline':
        plotobj = axis.axvline(prop['x'][0])
    elif prop['type'] == 'hspan':
        plotobj = axis.axhspan(prop['y'][0], prop['y'][1])
    elif prop['type'] == 'vspan':
        plotobj = axis.axvspan(prop['x'][0], prop['x'][1])
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


def _apply_axis_prop(prop, axis):
    if prop['type'] == 'figure':
        return

    prop = copy.copy(prop)

    # In some cases the order of these statements is important.
    # (For example xscale needs to come before xbound I think.)

    axis.tick_params('both'
                     , direction='in'
                     , which='both'
                     , top='on'
                     , bottom='on'
                     , left='on'
                     , right='on')

    ax_dir = dir(axis)
    for key in prop:
        if key == 'xscale':
            if prop['xscale'] == 'log':
                nonposx = 'clip'
            else:
                nonposx = None
            axis.set_xscale(prop['xscale'], nonposx=nonposx)
        elif key == 'yscale':
            if prop['yscale'] == 'log':
                nonposy = 'clip'
            else:
                nonposy = None
            axis.set_yscale(prop['yscale'], nonposy=nonposy)
        elif key == 'legend':
            axis.legend(loc=prop.get('legend_location')
                        , fontsize=prop.get('legend_fontsize')
                        , framealpha=prop.get('legend_framealpha')
                        )
        elif key == 'label_outer':
            axis.label_outer()
        else:
            # This will catch any properties can can be simply set with a
            # axis.set_prop(value) function.
            funcname = 'set_'+key
            if funcname in ax_dir:
                if prop[key] is not None:
                    getattr(axis, funcname)(prop[key])


def _apply_fig_prop(prop, ax):
    if not prop['type'] == 'figure':
        return

    if prop.get('suptitle'):
        x = prop.get('suptitle_x', 0.02)
        y = prop.get('suptitle_y', 0.98)
        ha = prop.get('suptitle_ha', 'left')
        weight = prop.get('suptitle_weight')
        ax.figure.suptitle(prop['suptitle'], x=x, y=y, ha=ha, weight=weight)


def _make_figure(namelist):
    size = _get_figure_size(len(namelist))
    fig = matplotlib.pyplot.figure(figsize=size)
    return fig


def _make_axes(namelist, fig):

    numaxis = len(namelist)
    axes = {}
    for ii, name in enumerate(namelist):
        axes[name] = fig.add_subplot(numaxis, 1, ii + 1)

    fig.axesdict = axes
    def _get_axesdict(self):
        return fig.axesdict
    setattr(fig, 'get_axesdict', _get_axesdict)

    return axes


def _autoname_plots(plotlist, sequential=False):
    """
    Automatically name any plots that were not given a name by the user.
    """
    namelist = []

    count = 0
    for plot in plotlist:
        if plot.get('type') == 'figure':
            continue
        name = plot.get('axis')
        if name is None:
            if sequential:
                num = count
            else:
                num = 0
            name = '_autoname_{:02d}'.format(num)
            count += 1
        plot['axis'] = name
        namelist.append(name)

    # Extract the unique names in order.
    namelist = list(dict.fromkeys(namelist))
    return namelist


def _get_figure_size(numaxis):
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
    figure_height = max(6, min(numaxis * 3, 10))

    return (figure_width, figure_height)
