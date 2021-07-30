# -*- coding:utf-8 -*-
"""
.. Authors
    Novimir pablant <npablant@pppl.gov>
    Yevgeniy Yakusevich <eugenethree@gmail.com>
    James Kring <jdk0026@tigermail.auburn.edu>

A set of tools for 2d visualization of the XICSRT results
"""
import numpy as np
import logging

from matplotlib import pyplot

from xicsrt.util import mirplot
from xicsrt import xicsrt_public
from xicsrt.tools import xicsrt_aperture

from xicsrt.visual import detview

log = logging.getLogger(__name__)

def plot_example(results, name):
    """
    A simplified plotting routine to serve as an example of how to develop
    xicsrt visualizations. This function will plot found ray intersections.
    """

    # Retrieve an object for the given optic/source.
    obj = xicsrt_public.get_element(results['config'], name)

    # Transform from global coordinate to local optic coordinates
    origin_ext = results['found']['history'][name]['origin']
    origin_loc = obj.point_to_local(origin_ext)

    # Use the mirplot utility to generate the plot.
    plotlist = [{
        'type':'scatter',
        'x':origin_loc[:, 0],
        'y':origin_loc[:, 1],
        }]
    fig = mirplot.plot_to_screen(plotlist)

    return fig


def plot_intersect(*args, **kwargs):
    """
    Plot the intersection of rays with the given optic.

    Parameters
    ----------
    results
      The restults dictionary from `raytrace()` that include the ray history.

    Keywords
    --------
    name :string (None)
      The name of the optic or source for which to plot intersections. The name
      refers to the key of the entry in the config dictionary. For example
      the name 'detector' will refer to config['optics']['detector'].

    section :string (None)
      [Optional] The name of the config section in which to search for `name`.
      This should typically be either 'optics' or 'sources'. If no section is
      given then then 'optics' will be searched first, then 'sources'.

    options :dict (None)
      [Optional] A dictionary containing plot options. All options can also be
      passed individually as keywords.

    Returns
    -------
    plotlist
      Will return a plotlist with the full plot definition.
    """

    # Aspect is handled here, so turn it off in the plotlist.
    plotlist = _get_intersect_plotlist(*args, **kwargs, _noaspect=True)

    gs = {
        'width_ratios':[3, 1],
        'height_ratios':[1, 3],
        'wspace':0.1,
        'hspace':0.1,
    }
    fig, axs = pyplot.subplots(2, 2, gridspec_kw=gs)
    axs[0, 1].set_axis_off()

    axesdict = {
        'scatter':axs[1, 0],
        'xhist':axs[0, 0],
        'yhist':axs[1, 1],
    }

    for ax in axesdict.values():
        ax.label_outer()

    axesdict['xhist'].sharex(axesdict['scatter'])
    axesdict['yhist'].sharey(axesdict['scatter'])

    mirplot.plot_to_axes(plotlist, axesdict)

    if kwargs.get('aspect','auto') == 'equal':
        _update_lim_aspect(axesdict['scatter'])
        axesdict['scatter'].callbacks.connect('ylim_changed', _on_ylims_change)

    return fig


def _get_intersect_plotlist(
        results,
        name=None,
        section=None,
        options=None,
        _noaspect=False,
        **kwargs,
        ):
    """
    Return a plotlist for :func:`plot_intersect`.
    """

    if options is None:
        opt = {}
    else:
        opt = options
    opt.update(kwargs)

    opt.setdefault('name', name)
    opt.setdefault('found', True)
    opt.setdefault('lost', True)
    opt.setdefault('bounds', True)
    opt.setdefault('aperture', True)
    opt.setdefault('aspect', 'equal')
    opt.setdefault('scale', None)
    opt.setdefault('units', None)
    opt.setdefault('lost_color', 'royalblue')
    opt.setdefault('found_color', 'red')
    opt.setdefault('alpha', None)
    opt.setdefault('lost_alpha', None)
    opt.setdefault('found_alpha', None)
    opt.setdefault('xbound', None)
    opt.setdefault('ybound', None)
    opt.setdefault('hist_size', None)
    opt.setdefault('hist_bins', None)
    opt.setdefault('hist_norm', False)
    opt.setdefault('drawstyle', 'steps-mid')
    opt.setdefault('linewidth', None)

    if opt['name'] is None:
        opt['name'] = 'detector'

    if (opt['alpha'] is None) and (opt['lost_alpha'] is None):
        opt['lost_alpha'] = 0.1
    elif (opt['lost_alpha'] is None):
        opt['lost_alpha'] = opt['alpha']

    if (opt['alpha'] is None) and (opt['found_alpha'] is None):
        opt['found_alpha'] = 0.5
    elif (opt['found_alpha'] is None):
        opt['found_alpha'] = opt['alpha']

    if opt['scale'] is None:opt['scale'] = 1.0

    if not opt.get('xlabel'):
        opt['xlabel'] = f'x'
        if opt.get('units'):
            opt['xlabel'] += f" [{opt['units']}]"

    if not opt.get('ylabel'):
        opt['ylabel'] = f'y'
        if opt.get('units'):
            opt['ylabel'] += f" [{opt['units']}]"

    if _noaspect:
        opt['aspect'] = None

    plotlist = []

    config = results['config']
    name = opt['name']

    # Get the crystal object from the dispatcher.
    obj = xicsrt_public.get_element(config, name, section)

    if opt['xbound'] is None:
        opt['xbound'] = np.array([-1 * obj.param['xsize'] / 2, obj.param['xsize'] / 2]) * 1.2
        opt['xbound'] *= opt['scale']
    else:
        opt['xbound'] = np.asarray(opt['xbound'])*opt['scale']

    if opt['ybound'] is None:
        opt['ybound'] = np.array([-1 * obj.param['ysize'] / 2, obj.param['ysize'] / 2]) * 1.2
        opt['ybound'] *= opt['scale']
    else:
        opt['ybound'] = np.asarray(opt['ybound'])*opt['scale']

    plotlist.extend([{'type':'figure', 'suptitle':name}])

    # Begin building plotlist.
    if opt['bounds']:
        plotlist.extend(_get_bounds_plotlist(obj, opt['scale']))

    if opt['aperture']:
        plotlist.extend(_get_aperture_plotlist(obj, opt['scale']))

    plotlist.extend(_get_hist(obj, results, opt, raytype='found', axis=0))
    plotlist.extend(_get_hist(obj, results, opt, raytype='found', axis=1))

    if opt['lost']:
        plotlist.extend(_get_rays_plotlist(obj, results, opt, raytype='lost'))

    if opt['found']:
        plotlist.extend(_get_rays_plotlist(obj, results, opt, raytype='found'))

    return plotlist


def _get_hist(obj, results, opt, raytype='found', axis=0):
    prefix = raytype + '_'
    name = opt['name']

    origin_ext = results[raytype]['history'][name]['origin']
    origin_loc = obj.point_to_local(origin_ext)
    mask = np.all(~ np.isnan(origin_ext[:, :]), axis=1)

    if axis == 0:
        range_ = opt['xbound']
        name = 'xhist'
    else:
        range_ = opt['ybound']
        name = 'yhist'

    if opt.get('hist_size'):
        binsize = opt['hist_size'] * opt['scale']
        bins = int(np.round((range_[1] - range_[0]) / binsize))
    elif opt.get('hist_bins'):
        bins = opt['hist_bins']
    elif obj.param.get('pixel_size'):
        binsize = obj.param['pixel_size'] * opt['scale']
        bins = int(np.round((range_[1] - range_[0]) / binsize))
    else:
        bins = 100

    # Some logging
    binsize = (range_[1] - range_[0])/bins
    log.info(f'Histogram bins, size:{binsize:0.6f} num:{bins:4d}')

    # Calculate the histogram.
    hist, bins = np.histogram(
        origin_loc[mask, axis] * opt['scale'],
        bins,
        range=range_)
    bins_c = (bins[0:-1] + bins[1:]) / 2

    if opt['hist_norm']:
        norm_scale = 1.0 / np.max(hist)
    else:
        norm_scale = 1.0

    if axis == 0:
        x = bins_c
        y = hist * norm_scale
    else:
        y = bins_c
        x = hist * norm_scale

    plotlist = []
    plotlist.append({
        'axes':name,
        'x':x,
        'y':y,
        'drawstyle':opt['drawstyle'],
        'color':'black',
        'linewidth':opt['linewidth'],
    })

    return plotlist


def _get_rays_plotlist(obj, results, opt, raytype='found'):
    prefix = raytype + '_'
    name = opt['name']

    origin_ext = results[raytype]['history'][name]['origin']
    origin_loc = obj.point_to_local(origin_ext)
    mask = np.all(~ np.isnan(origin_ext[:, :]), axis=1)

    log.debug(f'{raytype:5.5s} {name:10.10s} {sum(mask)}')

    plotlist = []
    if np.sum(mask) > 0:
        plotlist.append({
            'axes':'scatter',
            'type':'scatter',
            'x':origin_loc[mask, 0] * opt['scale'],
            'y':origin_loc[mask, 1] * opt['scale'],
            'xbound':opt['xbound'],
            'ybound':opt['ybound'],
            'aspect':opt['aspect'],
            'alpha':opt[raytype + '_alpha'],
            'color':opt[raytype + '_color'],
            'xlabel':opt['xlabel'],
            'ylabel':opt['ylabel'],
        })

    return plotlist


def _get_bounds_plotlist(obj, scale=None):
    if scale is None:scale = 1.0
    plotlist = []

    # Plot the optic extent as taken from the xsize and ysize.
    opt_x = obj.param['xsize'] / 2 * scale
    opt_y = obj.param['ysize'] / 2 * scale
    plotlist.append({
        'axes':'scatter',
        'x':[-1 * opt_x, opt_x, opt_x, -1 * opt_x, -1 * opt_x],
        'y':[opt_y, opt_y, -1 * opt_y, -1 * opt_y, opt_y],
        'linestyle':'--',
        'color':'gray',
    })

    return plotlist


def _get_aperture_plotlist(obj, scale=None):
    if scale is None:scale = 1.0
    plotlist = []

    if (not 'aperture' in obj.param) or (obj.param['aperture'] is None):
        return []

    for apt in obj.param['aperture']:
        apt = xicsrt_aperture._aperture_defaults(apt)
        shape = apt['shape']
        if shape == 'square':
            size = apt['size'][0] / 2 * scale
            origin = apt['origin'].copy() * scale
            x = np.array([-1, 1, 1, -1, -1]) * size + origin[0]
            y = np.array([1, 1, -1, -1, 1]) * size + origin[1]

        if shape == 'rectangle':
            size = apt['size'].copy() / 2 * scale
            origin = apt['origin'].copy() * scale
            x = np.array([-1, 1, 1, -1, -1]) * size[0] + origin[0]
            y = np.array([1, 1, -1, -1, 1]) * size[1] + origin[1]

        if shape == 'triangle':
            vert = apt['vertices'].copy() * scale
            origin = apt['origin'].copy() * scale
            x = np.array([vert[0, 0], vert[1, 0], vert[2, 0], vert[0, 0]]) + origin[0]
            y = np.array([vert[0, 1], vert[1, 1], vert[2, 1], vert[0, 1]]) + origin[0]
        else:
            log.warning(f'Plotting of {shape} aperture not yet implemented.')
            return []

        plotlist.append({
            'axes':'scatter',
            'x':x,
            'y':y,
            'linestyle':'-',
            'color':'black',
        })
    return plotlist


def _update_lim_aspect(ax):
    """
    Update the data limits (xlim & ylim) to produce an equal aspect given a
    fixed plot size.
    """
    xlim = ax.get_xlim()
    xspan = xlim[1] - xlim[0]
    ylim = ax.get_ylim()
    yspan = ylim[1] - ylim[0]
    h = ax.get_window_extent().height
    w = ax.get_window_extent().width
    w_aspect = h / w
    d_aspect = yspan / xspan

    if d_aspect < w_aspect:
        ylim_new = np.array([-0.5, 0.5]) * xspan * w_aspect + sum(ylim) / 2
        ax.set_ylim(ylim_new)
    else:
        xlim_new = np.array([-0.5, 0.5]) * yspan / w_aspect + sum(xlim) / 2
        ax.set_xlim(xlim_new)


def _on_ylims_change(event_ax):
    """
    An Axes callback to update the data limits after a change in the the data
    limits. This is primarily meant to allow retention af a fixed aspect
    after the user has zoomed into a region.
    """

    # We need to disconnect the callback before calling set_ylim
    # Otherwise we will end up in an infinite loop.
    cid_list = list(event_ax.callbacks.callbacks['ylim_changed'].keys())
    for cid in cid_list:
        event_ax.callbacks.disconnect(cid)

    _update_lim_aspect(event_ax)

    event_ax.callbacks.connect('ylim_changed', _on_ylims_change)


def plot_image(
        results,
        name=None,
        section=None,
        options=None,
        **kwargs,
        ):
    """
    Plot an intersection image along with column and row summation plots.
    """

    if options is None: options = {}
    options.update(kwargs)
    opt = options

    opt.setdefault('name', name)

    config = results['config']
    name = opt['name']

    # Get the crystal object from the dispatcher.
    obj = xicsrt_public.get_element(config, name, section)
    image = results['total']['image'][name]

    opt['pixel_size'] = obj.param['pixel_size']
    opt['size'] = (obj.param['xsize'], obj.param['ysize'])

    fig = detview.view(image, opt)
    controls = detview.add_controls(fig)
    return fig, controls
