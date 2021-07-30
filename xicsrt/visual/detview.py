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

def view(image, options=None, **kwargs):
    """
    Plot an image along with column and row summation plots.
    """

    if options is None: options = {}
    options.update(kwargs)
    opt = options


    opt.setdefault('size', None)
    opt.setdefault('pixel_size', None)
    opt.setdefault('aspect', 'equal')
    opt.setdefault('coord', 'pixel')
    opt.setdefault('scale', 1.0)
    opt.setdefault('units', None)
    opt.setdefault('cmap', 'turbo')

    if not opt.get('xlabel'):
        opt['xlabel'] = f'x'
        if opt.get('units'):
            opt['xlabel'] += f" [{opt['units']}]"

    if not opt.get('ylabel'):
        opt['ylabel'] = f'y'
        if opt.get('units'):
            opt['ylabel'] += f" [{opt['units']}]"

    gs = {
        'width_ratios':[3, 1],
        'height_ratios':[1, 3],
        'wspace':0.05,
        'hspace':0.05,
    }
    fig, axs = pyplot.subplots(2, 2, gridspec_kw=gs, figsize=(8,8))
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    axs[0, 1].set_axis_off()

    axesdict = {
        'image':axs[1, 0],
        'xsum':axs[0, 0],
        'ysum':axs[1, 1],
    }

    fig.axesdict = axesdict

    for ax in axesdict.values():
        ax.label_outer()

    axesdict['xsum'].sharex(axesdict['image'])
    axesdict['ysum'].sharey(axesdict['image'])

    if opt['coord'] == 'index':
        x = np.arange(image.shape[0])
        y = np.arange(image.shape[1])
    elif opt['coord'] == 'pixel':
        x = np.arange(image.shape[0])+opt['pixel_size']/2
        y = np.arange(image.shape[1])+opt['pixel_size']/2
    elif opt['coord'] == 'cpixel':
        # I haven't carefully checked that I am calculating this correctly.
        x = np.arange(image.shape[0])+opt['pixel_size']/2 - opt['xsize']/opt['pixel_size']/2
        y = np.arange(image.shape[1])+opt['pixel_size']/2 - opt['ysize']/opt['pixel_size']/2
    elif opt['coord'] == 'space':
        x = np.linspace(-0.5*opt['size'][0], 0.5*opt['size'][0], image.shape[0])
        y = np.linspace(-0.5*opt['size'][1], 0.5*opt['size'][1], image.shape[1])
    else:
        raise Exception(f"coord type {opt['coord']} unknown.")

    x *= opt['scale']
    y *= opt['scale']

    xsum = np.sum(image, axis=1)
    ysum = np.sum(image, axis=0)

    plotlist = []
    plotlist.append({
        'axes':'image',
        'type':'image',
        'x':x,
        'y':y,
        'z':np.rot90(image),
        'cmap':opt['cmap'],
        'xlabel':opt['xlabel'],
        'ylabel':opt['ylabel'],
        })
    plotlist.append({
        'axes':'xsum',
        'type':'hline',
        'y':[0.0],
        'color':'black',
        'alpha':0.2,
        'linestyle':'--',
        })
    plotlist.append({
        'axes':'xsum',
        'x':x,
        'y':xsum,
        })
    plotlist.append({
        'axes':'ysum',
        'type':'vline',
        'x':[0.0],
        'color':'black',
        'alpha':0.2,
        'linestyle':'--',
        })
    plotlist.append({
        'axes':'ysum',
        'x':ysum,
        'y':y,
        'ybound':None,
        })
    mirplot.plot_to_axes(plotlist, axesdict)

    if opt.get('aspect','equal') == 'equal':
        _update_lim_aspect(axesdict['image'])
        axesdict['image'].callbacks.connect('ylim_changed', _on_ylims_change)

    return fig

def add_controls(fig):
    """
    Add a set of controls for viewing the image.

    Programming Notes
    -----------------
    Jupyter apparently won't make these controls interactive unless it can see
    them in the local scope. As long as this function is called directly in
    the notebook then everything appears to work ok.
    """
    from matplotlib.widgets import RangeSlider

    im = fig.axesdict['image'].get_images()[0]
    image = im.get_array()

    fig.subplots_adjust(top=0.95, bottom=0.125, left=0.125, right=0.95)

    w = fig.axesdict['image'].get_position().width
    box = [0.125, 0.025, w, 0.02]

    # Create a background based on a count histogram.
    ax = fig.add_axes(box)
    ax.set_axis_off()
    hist, bins = np.histogram(
        image.flatten(),
        range=[image.min() - 0.5, image.max() + 0.5],
        bins=int(image.max() + 1),
        )
    bins_c = (bins[0:-1] + bins[1:]) / 2
    hist[0] = hist[1]
    z = np.reshape(hist, (1,len(hist)))
    ax.imshow(z, aspect='auto', cmap='gist_heat_r')
    ax.set_navigate(False)

    # Create the slider.
    ax = fig.add_axes(box)
    ax.patch.set_alpha(0.0)
    slider = RangeSlider(
        ax,
        '',
        im.norm.vmin,
        im.norm.vmax,
        valinit=(im.norm.vmin, im.norm.vmax),
        alpha=0.5,
        )

    def update(val):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        im.norm.vmin = val[0]
        im.norm.vmax = val[1]

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return slider


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
