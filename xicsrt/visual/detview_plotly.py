# -*- coding: utf-8 -*-
"""
A simple Plotly-based image viewer for use with pixelated detectors.

Reimplementation of detview.py using Plotly instead of Matplotlib/mirplot.

Portions of the code were created by generative AI (Claude Opus 4.8)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def view_static(image, options=None, **kwargs):
    """
    Plot an image along with column and row summation plots using Plotly.

    Parameters
    ----------
    image : 2D numpy array
        The detector image to display.
    options : dict, optional
        Dictionary of plotting options (can also be passed as keyword args).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if options is None:
        options = {}
    options.update(kwargs)
    opt = options

    # ---- Defaults (mirrors the original script) ----
    opt.setdefault('size', None)
    opt.setdefault('pixel_size', 0.1)
    opt.setdefault('aspect', 'equal')
    opt.setdefault('coord', 'index')
    opt.setdefault('scale', 1.0)
    opt.setdefault('units', None)
    opt.setdefault('cmap', 'turbo')
    opt.setdefault('xbound', None)
    opt.setdefault('ybound', None)

    image = np.asarray(image, dtype=float)

    # ---- Axis labels ----
    if not opt.get('xlabel'):
        opt['xlabel'] = 'x'
        if opt.get('units'):
            opt['xlabel'] += f" [{opt['units']}]"
    if not opt.get('ylabel'):
        opt['ylabel'] = 'y'
        if opt.get('units'):
            opt['ylabel'] += f" [{opt['units']}]"

    # ---- Coordinate setup ----
    if opt['coord'] == 'index':
        x = np.arange(image.shape[0], dtype=float)
        y = np.arange(image.shape[1], dtype=float)
    elif opt['coord'] == 'size':
        if opt['size'] is None:
            # Derive physical size from pixel_size if not given explicitly.
            opt['size'] = [
                image.shape[0] * opt['pixel_size'],
                image.shape[1] * opt['pixel_size'],
            ]
        # Pixel location defined by center.
        x = np.linspace(-0.5 * opt['size'][0], 0.5 * opt['size'][0], image.shape[0])
        y = np.linspace(-0.5 * opt['size'][1], 0.5 * opt['size'][1], image.shape[1])
    else:
        raise Exception(f"coord type {opt['coord']} unknown.")

    # ---- Apply scale ----
    x = x * opt['scale']
    y = y * opt['scale']

    # ---- Projections ----
    xsum = np.sum(image, axis=1)  # sum over columns -> function of x
    ysum = np.sum(image, axis=0)  # sum over rows    -> function of y

    # ---- Layout: 2x2 grid with shared axes ----
    #   top-left : column-sum (xsum) over x
    #   bottom-left : main image
    #   bottom-right : row-sum (ysum) over y
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.75, 0.25],
        row_heights=[0.25, 0.75],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        shared_xaxes=True,
        shared_yaxes=True,
        specs=[
            [{'type': 'xy'}, None],
            [{'type': 'heatmap'}, {'type': 'xy'}],
        ],
    )

    # ---- Main image (heatmap) ----
    fig.add_trace(
        go.Heatmap(
            z=image.T,          # transpose so rows->y, cols->x
            x=x,
            y=y,
            colorscale=_map_cmap(opt['cmap']),
            colorbar=dict(title='counts', x=1.02, len=0.75, y=0.375),
            zmin=opt['xbound'][0] if opt.get('xbound') else None,
            zmax=opt['xbound'][1] if opt.get('xbound') else None,
        ),
        row=2, col=1,
    )

    # ---- Column-sum plot (top) ----
    fig.add_trace(
        go.Scatter(x=x, y=xsum, mode='lines', line=dict(color='black'),
                   showlegend=False),
        row=1, col=1,
    )

    # ---- Row-sum plot (right), plotted horizontally ----
    fig.add_trace(
        go.Scatter(x=ysum, y=y, mode='lines', line=dict(color='black'),
                   showlegend=False),
        row=2, col=2,
    )

    # ---- Axis labels ----
    fig.update_xaxes(title_text=opt['xlabel'], row=2, col=1)
    fig.update_yaxes(title_text=opt['ylabel'], row=2, col=1)

    # ---- Equal aspect ratio for the image ----
    if opt.get('aspect', 'equal') == 'equal':
        fig.update_yaxes(scaleanchor='x', scaleratio=1, row=2, col=1)

    fig.update_layout(
        template='plotly_white',
        width=800,
        height=800,
        margin=dict(l=60, r=80, t=30, b=60),
    )

    return fig


def _map_cmap(name):
    """
    Map a matplotlib-style colormap name to a Plotly colorscale.
    Plotly supports 'turbo' natively; fall back to a sensible default otherwise.
    """
    supported = {
        'turbo': 'Turbo',
        'viridis': 'Viridis',
        'plasma': 'Plasma',
        'inferno': 'Inferno',
        'magma': 'Magma',
        'gray': 'Greys',
        'jet': 'Jet',
    }
    return supported.get(name.lower(), 'Turbo')


def view(image, options=None, **kwargs):
    """
    Version with an interactive contrast slider that adjusts the color scale
    limits (zmin/zmax) of the image, analogous to the RangeSlider control
    in the original add_controls().

    Best used inside a Jupyter notebook.
    """
    fig = view_static(image, options=options, **kwargs)
    image = np.asarray(image, dtype=float)

    vmin = float(np.min(image))
    vmax = float(np.max(image))

    # Build slider steps that adjust the heatmap's zmax.
    steps = []
    n_steps = 20
    for i in range(n_steps + 1):
        frac = i / n_steps
        zmax = vmin + frac * (vmax - vmin)
        # Guard against zmax == zmin.
        if zmax <= vmin:
            zmax = vmin + 1e-9
        steps.append(dict(
            method='restyle',
            label=f'{zmax:.0f}',
            args=[{'zmax': zmax}, [0]],  # trace index 0 = heatmap
        ))

    fig.update_layout(
        sliders=[dict(
            active=n_steps,
            currentvalue={'prefix': 'Max scale: '},
            pad={'t': 50},
            steps=steps,
        )]
    )

    return fig