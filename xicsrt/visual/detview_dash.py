# -*- coding: utf-8 -*-
"""
A Dash-based image viewer for use with pixelated detectors.

Reimplementation of detview_plotly.py using Dash for richer interactivity
inside a Jupyter notebook. The single "max scale" slider is replaced by a
*vertical* range slider (min + max) placed alongside the colorbar; both the
image and the colorbar update live as the range is dragged.

Requires: dash >= 2.11
    pip install dash
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output


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


def _prepare(image, options=None, **kwargs):
    """
    Resolve options and derive coordinates/projections.
    Returns (opt, image, x, y, xsum, ysum).
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

    return opt, image, x, y, xsum, ysum


def build_figure(image, options=None, **kwargs):
    """
    Build the 2x2 figure (projections + heatmap). Same layout as the original
    Plotly version, but the colorbar is arranged so it can align with the
    Dash range slider.

    Returns
    -------
    (fig, vmin, vmax) : (plotly.graph_objects.Figure, float, float)
    """
    opt, image, x, y, xsum, ysum = _prepare(image, options=options, **kwargs)

    vmin = float(np.min(image))
    vmax = float(np.max(image))

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
    zmin0 = opt['xbound'][0] if opt.get('xbound') else vmin
    zmax0 = opt['xbound'][1] if opt.get('xbound') else vmax

    fig.add_trace(
        go.Heatmap(
            z=image.T,          # transpose so rows->y, cols->x
            x=x,
            y=y,
            colorscale=_map_cmap(opt['cmap']),
            colorbar=dict(
                title='counts',
                x=1.02,
                len=0.75,       # heatmap occupies the bottom 75% of the height
                y=0.0,
                yanchor='bottom',
            ),
            zmin=zmin0,
            zmax=zmax0,
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
        uirevision='keep',   # preserve zoom/pan across colorscale updates
    )

    return fig, vmin, vmax


def view(image, options=None, port=8050, jupyter_mode='inline',
         jupyter_height=None, debug=False, **kwargs):
    """
    Launch a Dash app showing the detector image with a *vertical* range
    slider next to the colorbar. Dragging either handle updates zmin/zmax of
    the heatmap; the colorbar follows automatically.

    Parameters
    ----------
    image : 2D numpy array
    options : dict, optional
    port : int
        Port for the Dash server.
    jupyter_mode : {'inline', 'external', 'tab', 'jupyterlab'}
        How the app is displayed in the notebook (Dash >= 2.11).
    jupyter_height : int, optional
        Height (px) of the inline iframe. If None, auto-derived from the
        figure height (fig height + 100 px) so nothing is cut off.
    debug : bool
        Passed to app.run.

    Returns
    -------
    dash.Dash
        The running app instance.
    """
    fig, vmin, vmax = build_figure(image, options=options, **kwargs)

    # A small pad so the extreme handles are reachable, and a sensible step.
    span = vmax - vmin
    if span <= 0:
        span = 1.0
    step = span / 1000.0

    # These fractions must match the colorbar geometry above:
    #   heatmap row occupies bottom 75% of the figure (row_heights=[0.25, 0.75])
    #   figure height = 800, margins t=30 / b=60 -> plotting area = 710 px
    fig_height = 800
    top_margin = 30
    bottom_margin = 60
    plot_area = fig_height - top_margin - bottom_margin  # 710

    # The colorbar has len=0.75, anchored to the bottom of the plotting area.
    bar_len_px = 0.75 * plot_area                       # ~532 px
    slider_top_px = top_margin + (plot_area - bar_len_px)  # 30 + 178 = 208

    # Auto-derive the notebook iframe height from the figure height.
    if jupyter_height is None:
        jupyter_height = int(fig.layout.height or fig_height) + 100

    app = Dash(__name__)

    app.layout = html.Div(
        style={'position': 'relative', 'width': f'{fig.layout.width}px'},
        children=[
            dcc.Graph(id='det-graph', figure=fig,
                      config={'displayModeBar': True}),

            # Vertical range slider, absolutely positioned so its track aligns
            # exactly with the colorbar span.
            html.Div(
                style={
                    'position': 'absolute',
                    'top': f'{slider_top_px}px',
                    'left': f'{fig.layout.width - 20}px',
                    'height': f'{bar_len_px}px',
                },
                children=[
                    dcc.RangeSlider(
                        id='crange',
                        min=vmin,
                        max=vmax,
                        step=step,
                        value=[vmin, vmax],
                        vertical=True,
                        verticalHeight=bar_len_px,
                        allowCross=False,
                        tooltip={'placement': 'left',
                                 'always_visible': False},
                        updatemode='drag',
                        marks=None,
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output('det-graph', 'figure'),
        Input('crange', 'value'),
        prevent_initial_call=True,
    )
    def _update_range(rng):
        lo, hi = rng
        if hi <= lo:
            hi = lo + step
        fig.data[0].zmin = lo
        fig.data[0].zmax = hi
        return fig

    app.run(port=port, jupyter_mode=jupyter_mode,
            jupyter_height=jupyter_height, debug=debug)
    return app
