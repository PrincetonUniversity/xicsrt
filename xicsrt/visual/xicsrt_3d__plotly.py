# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

These are a set of routines for 3D visualization using the plotly library.

Example
-------
Example code for using this module within a Jupyter notebook.

.. code-block::

    import xicsrt.visual.xicsrt_3d__plotly as xicsrt_3d

    xicsrt_3d.figure()
    xicsrt_3d.add_rays(results)
    xicsrt_3d.add_optics(config)
    xicsrt_3d.add_sources(config)
    xicsrt_3d.show()
"""

import numpy as np
import scipy
import plotly.graph_objects as go
import matplotlib

from xicsrt import xicsrt_public
from xicsrt.objects._Dispatcher import Dispatcher

m_figure = None


def _thin_mask(mask, max_num):
    """
    Reduce the number of True elements in a mask array.

    Ray thinning is done randomly. Used to reduce the number of
    rays plotted.
    """
    num_mask = np.sum(mask)
    if num_mask > max_num:
        w = np.where(mask)[0]
        np.random.shuffle(w)
        mask[w[max_num:]] = False

    return mask


def figure(showbackground=False, visible=False):
    global m_figure

    layout = {
        'scene':{
            'aspectmode':'data'
            ,'xaxis':{
                'showbackground':showbackground
                ,'visible':visible
                }
            ,'yaxis':{
                'showbackground':showbackground
                ,'visible':visible
                }
            ,'zaxis':{
                'showbackground':showbackground
                ,'visible':visible
                }
            }
        }
    fig = go.Figure(layout=layout)
    m_figure = fig

    return fig


def show(figure=None):
    global m_figure
    if figure is None: figure = m_figure

    figure.show()

def add_rays(results, figure=None):
    _plot_ray_history(results['found']['history'], lost=False, figure=figure)
    _plot_ray_history(results['lost']['history'], lost=True, figure=figure)


def _plot_ray_history(history, lost=None, figure=None):
    global m_figure
    if figure is None: figure = m_figure

    if lost is False:
        color = 'rgba(255, 0, 0, 0.01)'
        name = 'found'
    elif lost is True:
        color = 'rgba(0, 0, 255, 0.01)'
        name = 'lost'
    else:
        color = 'rgba(0, 0, 0, 0.1)'
        name = 'other'
        
    num_elem = len(history)
    key_list = list(history.keys())
    
    for ii in range(num_elem-1):

        # All rays leaving this optic element.
        mask = history[key_list[ii]]['mask']

        # This is a temporary solution to filter lost rays for
        # which no intersection at the next optic was calculated.
        #
        # This is not a good solution overall since it is possible
        # to imagine a case where we would want to retain rays at
        # the origin.
        mask &= history[key_list[ii+1]]['origin'][:, 0] != 0.0

        num_mask = np.sum(mask)

        if num_mask == 0:
            continue
        
        x0 = history[key_list[ii]]['origin'][mask, 0]
        y0 = history[key_list[ii]]['origin'][mask, 1]
        z0 = history[key_list[ii]]['origin'][mask, 2]
        x1 = history[key_list[ii+1]]['origin'][mask, 0]
        y1 = history[key_list[ii+1]]['origin'][mask, 1]
        z1 = history[key_list[ii+1]]['origin'][mask, 2]
        nan = np.zeros(num_mask) * np.nan

        x = np.dstack((x0, x1, nan)).flatten()
        y = np.dstack((y0, y1, nan)).flatten()
        z = np.dstack((z0, z1, nan)).flatten()

        line = {'color': color}
        data = go.Scatter3d(
            x=x
            ,y=y
            ,z=z
            ,mode='lines'
            ,connectgaps=False
            ,line=line
            ,name=name
            ,showlegend=False)
        figure.add_trace(data)


def _add_trace_volume(obj, figure, name=None):
    global m_figure
    if figure is None: figure = m_figure

    w = obj.param['xsize'] / 2.0
    h = obj.param['ysize'] / 2.0
    d = obj.param['zsize'] / 2.0

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

    # Define all of the bounding points.
    if d > 0:
        tri = np.array((
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
        ))
    else:
        tri = np.array((
            (0, 3, 1)
            , (2, 3, 0)
        ))


    trace = go.Mesh3d(
        x=x
        ,y=y
        ,z=z
        ,i=tri[:, 0]
        ,j=tri[:, 1]
        ,k=tri[:, 2]
        ,flatshading=True
        ,opacity=0.50
        ,name=name)

    figure.add_trace(trace)

def _add_trace_mesh(obj, figure=None, name=None):
    """
    Add a meshgrid to the 3D plot.
    """
    global m_figure
    if figure is None: figure = m_figure

    faces = obj.param['mesh_faces']
    points = obj.param['mesh_points']

    if True:
        # Plot gridlines
        lines = []
        for f in faces:
            lines.append([f[0], f[1]])
            lines.append([f[0], f[2]])
            lines.append([f[1], f[2]])
        lines = np.array(lines)
        lines = np.unique(lines, axis=0)

        x0 = points[lines[:, 0], 0]
        y0 = points[lines[:, 0], 1]
        z0 = points[lines[:, 0], 2]
        x1 = points[lines[:, 1], 0]
        y1 = points[lines[:, 1], 1]
        z1 = points[lines[:, 1], 2]
        nan = np.zeros(len(x0)) * np.nan

        # Add nans between each line which Scatter3D will use to define linebreaks.
        x = np.dstack((x0, x1, nan)).flatten()
        y = np.dstack((y0, y1, nan)).flatten()
        z = np.dstack((z0, z1, nan)).flatten()

        trace = go.Scatter3d(
            x=x
            , y=y
            , z=z
            , mode='lines'
            , line={'color': 'black'}
            , connectgaps=False
            , showlegend=False)
        figure.add_trace(trace)

    if True:
        # Plot surface
        norm = matplotlib.colors.Normalize(np.min(points[:, 2]), np.max(points[:, 2]))
        cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
        color = cm.to_rgba(points[:, 2], alpha=0.75)

        trace = go.Mesh3d(
            x=points[:, 0]
            , y=points[:, 1]
            , z=points[:, 2]
            , i=faces[:, 0]
            , j=faces[:, 1]
            , k=faces[:, 2]
            , vertexcolor=color
            , flatshading=True
            , opacity=0.75)

        figure.add_trace(trace)


def add_optics(config, figure=None):
    section = 'optics'
    for name in config[section]:
        add_object(config, name, section, figure=figure)


def add_sources(config, figure=None):
    section = 'sources'
    for name in config[section]:
        add_object(config, name, section, figure=figure)


def add_object(config, name, section, figure=None):

    # Use the dispatcher to instantiate and initialize objects.
    optics = Dispatcher(config, section)
    optics.instantiate(name)
    optics.setup()
    optics.initialize()

    # Get the crystal object from the dispatcher.
    obj = optics.get_object(name)

    plot_mesh = obj.param.get('use_meshgrid', False)

    if plot_mesh:
        _add_trace_mesh(obj, figure, name)
    else:
        _add_trace_volume(obj, figure, name)


def _gen_fluxsurface_mesh(obj, s, range_m=None, range_n=None):
    """
    Generate points on a flux surface.
    The given input object must have a method 'car_from_flx'.
    """
    if range_m is None: range_m = (0, 2*np.pi)
    if range_n is None: range_n = (-np.pi/4, np.pi/4)

    num_m = 51
    num_n = 51

    num_points = num_m*num_n
    flx = np.empty((num_points, 3))

    val_m = np.linspace(range_m[0], range_m[1], num_m, endpoint=True)
    val_n = np.linspace(range_n[0], range_n[1], num_n, endpoint=True)

    num_points = num_m*num_n
    flx = np.empty((num_points, 3))

    val_mm, val_nn = np.meshgrid(val_m, val_n, indexing='ij')
    flx[:, 0] = s
    flx[:, 1] = val_mm.flatten()
    flx[:, 2] = val_nn.flatten()

    # This should be callable without a loop, but for now leave this as is
    # to support the VMEC stelltools module.
    car = np.empty(flx.shape)
    for ii in range(flx.shape[0]):
        car[ii, :] = obj.car_from_flx(flx[ii, :])

    return flx, car


def add_fluxsurfaces(
        config,
        name,
        section=None,
        figure=None,
        alpha=None,
        flatshading=None,
        **kwargs):
    """
    Plot the 3D flux surfaces of a plasma source.
    This should work for any object that has a 'car_from_flx' method.
    """
    global m_figure
    if figure is None: figure = m_figure

    if alpha is None: alpha = 0.5
    if flatshading is None: flatshading = True

    obj = xicsrt_public.get_element(config, name, section)

    norm = matplotlib.colors.Normalize(0.0, 1.0)
    cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma_r')

    rho_array = np.linspace(0.0, 1.0, 10, endpoint=True)
    for ii, rho in enumerate(reversed(rho_array)):
        flx, car = _gen_fluxsurface_mesh(obj, rho**2, **kwargs)

        # Triangulation only needs to be done once since all the meshes
        # are generated using the same point ordering.
        if ii == 0:
            tri = scipy.spatial.Delaunay(flx[:, 1:])

        faces = tri.simplices

        color = np.array(cm.to_rgba(rho))*255
        color = color.astype(int)
        color_string = f'rgb({color[0]}, {color[1]}, {color[2]})'

        trace = go.Mesh3d(
            x=car[:, 0],
            y=car[:, 1],
            z=car[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color_string,
            flatshading=flatshading,
            opacity=alpha
        )

        figure.add_trace(trace)

