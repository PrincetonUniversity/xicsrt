# -*- coding: utf-8 -*-
"""
.. Authors
     Nathan Bartlett <nbb0011@auburn.edu>
     Novimir Pablant <npablant@pppl.gov>

A set of apertures for ray filtering.
"""

import numpy as np
import xicsrt.tools.xicsrt_math as xm

def aperture_mask(X_local, m, aperture_info):
    """
    Generate a mask array for the given aperture (or array of apertures).
    """

    if aperture_info is None:
        return m

    # Convert the aperture info to an array, if was not already.
    aperture_info = xm.toarray_1d(aperture_info)

    m_out = m.copy()
    for aperture in aperture_info:
        aperture = _aperture_defaults(aperture)

        m_test = m.copy()
        m_test = aperture_selector(X_local, m_test, aperture, _internal=True)

        logic = aperture['logic']
        if logic == 'and':
            m_out[m] &= m_test[m]
        elif logic == 'not':
            m_out[m] &= ~m_test[m]
        elif logic == 'or':
            m_out[m] |= m_test[m]
        elif logic == 'nand':
            m_out[m] = ~(m_out[m] & m_test[m])
        elif logic == 'nor':
            m_out[m] = ~(m_out[m] | m_test[m])
        elif logic == 'xor':
            m_out[m] ^= m_test[m]
        elif logic == 'xnor':
            m_out[m] = ~(m_out[m] ^ m_test[m])
        else:
            raise Exception(f'Aperture logic "{logic}" is not known.')
            
    return m_out


def aperture_selector(X_local, m, aperture, _internal=False):
    """
    Will call the appropriate aperture function for the given aperture name.

    .. Note::
       This selector and all the individual function will modify the mask array
       in place.
    """

    if not _internal:
        aperture = _aperture_defaults(aperture)

    shape = aperture['shape']
    if shape == 'none':
        func = aperture_none
    elif shape == 'circle':
        func = aperture_circle
    elif shape == 'square':
        func = aperture_square
    elif shape == 'rectangle':
        func = aperture_rectangle
    elif shape == 'ellipse':
        func = aperture_ellipse
    elif shape == 'triangle':
        func = aperture_triangle
    else:
        raise Exception(f'Aperture shape: "{shape}" is not implemented.')

    return func(X_local, m, aperture)


def _aperture_defaults(aperture):
    new = {
        'shape':None,
        'origin':None,
        'logic':None,
        }
    new.update(aperture)

    if new['origin'] is None: new['origin'] = np.array([0.0, 0.0])
    if new['shape'] is None: new['shape'] = 'none'
    if new['logic'] is None: new['logic'] = 'and'

    new['origin'] = xm.toarray_1d(new['origin'])
    new['shape'] = new['shape'].lower()
    new['logic'] = new['logic'].lower()

    if 'size' in new:
        new['size'] = xm.toarray_1d(new['size'])

    if 'vertices' in new:
        new['vertices'] = np.asarray(new['vertices'])

    return new


def aperture_none(X_local, m, aperture):
    """
    An empty aperture object.
    """
    return m


def aperture_circle(X_local, m, aperture):
    """
    A circular Aperture.

    name: 'circle'

    size: [radius]
      Contains the radius of the aperture.
    """
    origin_x = aperture['origin'][0]
    origin_y = aperture['origin'][1]
    size = aperture['size'][0]
    m[m] &= (((X_local[m,0] - origin_x)**2 + (X_local[m,1] - origin_y)**2) < size**2)
    
    return m


def aperture_square(X_local, m, aperture):
    """
    A square Aperture.

    name: 'square'

    size: [x, y]
      Contains the x and y size (full width) of the aperture.
    """
    size = aperture['size'][0]
    origin_x = aperture['origin'][0]
    origin_y = aperture['origin'][1]
    m[m] &= (np.abs((X_local[m, 0] - origin_x)) < size/2)
    m[m] &= (np.abs((X_local[m, 1] - origin_y)) < size/2)

    return m


def aperture_rectangle(X_local, m, aperture):
    """
    A rectangular Aperture.

    name: 'rectangle'

    size: [x, y]
      Contains the x and y size (full width) of the aperture.
    """
    size_x = aperture['size'][0]
    size_y = aperture['size'][1]
    origin_x = aperture['origin'][0]
    origin_y = aperture['origin'][1]
    m[m] &= (np.abs((X_local[m,0] - origin_x)) < size_x / 2)
    m[m] &= (np.abs((X_local[m,1] - origin_y)) < size_y / 2)
    
    return m


def aperture_ellipse(X_local, m, aperture):
    """
    An elliptical Aperture.

    name: 'ellipse'

    size: [x, y]
      Contains the x and y size (full width) of the aperture.
    """
    size_x = aperture['size'][0]
    size_y = aperture['size'][1]
    origin_x = aperture['origin'][0]
    origin_y = aperture['origin'][1]
    m[m] &= ((((X_local[m,0] - origin_x)/size_x)**2 + ((X_local[m,1] - origin_y)/size_y)**2) < 1)
    
    return m


def aperture_triangle(X_local, m, aperture):
    """
    An triangular aperture defined by three vertices.

    name: 'ellipse'

    vertices: [[x0, y0], [x1,y1], [x2,y2]]
      Contains the three vertices of the aperture.
    """

    m[m] = xm.point_in_triangle_2d(
        X_local[m,0:2],
        aperture['vertices'][0,0:2]+aperture['origin'][0:2],
        aperture['vertices'][1,0:2]+aperture['origin'][0:2],
        aperture['vertices'][2,0:2]+aperture['origin'][0:2],
        )

    return m






















