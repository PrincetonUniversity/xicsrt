# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Fable 5)
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

Detector image accumulation for the jaxrt engine.

Port of `TraceObject.make_image` from the numpy engine. The per-ray
Python loop is replaced by a jax scatter-add (`.at[].add()`), which
produces identical images.

This module was AI generated using Claude (Fable 5).
"""

import jax.numpy as jnp


def setup(param):
    """
    Extract static image parameters from an initialized optic param dict.

    Returns None if imaging is not enabled for this optic (no
    xsize/ysize configured).
    """
    if not param.get('enable_image', False):
        return None

    image_params = {
        'origin': jnp.asarray(param['origin'], dtype=jnp.float64),
        'orientation': jnp.asarray(param['orientation'], dtype=jnp.float64),
        'pixel_size': float(param['pixel_size']),
        'pixel_xsize': int(param['pixel_xsize']),
        'pixel_ysize': int(param['pixel_ysize']),
    }
    return image_params


def make_image(rays, image_params):
    """
    Bin the masked ray intersections into a pixel image.

    The channel coordinate is defined from the *center* of the bottom
    left pixel; the pixel coordinate is defined from the geometrical
    center of the optic. This exactly follows the numpy engine.
    """
    # Transform intersections to optic-local 'pixel' coordinates.
    xlocal = jnp.einsum(
        'ij,kj->ki', image_params['orientation'],
        rays['origin'] - image_params['origin'])
    pix = xlocal / image_params['pixel_size']

    # Convert from pixels to channels (origin at bottom-left pixel center).
    channel_x = pix[:, 0] + (image_params['pixel_xsize'] - 1) / 2
    channel_y = pix[:, 1] + (image_params['pixel_ysize'] - 1) / 2

    channel_x = jnp.round(channel_x).astype(int)
    channel_y = jnp.round(channel_y).astype(int)

    # Exclude rays that are masked out or that fall outside the pixel
    # grid (possible due to floating point rounding).
    inside = rays['mask']
    inside &= (channel_x >= 0) & (channel_x < image_params['pixel_xsize'])
    inside &= (channel_y >= 0) & (channel_y < image_params['pixel_ysize'])

    image = jnp.zeros((image_params['pixel_xsize'], image_params['pixel_ysize']))

    # Scatter-add one count per ray. Excluded rays are directed to an
    # out-of-bounds index, which jax's 'drop' scatter mode ignores.
    # (A positive out-of-bounds index is used because negative indices
    # would wrap around, numpy-style.)
    channel_x = jnp.where(inside, channel_x, image_params['pixel_xsize'])
    image = image.at[channel_x, channel_y].add(1.0, mode='drop')

    return image
