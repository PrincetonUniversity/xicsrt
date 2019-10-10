# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>

Description
-----------
A set of math utilites for XICSRT.
"""

import numpy as np


def normalize(vector):
    """
    Normalize a vector or an array of vectors.
    If an array of vectors is given it should have the shape (N,M) where
      N: Number of vectors
      M: Vector length
    """

    if vector.ndim > 1:
        norm = np.linalg.norm(vector, axis=1)
        vector /= np.expand_dims(norm, 1)
    else:
        norm = np.linalg.norm(vector)
        vector /= norm

    return vector
