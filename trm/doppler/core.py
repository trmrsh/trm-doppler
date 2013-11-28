#!/usr/bin/env python

"""
Doppler tomography package

Core classes & routines
"""

def sameDims(arr1, arr2):
    """
    Checks that two numpy arrays have the same number of dimensions
    and the same dimensions
    """
    if arr1.ndim != arr2.ndim:
        return False

    for d1,d2 in zip(arr1.shape, arr2.shape):
        if d1 != d2:
            return False

    return True

class DopplerError(Exception):
    pass

