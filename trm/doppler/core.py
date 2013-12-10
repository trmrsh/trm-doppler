#!/usr/bin/env python

"""
Doppler tomography package

Core classes & routines
"""

# FWHM/sigma for a gaussian
EFAC = 2.354820045

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

def afits(fname):
    """
    Appends .fits to a filename if it does not end
    with .fits, .fit, .fits.gz or .fit.gz
    """

    if fname.endswith('.fits') or fname.endswith('.fits') or \
       fname.endswith('.fits.gz') or fname.endswith('.fit.gz'):
        return fname
    else:
        return fname + '.fits'

def acfg(fname):
    """
    Appends .cfg to a filename if it does not end with it.
    """

    if fname.endswith('.cfg'):
        return fname
    else:
        return fname + '.cfg'

class DopplerError(Exception):
    pass

