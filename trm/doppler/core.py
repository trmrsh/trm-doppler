#!/usr/bin/env python

"""
Doppler tomography package

Core classes & routines
"""

import numpy as np

# Version number. To be updated
# if data formats change.
VERSION = 20210826

# FWHM/sigma for a gaussian
EFAC = 2.354820045

# Speed of light in km/s
CKMS = 299792.458

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

def meshgrid(nxy, vxy, nz=1, dvz=0):
    """Carries out near-equivalent to numpy's meshgrid function for
    Doppler images. i.e. it returns grids of vx,vy or vx,vy,vz which
    have C-style dimensions (nxy,nxy) or (nz,nxy,nxy) and values along
    the vx,vy,vz axes equal to the vx, vy, vz values for each
    point. These arrays are then useful for creating and manipulating
    images.

    Example: suppose we want to set an image "image" equal to a gaussian spot
    centred at Vx=100,Vy=500,Vz=100 to an array called "image", then the
    following would do this (nxy etc assumed set already, nz > 1)::

      vx,vy,vz = meshgrid(nxy, nz, vxy, dvz)

      # set an RMS and height for the spot
      sigma  = 50.
      height = 5.
      array  = height*np.exp(-((vx-100)**2+(vy-500)**2+(vz-100)**2)/sigma**2/2.)

    Arguments::

      nxy : the number of pixels along the Vx & Vy sides.
      nz  : the number of planes in Vz
      vxy : the km/s/pixel in Vx-Vy (square pixels)
      dvz : the km/s spacing in Vz

    Returns::

      vx, vy : if nz > 1, Vx and Vy coordinates of every point in 2D image
             Each array is 2D

      vx, vy, vz : if nz == 1, Vx, Vy and Vz coordinates of every point in 3D
                image. Each array is 3D

    """

    # total km/s range in Vx-Vy plane
    vrange = vxy*(nxy-1)/2.

    # create 1D arrays of coords in Vx-Vy
    if nz == 1:
        # 2D case
        vy,vx = np.mgrid[
            -vrange:vrange:eval(f'{nxy}j'),
            -vrange:vrange:eval(f'{nxy}j'),
        ]
        return (vx,vy)
    else:
        nx = ny = nxy
        vzrange = dvz*(nz-1)/2.
        vz,vy,vx = np.mgrid[
            -vzrange:vzrange:eval(f'{nz}j'),
            -vrange:vrange:eval(f'{nxy}j'),
            -vrange:vrange:eval(f'{nxy}j')
        ]
        return (vx,vy,vz)

class DopplerError(Exception):
    pass
