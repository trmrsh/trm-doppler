#!/usr/bin/env python

"""
Doppler tomography package

Core classes & routines
"""

import numpy as np

# Version number. To be updated
# if data formats change.
VERSION = 20131210

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

def meshgrid(nxy, vxy, nz=1, vz=0):
    """
    Carries out near-equivalent to numpy's meshgrid function for Doppler
    images. i.e. it returns grids of x,y or x,y,z which have C-style dimensions
    (nxy,nxy) or (nz,nxy,nxy) and values along the x,y,z axes equal to the Vx,
    Vy, Vz values for each point. These arrays are then useful for creating
    and manipulating images.

    Example: suppose we want to set an image "image" equal to a gaussian spot
    centred at Vx=100,Vy=500,Vz=100 to an array called "image", then the
    following would do this (nxy etc assumed set already, nz > 1)::

      x,y,z = meshgrid(nxy, nz, vxy, vz)

      # set an RMS and height for the spot
      sigma  = 50.
      height = 5.
      array  = height*np.exp(-((x-100)**2+(y-500)**2+(z-100)**2)/sigma**2/2.)

    Arguments::

      nxy : the number of pixels along the Vx & Vy sides.
      nz  : the number of planes in Vz
      vxy : the km/s/pixel in Vx-Vy (square pixels)
      vz  : the km/s spacing in Vz

    Returns::

      x, y : if nz > 1, Vx and Vy coordinates of every point in 2D image
             Each array is 2D

      x, y, z : if nz == 1, Vx, Vy and Vz coordinates of every point in 3D
                image. Each array is 3D
    """

    # total km/s range in Vx-Vy plane
    vrange = vxy*(nxy-1)/2.

    # create 1D arrays of coords in Vx-Vy
    x = y = np.linspace(-vrange,vrange,nxy)
    if nz == 1:
        # 2D case
        return np.meshgrid(x, y)
    else:
        nx = ny = nxy

        # create 1D array of coords in Vz
        vzrange = vz*(nz-1)/2.
        z = np.linspace(-vzrange,vzrange,nz)

        # carry out 3D version of meshgrid.
        # First make all arrays 3D
        x = x.reshape(1,1,nx)
        y = y.reshape(1,ny,1)
        z = z.reshape(nz,1,1)

        # now extend each over the other 2 dimensions
        x = x.repeat(ny,axis=1)
        x = x.repeat(nz,axis=0)

        y = y.repeat(nx,axis=2)
        y = y.repeat(nz,axis=0)

        z = z.repeat(nx,axis=2)
        z = z.repeat(ny,axis=1)

        return (x,y,z)

class DopplerError(Exception):
    pass
