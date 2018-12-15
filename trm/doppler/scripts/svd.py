#!/usr/bin/env python

import argparse
import numpy as np
from trm import doppler
from astropy.io import fits

def svd(args=None):
    """Singular value decomposition routine. This one generates a FITS file
    containing all the SVD vectors, each of which is a square image matching
    the input Grid. It returns this as a 3D array which can be conveniently
    examined with ds9. For an NxN grid, there are N**2 possible SVD vectors so
    the array has C-style dimensions (N*N) x N x N = N**4, and one should
    therefore not go mad with overlay large N. Use is mainly educational.

    """

    parser = argparse.ArgumentParser(description=svd.__doc__)

    # positional
    parser.add_argument('grid', help='name of the input grid')
    parser.add_argument('data', help='data file')
    parser.add_argument('svd', help='name of the SVD output file')

    # optional
    parser.add_argument(
        '-n', dest='ntdiv', type=int,
        default=11, help='spectrum sub-division factor'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    # load grid and data
    grid = doppler.Grid.rfits(doppler.afits(args.grid))
    data = doppler.Data.rfits(doppler.afits(args.data))

    # generate the matrix
    A = doppler.genmat(grid, data, args.ntdiv)

    # Carry out full SVD, returning smallest matrices possible
    u, s, v = np.linalg.svd(A,full_matrices=False)

    ng = grid.data.shape[0]
    # Matrix v is the one we want. It should have dimensions (ng*ng) x (ng*ng)
    # where N is number along each side of the grids. Need to re-shape
    v = np.reshape(v, (ng*ng,ng,ng))

    # now write to FITS

    head = fits.Header()
    head['SOURCE'] = 'svd.py'
    hdu = fits.PrimaryHDU(data=v, header=head)
    hdu.writeto(doppler.afits(args.svd))

