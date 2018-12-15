#!/usr/bin/env python

import argparse
from trm import doppler

def svdfit(args=None):
    """singular value decomposition. Computes the Grid that best matches a Data,
    allowing the user to control the number of singular values used with the
    parameter 'cond'. If cond < 1, then only singular values greater than cond
    as a fraction of the highest are used. If cond >=1 then it is rounded to
    the nearest integer and used as the number of the highest singular values
    to use.  The maximum possible value equals the number of points in the
    grid.

    """

    parser = argparse.ArgumentParser(description=svdfit.__doc__)

    # positional
    parser.add_argument('igrid',  help='name of the input grid')
    parser.add_argument('data',   help='data file')
    parser.add_argument('cond',   type=float, help='SVD conditioning parameter')
    parser.add_argument('ogrid',  help='name of the output grid')

    # optional
    parser.add_argument('-n', dest='ntdiv', type=int,
                        default=11, help='spectrum sub-division factor')

    # OK, done with arguments.
    args = parser.parse_args()

    # load grid and data
    grid = doppler.Grid.rfits(doppler.afits(args.igrid))
    data = doppler.Data.rfits(doppler.afits(args.data))

    chisq, cred, sing, s, x = doppler.svd(grid, data, args.cond, args.ntdiv, True)

    grid.data = x[0]
    grid.wfits(doppler.afits(args.ogrid))

    print('Chi**2        =',chisq[0])
    print('Chi**2 / Ndof =',cred[0])
