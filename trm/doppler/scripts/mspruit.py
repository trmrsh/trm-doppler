#!/usr/bin/env python

import argparse
from astropy.io import fits
import numpy as np
from trm import doppler

def mspruit(args=None):
    """Converts a Doppler map produced by Henk Spruit's Doppler tomography
    package into a trm.doppler Map object which is then written as a FITS file
    to disk.

    """

    parser = argparse.ArgumentParser(description=mspruit.__doc__)

    # positional
    parser.add_argument('map',   help='name of output file for map')

    # optional
    parser.add_argument(
        '-d', dest='dop', default='dop.out',
        help='name of file containing the Spruit map'
    )
    parser.add_argument(
        '-f', dest='fwhm', type=float, default=300.,
        help='FWHM of gaussian default'
    )
    parser.add_argument(
        '-t', dest='tzero', type=float, default=0.,
        help='tzero of ephemeris to be stored in map'
    )
    parser.add_argument(
        '-p', dest='period', type=float, default=0.1,
        help='period of ephemeris to be stored in map'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    with open(args.dop) as fin:
        nph, nvp, nv, w0, gamma = fin.readline().split()
        nph, nvp = int(nph), int(nvp)
        fin.readline().split()
        pha = []
        while len(pha) < nph:
            line = fin.readline()
            pha += [float(p) for p in line.split()]
        pha = np.array(pha)
        fin.readline()

        vp = []
        while len(vp) < nvp:
            line = fin.readline()
            vp += line.split()

        spec = []
        while len(spec) < nph*nvp:
            line = fin.readline()
            spec += line.split()

        fin.readline()
        nv, va, ds = fin.readline().split()
        nv, va = int(nv), float(va)
        mp = []
        while len(mp) < nv*nv:
            line = fin.readline()
            mp += [float(f) for f in line.split()]
        mp = np.reshape(np.array(mp),(nv,nv))

        vxy = 2*va/1.e5/(nv-1)

    # OK have map, prepare for output
    head  = fits.Header()
    head['ORIGIN']= "Henk Spruit's doppler tom package"
    image = doppler.Image(
        mp, doppler.PUNIT, vxy, w0, 0., doppler.Default.gauss2d(1., args.fwhm)
    )
    dmap = doppler.Map(head, image, args.tzero, args.period, 0., vxy)
    dmap.wfits(doppler.afits(args.map))

