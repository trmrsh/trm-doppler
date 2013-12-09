#!/usr/bin/env python

usage = \
"""
comdat computes the data corresponding to a Doppler map, with the option of
adding noise.
"""

import argparse
import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler

parser = argparse.ArgumentParser(description=usage)

# positional
parser.add_argument('map', help='name of the input map')

# optional
parser.add_argument('-n', dest='noise', store_action=True, help='add noise')
parser.add_argument('-r', dest='read', type=float, default=10., help='RMS read noise per pixel.')
parser.add_argument('-g', dest='gain', type=float, default=1.0, help='gain, electrons/unit')
parser.add_argument('-c', dest='cont', type=float, default=1.0, help='effective continuum level')


# OK, done with arguments.
args = parser.parse_args()



    # associate with one wavelength
    wave2   = 468.6
    gamma2  = 150.
    def2    = doppler.Default.gauss2d(300.)

    # create Image
    image2  = doppler.Image(array2, vxy, wave2, gamma2, def2)

    # ephemeris etc
    tzero  = 50000.
    period = 0.1
    vfine  = 20.
    vpad   = 700.

    # create the Map
    map = doppler.Map(mhead,[image1,image2],tzero,period,vfine,vpad)
    print 'created map'

    # Write to a fits file
    map.wfits('map.fits')

    # Calculate the default
    mdef = doppler.comdef(map)

    # Write the result to a FITS file
    mdef.wfits('def.fits')

