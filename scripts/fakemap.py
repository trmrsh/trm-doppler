#!/usr/bin/env python

usage = \
"""
fakemap creates a fake Dopper map which it writes to disc as a FITS file.
"""

import argparse
import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler

parser = argparse.ArgumentParser(description=usage)

# positional
parser.add_argument('fake', help='name of fake image')

# optional
parser.add_argument('-nxy', type=int,   default=100, help='Number of pixels on a side in Vx-Vy space')
parser.add_argument('-vxy', type=float, default=40, help='km/s/pixel in Vx-Vy space')

# OK, done with arguments.
args = parser.parse_args()

nxy = args.nxy
vxy = args.vxy

# a header
mhead = fits.Header()
mhead['ORIGIN']   = ('fakemap.py', 'Origin of this image')

vrange = vxy*(nxy-1)/2.

v       = np.linspace(-vrange,vrange,nxy)
VX, VY  = np.meshgrid(v, v)

# create spot at 600, 300
array1  = np.exp(-(((VX-600.)/150.)**2+((VY-300.)/150.)**2)/2.)

# add another at -300, -500
array1 += 0.8*np.exp(-(((VX+300.)/150.)**2+((VY+500.)/150.)**2)/2.)

# associate two wavelengths with this image
wave1   = np.array((486.2, 434.0))
gamma1  = np.array((100., 100.))
scale1  = np.array((1.0, 0.5))

# Gaussian default
def1  = doppler.Default.gauss2d(500.)

# create Image
image1  = doppler.Image(array1, vxy, wave1, gamma1, def1, scale1)

# create spot at -300, -300
array2   = np.exp(-(((VX+300.)/300.)**2+((VY+300.)/300.)**2)/2.)

# add a crude disc
array2  += 0.8*np.exp(-((np.sqrt(VX**2+VY**2)-800.)/300.)**2/2.)

# associate with one wavelength
wave2   = 468.6
gamma2  = 150.
def2    = doppler.Default.gauss2d(500.)

# create Image
image2  = doppler.Image(array2, vxy, wave2, gamma2, def2)

# ephemeris etc
tzero  = 50000.
period = 0.1
vfine  = 10.
vpad   = 700.

# create the Map
map = doppler.Map(mhead,[image1,image2],tzero,period,vfine,vpad)

# Write to a fits file
if args.fake.endswith('.fits') or args.fake.endswith('.fits') or \
        args.fake.endswith('.fits.gz') or args.fake.endswith('.fit.gz'):
    oname = args.fake
else:
    oname = args.fake + '.fits'

map.wfits(oname)


