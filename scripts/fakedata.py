#!/usr/bin/env python

usage = \
"""
fakedata creates a fake data template file which it writes to disc 
as a FITS file. The data file will be blank. Use 'comdat' to create
some realistic data.
"""

import argparse
import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler

parser = argparse.ArgumentParser(description=usage)

# positional
parser.add_argument('fake', help='name of the output data file')

# optional
parser.add_argument('-w1', type=float, default=420., help='Minimum wavelength')
parser.add_argument('-w2', type=float, default=520., help='Maximum wavelength')
parser.add_argument('-nw', type=int,   default=1000, help='Number of wavelengths')
parser.add_argument('-t1', type=float, default=420., help='First time')
parser.add_argument('-t2', type=float, default=520., help='Last time')
parser.add_argument('-ns', type=int,   default=40,   help='Number of spectra')
parser.add_argument('-e', dest='error', type=float, default=1.0, help='RMS noise level per pixel')
parser.add_argument('-f', dest='fwhm', type=float, default=200., help='FWHM resoluton (km/s)')
parser.add_argument('-ndiv', type=int,  default=1,   help='Spectrum sub-division factor')

# OK, done with arguments.
args = parser.parse_args()

# a header
head = fits.Header()
head['ORIGIN']   = ('fakedata.py', 'Origin of this data')

wave = np.linspace(args.w1, args.w2, args.nw)
time = np.linspace(args.t1, args.t2, args.ns)

wave, ignore = np.meshgrid(wave, time)

shape  = (10,100)
flux   = np.zeros_like(wave,dtype=np.float32)
ferr   = args.error*np.ones_like(flux)
expose = (time[-1]-time[0])/(len(time)-1)*np.ones_like(time, dtype=np.float32)
ndiv   = args.ndiv*np.ones_like(time,dtype=np.int)
fwhm   = args.fwhm

# create the Spectra
spectra = doppler.Spectra(flux,ferr,wave,time,expose,ndiv,fwhm)

# create the Data
data = doppler.Data(head,spectra)

# Write to a fits file
if args.fake.endswith('.fits') or args.fake.endswith('.fits') or \
        args.fake.endswith('.fits.gz') or args.fake.endswith('.fit.gz'):
    oname = args.fake
else:
    oname = args.fake + '.fits'

data.wfits(oname)


