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
import copy

parser = argparse.ArgumentParser(description=usage)

# positional
parser.add_argument('map',   help='name of the input map')
parser.add_argument('dtemp', help='data template file')
parser.add_argument('dout',  help='data output file')

# optional
parser.add_argument('-n', dest='noise', action='store_true',
                    help='add noise according to uncertainty array in template')

# OK, done with arguments.
args = parser.parse_args()

# load map and data
dmap  = doppler.Map.rfits(doppler.afits(args.map))
dtemp = doppler.Data.rfits(doppler.afits(args.dtemp))

# compute data
dcopy = copy.deepcopy(dtemp)
doppler.comdat(dmap, dtemp)

# optionally add noise
if args.noise:
    for spectra in dtemp.data:
        spectra.flux = np.random.normal(spectra.flux, spectra.ferr)
else:
    chisq = 0.
    ndata = 0
    for cspec, dspec in zip(dtemp.data, dcopy.data):
        ok = dspec.ferr > 0.
        chisq += (((dspec.flux[ok]-cspec.flux[ok])/dspec.ferr[ok])**2).sum()
        ndata += len(cspec.flux[ok].flat)
    print 'Chi**2 = ',chisq,', chi**2/N =',chisq/ndata

# Write to a fits file
dtemp.wfits(doppler.afits(args.dout))
