#!/usr/bin/env python

usage = \
"""
optscl computes optimum scaling and in useful when starting to get into
the right ballpark and thereby save iterations.
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
parser.add_argument('data',  help='data file')
parser.add_argument('scaled',  help='scaled output map')

# optional
#parser.add_argument('-n', dest='noise', action='store_true',
#                    help='add noise according to uncertainty array in template')

# OK, done with arguments.
args = parser.parse_args()

# load map and data
dmap  = doppler.Map.rfits(doppler.afits(args.map))
data  = doppler.Data.rfits(doppler.afits(args.data))

# compute data equivelent to data
dcalc = copy.deepcopy(data)
doppler.comdat(dmap, dcalc)

# compute optimum scale factor.
sum0 = 0.
sum1 = 0.
sum2 = 0.
ndata = 0
for cspec, dspec in zip(dcalc.data, data.data):
    ok    = dspec.ferr > 0.
    sum0 += ((dspec.flux[ok]/dspec.ferr[ok])**2).sum()
    sum1 += ((cspec.flux[ok]/dspec.ferr[ok])*(dspec.flux[ok]/dspec.ferr[ok])).sum()
    sum2 += ((cspec.flux[ok]/dspec.ferr[ok])**2).sum()
    ndata += len(dspec.ferr[ok])

scale = sum1 / sum2
chisq = sum0 - sum1**2/sum2
print 'Optimum scale factor =',scale
print 'Chi**2/ndata (before) =',(sum0-2*sum1+sum2)/ndata,' (after) =',chisq/ndata

# scale images and write out
for image in dmap.data:
    image.data *= scale

# Write to a fits file
dmap.wfits(doppler.afits(args.scaled))
