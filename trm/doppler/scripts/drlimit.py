#!/usr/bin/env python

import argparse
import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler
import copy

def drlimit(args=None):
    """drlimit limits the dynamic range in an image (which otherwise can cause
    problems) by raising the lowest value in an image to a user-specified
    fraction of the maximum.

    """

    parser = argparse.ArgumentParser(description=drlimit.__doc__)

    # positional
    parser.add_argument('inmap',   help='name of the input map')
    parser.add_argument('llim', type=float,  help='lower limit as fraction of maximum')
    parser.add_argument('outmap',  help='name of output map')

    # OK, done with arguments.
    args = parser.parse_args()

    # load map
    imap  = doppler.Map.rfits(doppler.afits(args.inmap))

    for image in imap.data:
        fmax = image.data.max()
        image.data = np.maximum(args.llim*fmax, image.data)

    # write the result to a FITS file
    imap.wfits(doppler.afits(args.outmap))

