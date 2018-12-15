#!/usr/bin/env python

import argparse
import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler
import copy

def comdef(args=None):
    """
    comdef computes the default equivalent to an image.
    """

    parser = argparse.ArgumentParser(description=comdef.__doc__)

    # positional
    parser.add_argument('map',   help='name of the input map')
    parser.add_argument('dout',  help='default output file')

    # OK, done with arguments.
    args = parser.parse_args()

    # load map
    dmap  = doppler.Map.rfits(doppler.afits(args.map))

    # copy the map to compute the entropy
    mcopy = copy.deepcopy(dmap)

    # compute default
    doppler.comdef(dmap)

    # write the result to a FITS file
    dmap.wfits(doppler.afits(args.dout))
