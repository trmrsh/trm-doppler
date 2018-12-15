#!/usr/bin/env python

import argparse
import numpy as np
import pylab as plt
import copy
from astropy.io import fits
from trm import doppler

def trtest(args=None):
    """trtest carries out a test of tropus. The idea is to generate random inputs
    for opus and tropus and compute two quadratic forms that should
    match. This routine is only of interest for development. The reported
    numbers should be << 1.

    """

    parser = argparse.ArgumentParser(description=trtest.__doc__)

    # positional
    parser.add_argument('map', help='name of the input map template')
    parser.add_argument('data', help='data template file data template')

    # optional
    parser.add_argument(
        '-n', dest='ntrial', type=int, default=1,
        help='number of tests to carry out.'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    if args.ntrial < 1:
        print('You must carry out at least one trial.')
        exit(1)

    # load map and data
    dmap = doppler.Map.rfits(doppler.afits(args.map))
    ddat = doppler.Data.rfits(doppler.afits(args.data))

    # make copies
    cmap = copy.deepcopy(dmap)
    cdat = copy.deepcopy(ddat)

    for i in range(args.ntrial):

        # fill map with random noise
        for image in dmap.data:
            image.data = np.asarray(
                np.random.normal(size=image.data.shape),dtype=np.float32
            )

        #  enact opus
        doppler.comdat(dmap, cdat)

        # fill data with random noise
        for spectra in ddat.data:
            spectra.flux = np.asarray(
                np.random.normal(size=spectra.flux.shape),dtype=np.float32
            )

        # enact tropus
        doppler.datcom(ddat, cmap)

        # now compute quadratic forms
        qform1 = 0.
        dnorm1 = 0.
        dnorm2 = 0.
        for spectra1, spectra2 in zip(ddat.data, cdat.data):
            qform1 += (spectra1.flux*spectra2.flux).sum()
            dnorm1 += (spectra1.flux**2).sum()
            dnorm2 += (spectra2.flux**2).sum()

        qform2 = 0.
        mnorm1 = 0.
        mnorm2 = 0.
        for image1, image2 in zip(dmap.data, cmap.data):
            qform2 += (image1.data*image2.data).sum()
            mnorm1 += (image1.data**2).sum()
            mnorm2 += (image2.data**2).sum()

        print(
            'Test',i+1,'=',
            abs(qform2-qform1)/(dnorm1*dnorm2*mnorm1*mnorm2)**0.25
        )
