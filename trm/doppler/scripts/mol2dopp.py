#!/usr/bin/env python

import argparse
import numpy as np
from trm import doppler, molly
from astropy.io import fits

def mol2dopp(args=None):
    """mol2dopp reads a single molly file and converts it into a data file
    suitable for the Python Doppler package. The molly file should be
    continuum subtracted and cover the line or lines of interest.  All spectra
    should have the same number of pixels.

    """

    parser = argparse.ArgumentParser(description=mol2dopp.__doc__)

    # positional
    parser.add_argument('molly', help='name of the molly input file')
    parser.add_argument(
        'fwhm',  type=float,help='FWHM resolution of data in km/s'
    )
    parser.add_argument('dout',  help='data output file')

    # optional
    parser.add_argument(
        '-n', dest='nsub', type=int, default=1,
        help='sub-division factor to use to compute exposure smearing'
    )
    parser.add_argument(
        '-p', dest='usephases', action='store_true',
        help='use phases rather than times. Use this switch when converting phase-folded data'
    )

    # OK, done with arguments.
    args = parser.parse_args()
    if args.nsub < 0 or args.nsub > 100:
        print('ERROR: the nsub parameter should range from 1 to 100')
        exit(1)

    # load data [for other formats other than molly,
    # this would need changing]
    mspec = molly.rmolly(args.molly)

    # Set up 2D arrays of dimension:
    #
    # (rows,cols) = (ny,nx) = (number of spectra, number of pixels)
    #
    # NB In this script we assume all spectra have the same number
    # of pixels. It is also possible to cope with more heterogenous
    # sets of data by creating multiple "Spectra" objects (see below)
    #
    # The arrays contain the fluxes, errors on the fluxes and wavelengths.
    # Note that the wavelengths do not need to be precisely the same for all
    # spectra, or have a special scale, e.g. logarithmic, although they ought
    # to smoothly vary from pixel to pixel.
    flux   = np.empty((len(mspec),len(mspec[0])))
    ferr   = np.empty((len(mspec),len(mspec[0])))
    wave   = np.empty((len(mspec),len(mspec[0])))

    # 1D arrays of dimension (number of spectra) containing the mid-exposure
    # time (in any timescale you like, HMJD here), the exposure times in the
    # same units (i.e. days here), and sub-division factors which are used to
    # allow for exposure smearing.
    time   = np.empty((len(mspec),))
    expose = np.empty((len(mspec),))
    nsub   = args.nsub*np.ones_like(expose,dtype=int)

    # wind through the spectra, loading the molly spectra
    # into the 2D arrays and reading the mid-exposure times
    # and lengths into the 1D arrays.
    for n, spc in enumerate(mspec):
        flux[n,:] = spc.f
        ferr[n,:] = spc.fe
        wave[n,:] = spc.wave
        if args.usephases:
            period = spc.head['PeriodO']
            time[n]   = spc.head['Orbital phase']
            expose[n] = spc.head['Dwell']/86400./period
        else:
            time[n]   = spc.head['HJD']-2400000.5
            expose[n] = spc.head['Dwell']/86400.

    if np.isnan(flux).any() or np.isnan(ferr).any() or np.isnan(wave).any():
        print('One or more of flux / ferr / wave contains a NaN')
 
    # Create a "Spectra" object [this is where you could cope with
    # heterogenous data by creating more than one such object]
    spectra = doppler.Spectra(flux, ferr, wave, time, expose, nsub, args.fwhm)

    # Blank header
    head = fits.Header()

    # Create a "Data" object
    data = doppler.Data(head, spectra)

    # Write out the Data object for use by the rest of the routines.
    data.wfits(args.dout)
