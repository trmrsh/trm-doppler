#!/usr/bin/env python

"""
memit carries out MEM iterations on an image.
"""

import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler

if __name__ == '__main__':

    # Fake up a Doppler map

    # a header
    mhead = fits.Header()
    mhead['OBJECT']   = ('IP Peg', 'Object name')
    mhead['TELESCOP'] = ('William Herschel Telescope', 'Telescope name')

    # create two square images
    ny, nx = 101, 101
    x      = np.linspace(-2000.,2000.,nx)
    y      = x.copy()
    vxy    = (x[-1]-x[0])/(nx-1)
    X, Y   = np.meshgrid(x, y)

    # create spot at 600, 300
    array1  = np.exp(-(((X-600.)/150.)**2+((Y-300.)/150.)**2)/2.)

    # add another at -300, -500
    array1 += np.exp(-(((X+300.)/150.)**2+((Y+500.)/150.)**2)/2.)

    # associate two wavelengths with this image
    wave1   = np.array((486.2, 434.0))
    gamma1  = np.array((100., 100.))
    def1    = doppler.Default.gauss2d(200.)
    scale1  = np.array((1.0, 0.5))

    # create Image
    image1  = doppler.Image(array1, vxy, wave1, gamma1, def1, scale1)

    # create spot at -300, -600
    array2   = np.exp(-(((X+300.)/300.)**2+((Y+300.)/300.)**2)/2.)
    array2  += 0.8*np.exp(-((np.sqrt(X**2+Y**2)-800.)/300.)**2/2.)

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

