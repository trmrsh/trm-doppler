#!/usr/bin/env python

"""
Defines the classes needed to represent the multiple spectra needed for 
Doppler imaging in a FITS-compatible manner.
"""

import numpy as np
try:
    from astropy.io import fits
except:
    import pyfits as fits
from core import *


class Spectra(object):
    """
    Container for a set of spectra. This is meant to represent a homogenous
    set of spectra, e.g. a set of spectra taken on one arm of a spectrograph
    over two orbits of a binary for instance. The format is designed so that
    raw spectra can be submitted with minimal manipulation. The Data class
    takes a list of such objects to represent heterogeneous data.

    Attributes:

      flux : flux densities in the form of a 2D array, wavelength along X-axis, Y-values
             represent different times. These should usually have been continuum subtracted.

      ferr : uncertainties on the flux densities, 2D array matching flux

      wave : wavelengths, 2D array matching flux. These should be tied to a constant or
             near-enough constant zeropoint (e.g. heliocentric or better still, barycentric)

      time : 1D array of times, matching the Y-dimension of flux. These are the times at the centres
             of the exposures.

      expose : 1D array of exposure lengths (same units as the times).

      fwhm : FWHM of point spread function of spectra, in terms of pixels.
    """

    def __init__(self, flux, ferr, wave, time, expose, fwhm):
        """
        Creates a Spectra object.
        """

        # some checks
        if len(flux.shape) != 2:
            raise DopplerError('Data.__init__: flux must be a 2D array')
        if flux.shape != ferr.shape:
            raise DopplerError('Data.__init__: flux and ferr have conflicting array sizes')
        if flux.shape != wave.shape:
            raise DopplerError('Data.__init__: flux and ferr have conflicting array sizes')
        if flux.shape[0] != len(time):
            raise DopplerError('Data.__init__: flux and time have conflicting sizes')
        if flux.shape[0] != len(time):
            raise DopplerError('Data.__init__: flux and expose have conflicting sizes')

        self.flux = flux
        self.ferr = ferr
        self.wave = wave
        self.time = time
        self.time = expose
        self.fwhm = fwhm

    def wfits(self, hdul):
        """
        Writes out the Spectra as a series of HDUs to an open FITS file 
        """
        pass


class Data(object):
    """
    This class represents all the data needed for Doppler tomogarphy.
    It has the following attributes::

      head : an astropy.io.fits.Header object

      data : a list of Spectra objects.
    """

    def __init__(self, head, data):
        """
        Creates a Data object

        head : an astropy.io.fits.Header object

        data : a Spectra or a list of Spectra
        """

        # some checks
        if not isinstance(head, fits.Header):
            raise DopplerError('Data.__init__: head must be a fits.Header object')
        self.head = head
        
        try:
            for i, spectra in enumerate(data):
                if not isinstance(spectra, Spectra):
                    raise DopplerError('Data.__init__: element ' + str(i) + 'of data is not a Spectra.')

            self.data = data
        except TypeError, err:
            if not isinstance(data, Spectra):
                raise DopplerError('Data.__init__: data must be a Spectra or a list of Spectra')
            self.data = [data,]
            
    def rfits(self, fname):
        """
        Reads in data from a fits file. The primary HDU's header is
        read along with data which should be 2D representing flux densities.
        Three more HDUs are expected containing (1) uncertainties in the
        flux densities, (2) wavelengths, and (3) times.
        """
        pass

    def wfits(self, fname):
        """
        Write out data to a fits file.
        """
        hdul = fits.open(fname,'w')
        pass


if __name__ == '__main__':

    # a header
    head = fits.Header()
    head['OBJECT']   = ('IP Peg', 'Object name')
    head['TELESCOP'] = ('William Herschel Telescope', 'Telescope name')
    
    # create arrays
    shape  = (10,100)
    flux   = np.empty(shape)
    ferr   = 0.1*np.ones_like(flux)
    wave   = np.empty_like(flux)
    time   = np.linspace(50000.,50000.1,shape[0])
    expose = 0.001*np.ones_like(flux)
    fwhm   = 2.2
                                                                
    # create the Spectra
    spectra = Spectra(flux,ferr,wave,time,expose,fwhm)
 
    # create the Data
    data = Data(head,spectra)

