"""
Defines the classes needed to represent the multiple spectra needed for 
Doppler imaging in a FITS-compatible manner.
"""

import numpy as np
from astropy.io import fits

from core import *

class Spectra(object):
    """
    Container for a set of spectra. This is meant to represent a homogenous
    set of spectra, e.g. a set of spectra taken on one arm of a spectrograph
    over two orbits of a binary for instance. The format is designed so that
    raw spectra can be submitted with minimal manipulation. The Data class
    takes a list of such objects to represent heterogeneous data.

    Attributes:

      flux : flux densities in the form of a 2D array, wavelength along
             X-axis, Y-values represent different times. These should
             usually have been continuum subtracted.

      ferr : uncertainties on the flux densities, 2D array matching flux

      wave : wavelengths, 2D array matching flux. These should be tied to
             a constant or near-enough constant zeropoint (e.g. heliocentric
             or better still, barycentric)

      time : 1D array of times, matching the Y-dimension of flux. These
             are the times at the centres of the exposures.

      expose : 1D array of exposure lengths (same units as the times).

      fwhm : FWHM of point spread function of spectra, in terms of pixels.
    """

    def __init__(self, flux, ferr, wave, time, expose, fwhm):
        """
        Creates a Spectra object.
        """

        # checks
        if not isinstance(flux, np.ndarray):
            raise DopplerError('Data.__init__: flux must be a numpy array')
        if not isinstance(ferr, np.ndarray):
            raise DopplerError('Data.__init__: ferr must be a numpy array')
        if not isinstance(wave, np.ndarray):
            raise DopplerError('Data.__init__: wave must be a numpy array')
        if not isinstance(time, np.ndarray):
            raise DopplerError('Data.__init__: time must be a numpy array')
        if not isinstance(expose, np.ndarray):
            raise DopplerError('Data.__init__: expose must be a numpy array')
        if len(flux.shape) != 2:
            raise DopplerError('Data.__init__: flux must be a 2D array')
        if not sameDims(flux,ferr):
            raise DopplerError('Data.__init__: flux and ferr are incompatible.')
        if not sameDims(flux,wave):
            raise DopplerError('Data.__init__: flux and wave are incompatible')
        if len(time.shape) != 1 or flux.shape[0] != len(time):
            raise DopplerError('Data.__init__: flux and time have conflicting sizes')
        if len(expose.shape) != 1 or flux.shape[0] != len(expose):
            raise DopplerError('Data.__init__: flux and expose have conflicting sizes')

        self.flux   = flux
        self.ferr   = ferr
        self.wave   = wave
        self.time   = time
        self.expose = expose
        self.fwhm   = fwhm

    @classmethod
    def fromHDUl(cls, hdul):
        """
        Creates a Spectra from a list of (at least) 5
        HDUs.
        """
        if len(hdul) < 5:
            raise DopplerError('Spectra.fromHDUl: at least 5 HDUs are required.')

        flux   = hdul[0].data
        fwhm   = hdul[0].header['FWHM']
        ferr   = hdul[1].data
        wave   = hdul[2].data
        time   = hdul[3].data
        expose = hdul[4].data

        return cls(flux,ferr,wave,time,expose,fwhm)

    def toHDUl(self):
        """
        Returns the Spectra as an equivalent list of
        astropy.io.ImageHDUs (*not* an HDUList), suited
        to adding onto other hdus for eventual writing
        to a FITS file
        """
        head = fits.Header()
        head['TYPE'] = 'Fluxes'
        head['FWHM']  = (self.fwhm, 'FWHM in pixels')
        hdul = [fits.ImageHDU(self.flux,head),]

        head = fits.Header()
        head['TYPE'] = 'Flux errors'
        hdul.append(fits.ImageHDU(self.ferr,head))

        head = fits.Header()
        head['TYPE'] = 'Wavelengths'
        hdul.append(fits.ImageHDU(self.wave,head))

        head = fits.Header()
        head['TYPE'] = 'Times'
        hdul.append(fits.ImageHDU(self.time,head))

        head = fits.Header()
        head['TYPE'] = 'Exposure times'
        hdul.append(fits.ImageHDU(self.expose,head))

        return hdul

    def __repr__(self):
        return 'Spectra(flux=' + repr(self.flux) + \
            ', ferr=' + repr(self.ferr) + ', wave=' + repr(self.wave) + \
            ', time=' + repr(self.time) + ', expose=' + repr(self.expose) + \
            ', fwhm=' + repr(self.fwhm) + ')'

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
            raise DopplerError('Data.__init__: head' +
                               ' must be a fits.Header object')

        try:
            for i, spectra in enumerate(data):
                if not isinstance(spectra, Spectra):
                    raise DopplerError('Data.__init__: element ' + str(i) +
                                       'of data is not a Spectra.')

            self.data = data
        except TypeError, err:
            if not isinstance(data, Spectra):
                raise DopplerError('Data.__init__: data must be a' +
                                   ' Spectra or a list of Spectra')
            self.data = [data,]

        self.head = head.copy()
        self.head.add_blank('............................')
        self.head['COMMENT'] = 'This file contains spectroscopic data for Doppler imaging.'
        self.head['COMMENT'] = 'Fluxes, flux errors and wavelengths are stored in 2D arrays'
        self.head['COMMENT'] = 'each row of which represents one spectrum.'
        self.head['COMMENT'] = 'Times and exposure times are stored in 1D arrays.'
        self.head['HISTORY'] = 'Created from a doppler.Data object'

    @classmethod
    def rfits(cls, fname):
        """
        Reads in data from a fits file. The primary HDU's header is
        read along with data which should be 2D representing flux densities.
        Three more HDUs are expected containing (1) uncertainties in the
        flux densities, (2) wavelengths, and (3) times.
        """
        hdul = fits.open(fname)
        if len(hdul) < 6 or len(hdul) % 5 != 1:
            raise DopplerError('Data.rfits: ' + fname + ' did not have 5n+1 HDUs')
        head = hdul[0].header
        data = []
        for nhdu in xrange(1,len(hdul),5):
            data.append(Spectra.fromHDUl(hdul[nhdu]))

        return cls(head, data)

    def wfits(self, fname, clobber=True):
        """
        Writes a Data to an hdu list
        """
        hdul  = [fits.PrimaryHDU(header=self.head),]
        for spectra in self.data:
            hdul += spectra.toHDUl()
        hdulist = fits.HDUList(hdul)
        hdulist.writeto(fname, clobber=clobber)

    def __repr__(self):
        return 'Data(head=' + repr(self.head) + \
            ', data=' + repr(self.data) + ')'

if __name__ == '__main__':

    # a header
    head = fits.Header()
    head['OBJECT']   = ('IP Peg', 'Object name')
    head['TELESCOP'] = ('William Herschel Telescope', 'Telescope name')

    # create arrays
    shape  = (10,100)
    flux   = np.zeros(shape)
    ferr   = 0.1*np.ones_like(flux)
    wave   = np.linspace(490.,510.,shape[1])
    time   = np.linspace(50000.,50000.1,shape[0])
    expose = 0.001*np.ones_like(time)
    fwhm   = 2.2

    # manipulate the fluxes to be vaguely interesting
    # (gaussian sinusoidally varying in velocity)
    wave, times = np.meshgrid(wave, time)
    flux = np.exp(-((wave-500.*(1+300./3e5*np.sin(
                        2.*np.pi*(times-50000.)/0.1)))/1.5)**2/2.)

    # create the Spectra
    spectra = Spectra(flux,ferr,wave,time,expose,fwhm)

    # create the Data
    data = Data(head,spectra)

    data.wfits('test.fits')

    d = Data.rfits('test.fits')
