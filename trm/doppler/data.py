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
    over two orbits of a binary for instance. Each spectrum must have the same
    number of pixels, but there can be wavelength shifts between spectra, and
    their scales can be non-linear, so pretty much raw data can be passed
    through. The Data class takes a list of such objects to represent
    heterogeneous data.

    Attributes:

      flux : flux densities in the form of a 2D array, wavelength along
             X-axis, Y-values represent different times. These should
             usually have been continuum subtracted.

      ferr : uncertainties on the flux densities, 2D array matching flux

      wave : wavelengths, 2D array matching flux. These should be tied to
             a constant or near-enough constant zeropoint (e.g. heliocentric
             or better still, barycentric)

      time : times at centre of each spectrum

      expose : length of each spectrum

      ndiv : sub-division factors for accounting for finite exposures

      fwhm : FWHM of point spread function of spectra, in terms of pixels.
    """

    def __init__(self, flux, ferr, wave, time, expose, ndiv, fwhm):
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

        if len(flux.shape) != 2:
            raise DopplerError('Data.__init__: flux must be a 2D array')
        if not sameDims(flux,ferr):
            raise DopplerError('Data.__init__: flux and ferr are incompatible.')
        if not sameDims(flux,wave):
            raise DopplerError('Data.__init__: flux and wave are incompatible')
        if len(time.shape) != 1 or flux.shape[0] != len(time):
            raise DopplerError('Data.__init__: flux and time have' +
                               ' conflicting sizes')
        if len(expose.shape) != 1 or flux.shape[0] != len(expose):
            raise DopplerError('Data.__init__: flux and expose have' +
                               ' conflicting sizes')
        if len(ndiv.shape) != 1 or flux.shape[0] != len(ndiv):
            raise DopplerError('Data.__init__: flux and ndiv have' +
                               ' conflicting sizes')

        # Manipulate data types for efficiency savings when calling
        # the C++ interface routines.
        self.flux = flux if flux.dtype == np.float32 \
            else flux.astype(np.float32)
        self.ferr = ferr if ferr.dtype == np.float32 \
            else ferr.astype(np.float32)
        self.wave = wave if wave.dtype == np.float64 \
            else wave.astype(np.float64)
        self.time = time if time.dtype == np.float64 \
            else time.astype(np.float64)
        self.expose = expose if expose.dtype == np.float32 \
            else expose.astype(np.float32)
        self.ndiv = ndiv if ndiv.dtype == np.int32 \
            else ndiv.astype(np.int32)
        self.fwhm   = fwhm

    @classmethod
    def fromHDUl(cls, hdul):
        """
        Creates a Spectra from a list of 4 HDUs.
        """
        if len(hdul) < 4:
            raise DopplerError('Spectra.fromHDUl: minimum 4' +
                               ' HDUs are required.')

        flux   = hdul[0].data
        fwhm   = hdul[0].header['FWHM']
        ferr   = hdul[1].data
        wave   = hdul[2].data
        table  = hdul[3].data
        time   = table['time']
        expose = table['expose']
        ndiv   = table['ndiv']

        return cls(flux,ferr,wave,time,expose,ndiv,fwhm)

    def toHDUl(self):
        """
        Returns the Spectra as an equivalent list of astropy.io.fits HDUs (3
        image, 1 table, *not* an HDUList), suited to adding onto other hdus
        for eventual writing to a FITS file
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

        # make a table HDU
        c1 = fits.Column(name='time', format='D', array=self.time)
        c2 = fits.Column(name='expose', format='E', array=self.expose)
        c3 = fits.Column(name='ndiv', format='J', array=self.ndiv)
        hdul.append(fits.new_table(fits.ColDefs([c1,c2,c3]),head))

        return hdul

    def __repr__(self):
        return 'Spectra(flux=' + repr(self.flux) + \
            ', ferr=' + repr(self.ferr) + ', wave=' + repr(self.wave) + \
            ', time=' + repr(self.time) + ', expose=' + repr(self.expose) + \
            ', ndiv=' + repr(self.ndiv) + ', fwhm=' + repr(self.fwhm) + ')'

class Data(object):
    """
    This class represents all the data needed for Doppler tomogarphy.  It has
    the following attributes::

      head : an astropy.io.fits.Header object

      data : a list of Spectra objects.
    """

    def __init__(self, head, data):
        """
        Creates a Data object

        head : an astropy.io.fits.Header object. This is copied
               and some extra comments added to it internally.

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

        self.head = head

    @classmethod
    def rfits(cls, fname):
        """
        Reads in data from a fits file. The primary HDU's header is read along
        with data which should be 2D representing flux densities.  Three more
        HDUs are expected containing (1) uncertainties in the flux densities,
        (2) wavelengths, and (3) a table HDU containing central exposure times,
        exposure lengths and sub-division factors.
        """
        hdul = fits.open(fname)
        if len(hdul) < 5 or len(hdul) % 4 != 1:
            raise DopplerError('Data.rfits: ' + fname + 
                               ' did not have 4n+1 HDUs')
        head = hdul[0].header
        data = []
        for nhdu in xrange(1,len(hdul),6):
            data.append(Spectra.fromHDUl(hdul[nhdu:]))

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
    flux   = np.zeros(shape,dtype=np.float32)
    ferr   = 0.1*np.ones_like(flux)
    wave   = np.linspace(490.,510.,shape[1])
    time   = np.linspace(50000.,50000.1,shape[0])
    expose = 0.001*np.ones_like(time)
    ndiv   = np.ones_like(time,dtype=np.int)
    fwhm   = 2.2

    # manipulate the fluxes to be vaguely interesting
    # (gaussian sinusoidally varying in velocity)
    wave, times = np.meshgrid(wave, time)
    flux = np.exp(-((wave-500.*(1+300./3e5*np.sin(
                        2.*np.pi*(times-50000.)/0.1)))/1.5)**2/2.)

    # create the Spectra
    spectra = Spectra(flux,ferr,wave,time,expose,ndiv,fwhm)

    # create the Data
    data = Data(head,spectra)

    data.wfits('test.fits')

    d = Data.rfits('test.fits')
