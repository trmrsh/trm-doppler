"""
Defines the classes needed to represent the multiple spectra needed for
Doppler imaging in a FITS-compatible manner.
"""

import numpy as np
from astropy.io import fits

from .core import *

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
             X-axis, Y-values represent different times. These should usually
             have been continuum subtracted.

      ferr : uncertainties on the flux densities, 2D array matching flux

      wave : wavelengths, 2D array matching flux. These should be tied to a
             constant or near-enough constant zeropoint (e.g. heliocentric or
             better still, barycentric). The wavelengths should be
             monotonically increasing and vary smoothly with index. If there
             are jumps, you should split the dataset into multiple. Invert the
             order (easy in Python) if your wavelength scale decreases with
             index number, e.g. wave = wave[:,::-1], flux = flux[:,::-1] etc.
             Finally the wavelengths for each spectrum can vary, hence the 2D
             array, but it is assumed that they are similar. The expectation
             is that the wavelengths come from a sequence of spectra taken
             over a few nights with the same setup. They may drift by a few
             pixels, but should not differ radically from each other. If they
             do, you should be splitting into multiple datasets.

      time : times at the middle of each exposure. Any units as long as they
             match the ephemeris specified in the associated Map. Typically
             BTDB in days.  No order required. The number of times should
             match the first (Y) dimension of flux.

      expose : length of each spectrum. Same units as the times. These are
               always needed but only used if the nsub factors (see next) are
               > 1. Same size as 'time'


      nsub : sub-division factors for accounting for finite exposures. The
             projections are computed at a series of phases equally spaced
             from the start to the end of the exposure and trapezoidally
             averaged. Same size as 'time'.

      fwhm : FWHM of point spread function of spectra, km/s. If you think this
             varies significantly across your dataset, then you might want to
             split the dataset into chunks. Specifying in km/s rather than
             pixels is necessary for fast computation of the image to data
             transform.
    """

    def __init__(self, flux, ferr, wave, time, expose, nsub, fwhm):
        """
        Creates a Spectra object.
        """

        # checks
        self.flux = np.asarray(flux, dtype=np.float32)
        if len(flux.shape) != 2:
            raise DopplerError('Data.__init__: flux must be a 2D array')

        self.ferr = np.asarray(ferr, dtype=np.float32)
        if not sameDims(flux,ferr):
            raise DopplerError('Data.__init__: flux and ferr are incompatible.')

        self.wave = np.asarray(wave, dtype=np.float64)
        if not sameDims(flux,wave):
            raise DopplerError('Data.__init__: flux and wave are incompatible.')

        # check wavelengths increase in X
        diff = self.wave[:,1:]-self.wave[:,:-1]
        if not (diff > 0).all():
            raise DopplerError('Data.__init__: wavelengths must increase for all X')

        self.time = np.asarray(time, dtype=np.float64)
        if len(time.shape) != 1 or flux.shape[0] != len(time):
            raise DopplerError('Data.__init__: flux and time have' +
                               ' conflicting sizes')

        self.expose = np.asarray(expose, dtype=np.float32)
        if len(expose.shape) != 1 or flux.shape[0] != len(expose):
            raise DopplerError('Data.__init__: flux and expose have' +
                               ' conflicting sizes')

        self.nsub = np.asarray(nsub, dtype=np.int32)
        if len(nsub.shape) != 1 or flux.shape[0] != len(nsub):
            raise DopplerError('Data.__init__: flux and nsub have' +
                               ' conflicting sizes')
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
        nsub   = table['nsub']

        return cls(flux,ferr,wave,time,expose,nsub,fwhm)

    def toHDUl(self, n):
        """
        Returns the Spectra as an equivalent list of astropy.io.fits HDUs (3
        image, 1 table, *not* an HDUList), suited to adding onto other hdus
        for eventual writing to a FITS file

        Arguments::

          n : a number to append to the EXTNAME header extension names.
        """
        head = fits.Header()
        head['TYPE']    = 'Fluxes'
        head['EXTNAME'] = 'Flux' + str(n)
        head['FWHM']    = (self.fwhm, 'FWHM in km/s')
        hdul = [fits.ImageHDU(self.flux,head),]

        head = fits.Header()
        head['TYPE'] = 'Flux errors'
        head['EXTNAME'] = 'Ferr' + str(n)
        hdul.append(fits.ImageHDU(self.ferr,head))

        head = fits.Header()
        head['TYPE'] = 'Wavelengths'
        head['EXTNAME'] = 'Wave' + str(n)
        hdul.append(fits.ImageHDU(self.wave,head))

        head = fits.Header()
        head['EXTNAME'] = 'Time' + str(n)
        head['TYPE'] = 'Times'

        # make a table HDU
        c1 = fits.Column(name='time', format='D', array=self.time)
        c2 = fits.Column(name='expose', format='E', array=self.expose)
        c3 = fits.Column(name='nsub', format='J', array=self.nsub)
        hdul.append(fits.BinTableHDU.from_columns([c1,c2,c3],head))

        return hdul

    def __repr__(self):
        return 'Spectra(flux=' + repr(self.flux) + \
            ', ferr=' + repr(self.ferr) + ', wave=' + repr(self.wave) + \
            ', time=' + repr(self.time) + ', expose=' + repr(self.expose) + \
            ', nsub=' + repr(self.nsub) + ', fwhm=' + repr(self.fwhm) + ')'

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

        self.head = head.copy()
        # Here add comments on the nature of the data. The check
        # on the presence of VERSION is to avoid adding the comments
        # in when they are already there.
        if 'VERSION' not in self.head:
            self.head['VERSION'] = (VERSION, 'Software version number.')
            self.head.add_blank('.....................................')
            self.head.add_comment(
                'This is a data file for the Python Doppler imaging package trm.doppler.')
            self.head.add_comment(
                'The Doppler data format stores one or more datasets using a set of four')
            self.head.add_comment(
                'HDUs per dataset following the (empty) primary HDU. The 4 HDUs contain')
            self.head.add_comment(
                '(i) the fluxes, (ii) the flux uncertainties, (iii) the wavelengths,')
            self.head.add_comment(
                'and (iv) the times at the centres of the exposures, the exposure')
            self.head.add_comment(
                'durations and sub-division factors used to model finite duration')
            self.head.add_comment(
                'exposures. The last three are contained in a table HDU. The units')
            self.head.add_comment(
                'to be used are arbitrary as long as they are consistent in the case')
            self.head.add_comment(
                'of the wavelengths and times with the units used in the Doppler image')
            self.head.add_comment(
                'file. The only piece of numerical header information is the FWHM')
            self.head.add_comment(
                'resolution. This is specified per dataset and stored in the headers of')
            self.head.add_comment(
                'the first HDU of each dataset, the one which contains the fluxes.')

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
        for nhdu in xrange(1,len(hdul),4):
            data.append(Spectra.fromHDUl(hdul[nhdu:]))

        return cls(head, data)

    def wfits(self, fname, clobber=True):
        """
        Writes a Data to an hdu list
        """
        hdul  = [fits.PrimaryHDU(header=self.head),]
        for i, spectra in enumerate(self.data):
            hdul += spectra.toHDUl(i+1)
        hdulist = fits.HDUList(hdul)
        hdulist.writeto(fname, clobber=clobber)

    def __repr__(self):
        return 'Data(head=' + repr(self.head) + \
            ', data=' + repr(self.data) + ')'

    @property
    def size(self):
        """
        Returns total number of data points
        """
        n = 0
        for spectra in self.data:
            n += spectra.flux.size
        return n

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
    nsub   = np.ones_like(time,dtype=np.int)
    fwhm   = 2.2

    # manipulate the fluxes to be vaguely interesting
    # (gaussian sinusoidally varying in velocity)
    wave, times = np.meshgrid(wave, time)
    flux = np.exp(-((wave-500.*(1+300./3e5*np.sin(
                        2.*np.pi*(times-50000.)/0.1)))/1.5)**2/2.)

    # create the Spectra
    spectra = Spectra(flux,ferr,wave,time,expose,nsub,fwhm)

    # create the Data
    data = Data(head,spectra)

    data.wfits('test.fits')

    d = Data.rfits('test.fits')
