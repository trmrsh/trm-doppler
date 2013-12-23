"""
Defines a grid for non-regularised Doppler imaging.
"""

import collections
import numpy as np
from astropy.io import fits

from core import *

class Grid(object):
    """
    This class represents a Doppler image through a square grid of
    gaussians. The idea is to carry out pure chi**2 minimisation with
    a relatively small number of gaussians to allow optimisation of
    non-linear parameters such as the orbital peripod.

    Attributes::

      head : an astropy.io.fits.Header object

      data : 2D square array of gaussian heights.

      tzero  : zeropoint of ephemeris in the same units as the times of the data

      period : period of ephemeris in the same units as the times of the data

      quad   : quadratic term of ephemeris in the same units as the times of
               the data

      vgrid  : km/s spacing of gaussians

      fwhm   : km/s FWHM of gaussians

      wave : wavelength(s) of line(s) to apply grid to

      gamma : systemic velocity(s) of each line

      scale : scale factors of each line

      sfac : scale factor to use when computing data from the map. This is
             designed to allow the map images to take on "reasonable" values
             when matching a set of data.

    """

    def __init__(self, head, data, tzero, period, quad, vgrid, fwhm,
                 wave, gamma, scale, sfac=0.0001):
        """
        Creates a Grid object

        head : an astropy.io.fits.Header object. A copy is taken as it
               is likely to be modified (comments added if keyword VERSION
               is not found)

        data : 2D squaan Image or aarray of heights of gaussians.

        tzero : zeropoint of ephemeris in the same units as the times of the
                data

        period : period of ephemeris in the same units as the times of the data

        quad   : quadratic term of ephemeris in the same units as the times
                 of the data

        vgrid : km/s to use for spacing of the gaussians

        fwhm  : km/s FWHM of gaussians

        wave : wavelength(s) of line(s) to apply grid to

        gamma : systemic velocity(s) of each line

        scale : scale factors of each line

        sfac : factor to multiply by when computing data corresponding to the
               map.
        """

        # some checks
        if not isinstance(head, fits.Header):
            raise DopplerError('Grid.__init__: head' +
                               ' must be a fits.Header object')
        self.head = head.copy()
        # Here add comments on the nature of the data. The check
        # on the presence of VERSION is to avoid adding the comments
        # in when they are already there.
        if 'VERSION' not in self.head:
            self.head['VERSION'] = (VERSION, 'Software version number.')
            self.head.add_blank('.....................................')
            self.head.add_comment(
                'This is a grid file for the Python Doppler imaging package trm.doppler.')
            self.head.add_comment(
                'The grid defines a set of 2D gaussians for modelling a Doppler image in')
            self.head.add_comment(
                'a manner that allows normal chi**2 minimisation. The gaussians are')
            self.head.add_comment(
                'spaced by VGRID (header parameter) with FWHM = FWHM. Multiple lines can')
            self.head.add_comment(
                'be modelled by the set of gaussians. Each line requires specification of')
            self.head.add_comment(
                'a laboratory wavelength (WAVE) and systemic velocity (GAMMA). If there')
            self.head.add_comment(
                'is more than one line, then each requires a scaling factor (SCALE).')
            self.head.add_comment(
                'Other header parameters specify an ephemeris (TZERO, PERIOD, QUAD),')
             self.head.add_comment(
                'an overall scale factor (SFAC) designed to allow')
            self.head.add_comment(
                'image values matching a given data set to have values of order unity.')

        self.data = np.asarray(data)
        if self.data !=2 or self.data.shape[0] != self.data.shape[1]:
            raise DopplerError('Grid.__init__: data is not 2D or not square')

        self.tzero  = tzero
        self.period = period
        self.quad   = quad
        self.vgrid  = vfine
        self.fwhm   = fwhm
        self.sfac   = sfac

        # wavelengths
        self.wave = np.asarray(wave)
        if self.wave.ndim == 0:
            self.wave = np.array([float(wave),])
        elif self.wave.ndim > 1:
            raise DopplerError('Image.__init__: wave can at most' +
                               ' be one dimensional')
        # systemic velocities
        self.gamma = np.asarray(gamma, dtype=np.float32)
        if self.gamma.ndim == 0:
            self.gamma = np.array([float(gamma),],dtype=np.float32)
        elif self.gamma.ndim > 1:
            raise DopplerError('Image.__init__: gamma can at most' +
                               ' be one dimensional')

        if len(self.gamma) != len(self.wave):
            raise DopplerError('Image.__init__: gamma and wave must' +
                               ' match in size')

        # default
        if not isinstance(default, Default):
            raise DopplerError('Image.__init__: default must be a Default object')

        if (default.option == Default.GAUSS2D and data.ndim == 3) or \
                (default.option == Default.GAUSS3D and data.ndim == 2):
            raise DopplerError('Image.__init__: default option must match'
                               ' image dimension, e.g. GAUSS2D for 2D images')

        self.default = default

        self.group = group
        if self.group < 0:
            raise DopplerError('Image.__init__: group must be an integer >= 0')

        self.pgroup = pgroup
        if self.pgroup < 0:
            raise DopplerError('Image.__init__: pgroup must be an integer >= 0')

        # scale factors
        if isinstance(scale, np.ndarray):
            if scale.ndim > 1:
                raise DopplerError('Grid.__init__: scale can at most' +
                                   ' be one dimensional')
            self.scale = scale

            if len(self.scale) != len(self.wave):
                raise DopplerError('Grid.__init__: scale and wave must' +
                                   ' match in size')

        elif isinstance(scale, collections.Iterable):
            self.scale = np.array(scale)
            if len(self.scale) != len(self.wave):
                raise DopplerError('Grid.__init__: scale and wave must' +
                                   ' match in size')

        elif len(self.wave) > 1:
            raise DopplerError('Grid.__init__: scale must be an array' +
                               ' if wave is')
        else:
            self.scale = None

    @classmethod
    def rfits(cls, fname):
        """
        Reads in a Grid from a suitable FITS file.
        """
        hdul = fits.open(fname)
        head = hdul[0].header

        # Extract standard values that must be present
        tzero  = head['TZERO']
        period = head['PERIOD']
        quad   = head['QUAD']
        vgrid  = head['VGRID']
        fwhm   = head['FWHM']
        sfac   = head['SFAC']

        # Remove from the header
        del head['TZERO']
        del head['PERIOD']
        del head['QUAD']
        del head['VGRID']
        del head['FWHM']
        del head['SFAC']

        nwave  = head['NWAVE']
        wave   = np.empty((nwave))
        gamma  = np.empty((nwave))
        scale  = np.empty((nwave)) if nwave > 1 else None
        for n in xrange(nwave):
            wave[n]  = head['WAVE' + str(n+1)]
            del head['WAVE' + str(n+1)]
            gamma[n] = head['GAMMA' + str(n+1)]
            del head['GAMMA' + str(n+1)]
            if nwave > 1:
                scale[n] = head['SCALE' + str(n+1)]
                del head['SCALE' + str(n+1)]

        # Now the data
        data = hdul[0].data

        # OK, now make the map
        return cls(head, data, tzero, period, quad, vgrid, fwhm,
                   wave, gamma, scale, sfac)

    def wfits(self, fname, clobber=True):
        """
        Writes a Grid to a file
        """
        # copy the header so we can safely modify it
        head = self.head.copy()
        head['TZERO']  = (self.tzero, 'Zeropoint of ephemeris')
        head['PERIOD'] = (self.period, 'Period of ephemeris')
        head['QUAD']   = (self.quad, 'Quadratic coefficient of ephemeris')
        head['VGRID']  = (self.vgrid, 'Gaussian spacing, km/s')
        head['FWHM']   = (self.fwhm, 'FWHM of gaussians, km/s')
        head['NWAVE']  = (len(self.wave), 'Number of wavelengths')
        if len(self.wave) > 1:
            n = 1
            for w, g, s in zip(self.wave,self.gamma,self.scale):
                head['WAVE'  + str(n)] = (w, 'Central wavelength')
                head['GAMMA' + str(n)] = (g, 'Systemic velocity, km/s')
                head['SCALE' + str(n)] = (s, 'Scaling factor')
                n += 1
        else:
            head['WAVE1']  = (self.wave[0], 'Central wavelength')
            head['GAMMA1'] = (self.gamma[0], 'Systemic velocity, km/s')
        head['SFAC']   = (self.sfac, 'Global scaling factor')

        hdul  = [fits.PrimaryHDU(header=head, data=self.data),]
        hdulist = fits.HDUList(hdul)
        hdulist.writeto(fname, clobber=clobber)

    def __repr__(self):
        return \
            'Grid(head=' + repr(self.head) + ', data=' + repr(self.data) + \
            ', tzero=' + repr(self.tzero) + ', period=' + repr(self.period) + \
            ', quad=' + repr(self.quad) + ', vgrid=' + repr(self.vgrid) + \
            ', fwhm=' + repr(self.fwhm) + ', wave=' + repr(self.wave) + \
            ', gamma=' + repr(self.gamma) + ', wave=' + repr(self.wave) + \
            ', scale=' + repr(self.scale) + ', sfac=' + repr(self.sfac) + ')'

if __name__ == '__main__':

    # Generates a map, writes it to disc, reads it back, prints it
    head = fits.Header()
    head['OBJECT']   = ('IP Peg', 'Object name')
    head['TELESCOP'] = ('William Herschel Telescope', 'Telescope name')

    # create some images
    data = np.empty((10,10))
    data = np.random.normal(data)

    wave  = np.array((486.2, 434.0))
    gamma = np.array((100., 100.))
    scale = np.array((1.0, 0.5))

    tzero   =  2550000.
    period  =  0.15
    quad    =  0.0
    vgrid   =  300.
    fwhm    =  300.

    # create the Grid
    grid = Grid(head,data,tzero,period,quad,vgrid,fwhm,wave.gamma,scale)

    # write to fits
    grid.wfits('test.fits')

    # read from fits
    g = Grid.rfits('test.fits')

    # print to screen
    print g
