"""
Defines the classes needed to represent Doppler maps.
"""

import collections
import numpy as np
from astropy.io import fits

from core import *

class Default (object):
    """
    Class defining the way in which the default image is computed.
    Main attribute is called 'option' which can have the following values:

      UNIFORM   : uniform default
      GAUSS2D   : Gaussian blurr default for 2D images
      GAUSS3D   : Gaussian blurr default for 3D images

    """
    UNIFORM  = 1
    GAUSS2D  = 2
    GAUSS3D  = 3

    def __init__(self, option, *args):
        """
        Creates a Default. 

        option = UNIFORM needs no other arguments.
        option = GAUSS2D need the FWHM blurr to use over Vx-Vy (in km/s)
        option = GAUSS3D need the FWHM blurr to use over Vx-Vy (in km/s) AND another FWHM to use
                 in the Vz direction, also km/s.

        See also Default.uniform, Default.gauss2D, Default.gauss3d
        """
        self.option = option
        if option == Default.GAUSS2D and len(args) == 1:
            self.fwhmxy = args[0]
        elif option == Default.GAUSS3D and len(args) == 2:
            self.fwhmxy = args[0]
            self.fwhmz  = args[1]
        elif option != Default.UNIFORM:
            raise DopplerError('Default.__init__: invalid option and/or wrong number of arguments.')

    @classmethod
    def uniform(cls):
        "Returns a uniform Default object"
        return cls(Default.UNIFORM)

    @classmethod
    def gauss2d(cls, fwhmxy):
        "Returns a gaussian Default object for a 2D image"
        return cls(Default.GAUSS2D, fwhmxy)

    @classmethod
    def gauss3d(cls, fwhmxy, fwhmz):
        "Returns a gaussian Default object for a 3D image"
        return cls(Default.GAUSS3D, fwhmxy, fwhmz)

    def __repr__(self):
        """
        Returns a string representation of the Default
        """
        rep = 'Default(option=' + repr(self.option)
        if self.option == Default.UNIFORM:
            return rep + ')'
        elif self.option == Default.GAUSS2D:
            return rep + ',fwhmxy=' + repr(self.fwhmxy) + ')'
        elif self.option == Default.GAUSS3D:
            return rep + ', fwhmxy=' + repr(self.fwhmxy) + ', fwhmz=' + repr(self.fwhmz) + ')'

class Image(object):
    """
    This class contains all the information needed to specify a single image,
    including the wavelength of the line or lines associated with the image,
    systemic velocities, scaling factors, velocity scales of the image,
    default parameters. 2D images have square pixels in velocity space VXY on
    a side. 3D images can be thought of as a series of 2D images spaced by
    VZ.

    The following attributes are set::

      data    : the image data array, 2D or 3D.

      wave    : array of associated wavelengths (will be an array even if
                only 1 value)

      gamma   : array of systemic velocities, one per wavelength

      default : a Default object defining how the default is calculated for 
                mem.

      vxy     : pixel size in Vx-Vy plane, km/s, square.

      scale   : scale factors to use if len(wave) > 1 (will still be defined
                but probably = None otherwise)

      vz      : km/s in vz direction if data.ndim == 3 (will still be defined
                but probably = None otherwise)

    """

    def __init__(self, data, vxy, wave, gamma, default, scale=None, vz=None):
        """
        Defines an Image. Arguments::

          data : the data array, either 2D or 3D.

          vxy : the pixel size in the X-Y plane, same in both X and Y, units
                km/s.

          wave : the wavelength or wavelengths associated with this Image. The
                 same image can represent multiple lines, in which case a set
                 of scale factors must be supplied as well. Can either be a
                 single float or an array.

          gamma : systemic velocity or velocities for each lines, km/s

          default : how to calculate the default image during mem iterations. This
                    should be a Default object.

          scale : if there are multiple lines modelled by this Image (e.g. the
                  Balmer series) then you must supply scaling factors to be
                  applied for each one as well.  scale must have the same
                  dimension as wave in this case.

          vz : if data is 3D then you must supply a z-velocity spacing in
               km/s.
        """
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim < 2 or self.data.ndim > 3:
            raise DopplerError('Image.__init__: data must be a 2D' +
                               ' or 3D numpy array')
        if self.data.ndim == 3 and vz is None:
            raise DopplerError('Image.__init__: vz must be defined for 3D data')

        self.vxy  = vxy
        self.vz   = vz

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
            raise DopplerError('Image.__init__: default option must match image dimension, e.g. GAUSS2D for 2D images')

        self.default = default

        # scale factors
        if isinstance(scale, np.ndarray):
            if scale.ndim > 1:
                raise DopplerError('Image.__init__: scale can at most' +
                                   ' be one dimensional')
            self.scale = scale

            if len(self.scale) != len(self.wave):
                raise DopplerError('Image.__init__: scale and wave must' +
                                   ' match in size')

        elif isinstance(scale, collections.Iterable):
            self.scale = np.array(scale)
            if len(self.scale) != len(self.wave):
                raise DopplerError('Image.__init__: scale and wave must' +
                                   ' match in size')

        elif len(self.wave) > 1:
            raise DopplerError('Image.__init__: scale must be an array' +
                               ' if wave is')
        else:
            self.scale = None

    def toHDU(self, next):
        """
        Returns the Image as an astropy.io.fits.ImageHDU. The map is held as
        the main array. All the rest of the information is stored in the
        header.

        Arguments::

          next : a number to append to the EXTNAME header extension names.
        """

        # create header which contains all but the actual data array
        head = fits.Header()
        head['VXY']  = (self.vxy, 'Vx-Vy pixel size, km/s')
        if self.data.ndim == 3:
            head['VZ']  = (self.vz, 'Vz pixel size, km/s')
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

        if self.default.option == Default.UNIFORM:
            head['DEFAULT'] = ('Uniform', 'Default option')
        elif self.default.option == Default.GAUSS2D:
            head['DEFAULT'] = ('Gaussian', 'Default option')
            head['FWHMXY']  = (self.default.fwhmxy, 'Vx-Vy blurring, km/s')
        elif self.default.option == Default.GAUSS3D:
            head['DEFAULT'] = ('Gaussian', 'Default option')
            head['FWHMXY']  = (self.default.fwhmxy, 'Vx-Vy blurring, km/s')
            head['FWHMZ']   = (self.default.fwhmz, 'Vz blurring, km/s')
        head['EXTNAME'] = 'Image' + str(next)

        # ok return with ImageHDU
        return fits.ImageHDU(self.data,head)

    @classmethod
    def fromHDU(cls, hdu):
        """
        Create an Image given an HDU of the correct nature
        """

        data = hdu.data
        head = hdu.header
        if 'VXY' not in head or 'NWAVE' not in head \
                or 'WAVE1' not in head or 'GAMMA1' not in head \
                or 'DEFAULT' not in head:
            raise DopplerError('Image.fromHDU: one or more of' +
                               ' VXY, NWAVE, WAVE1, GAMMA1, ' +
                               'DEFAULT not found in HDU header')

        vxy = head['VXY']
        if data.ndim == 3:
            vz = head['VZ']
        else:
            vz = None

        nwave = head['NWAVE']
        wave  = np.empty((nwave))
        gamma = np.empty((nwave))
        scale = np.empty((nwave)) if nwave > 1 else None
        for n in xrange(nwave):
            wave[n]  = head['WAVE' + str(n+1)]
            gamma[n] = head['GAMMA' + str(n+1)]
            if nwave > 1:
                scale[n] = head['SCALE' + str(n+1)]

        if head['DEFAULT'] == 'Uniform':
            default = Default.uniform()
        elif head['DEFAULT'] == 'Gaussian':
            if data.ndim == 2:
                if 'FWHMXY' not in head:
                    raise DopplerError('Image.fromHDU: could not find FWHMXY')
                default = Default.gauss2d(head['FWHMXY'])
            else:
                if 'FWHMXY' not in head or 'FWHMZ' not in head:
                    raise DopplerError('Image.fromHDU: could not find FWHMXY and/or FWHMZ')
                default = Default.gauss2d( head['FWHMXY'], head['FWHMZ'])

        return cls(data, vxy, wave, gamma, default, scale, vz)

    def __repr__(self):
        return 'Image(data=' + repr(self.data) + \
            ', vxy=' + repr(self.vxy) + ', wave=' + repr(self.wave) + \
            ', gamma=' + repr(self.gamma) + ', default=' + repr(self.default) + \
            ', scale=' + repr(self.scale) + ', vz=' + repr(self.vz) + ')'

class Map(object):
    """
    This class represents a complete Doppler image. Features include:
    (1) different maps for different lines, (2) the same map
    for different lines, (3) 3D maps.

    Attributes::

      head : an astropy.io.fits.Header object

      data : a list of Image objects.

      tzero  : zeropoint of ephemeris in the same units as the times of the data

      period : period of ephemeris in the same units as the times of the data

      quad   : quadratic term of ephemeris in the same units as the times of the data

      vfine  : km/s to use for the fine array used to project into before blurring.
               Should be a few times (5x at most) smaller than the km/s used for any
               image.

      sfac   : scale factor to use when computing data from the map. This is designed
               to allow the map images to take on "reasonable" values when matching
               a set of data. In the old F77 doppler code it was set to 0.0001 by
               default.
    """

    def __init__(self, head, data, tzero, period, quad, vfine, sfac=0.0001):
        """
        Creates a Map object

        head : an astropy.io.fits.Header object. A copy is taken as it
               is likely to be modified (comments added if keyword VERSION
               is not found)

        data : an Image or a list of Images

        tzero  : zeropoint of ephemeris in the same units as the times of the data

        period : period of ephemeris in the same units as the times of the data

        quad   : quadratic term of ephemeris in the same units as the times of the data

        vfine : km/s to use for the fine array used to project into before
                blurring.  Should be a few times (5x at most) smaller than
                the km/s used for any image.

        sfac : factor to multiply by when computing data corresponding to the
               map.
        """

        # some checks
        if not isinstance(head, fits.Header):
            raise DopplerError('Map.__init__: head' +
                               ' must be a fits.Header object')
        self.head = head.copy()
        # Here add comments on the nature of the data. The check
        # on the presence of VERSION is to avoid adding the comments
        # in when they are already there.
        if 'VERSION' not in self.head:
            self.head['VERSION'] = (VERSION, 'Software version number.')
            self.head.add_blank('.....................................')
            self.head.add_comment(
                'This is a map file for the Python Doppler imaging package trm.doppler.')
            self.head.add_comment(
                'The Doppler map format stores one or more images in a series of HDUs')
            self.head.add_comment(
                'following the (empty) primary HDU. The images are either 2 or 3D and')
            self.head.add_comment(
                'span (Vx,Vy) or (Vx,Vy,Vz) space. The images are square in the Vx-Vy')
            self.head.add_comment(
                'plane, but can have an arbitrary dimension along the Vz axis. Likewise')
            self.head.add_comment(
                'the pixels are square in Vx-Vy (size VXY), but can have a different')
            self.head.add_comment(
                'size along the Vz axis (VZ). The values VXY and VZ are stored in the')
            self.head.add_comment(
                'headers of each HDU. Each image can apply to one or more atomic lines.')
            self.head.add_comment(
                'Each line requires specification of a laboratory wavelength (WAVE) and')
            self.head.add_comment(
                'systemic velocity (GAMMA). If there is more than one line, then each')
            self.head.add_comment(
                'requires a scaling factor (SCALE). Again these are contained in the HDU.')
            self.head.add_comment(
                'Each line also requires information on how to construct a default image')
            self.head.add_comment(
                'contained in the parameters DEFAULT, FWHMXY and FWHMZ (3D only).')
            self.head.add_comment(
                'The primary HDU contains some parameters that apply to all images. These')
            self.head.add_comment(
                'specify an ephemeris (TZERO, PERIOD, QUAD), a pixel size (VFINE) to be')
            self.head.add_comment(
                'use for an intermediate finely-spaced array during projection and an')
            self.head.add_comment(
                'overall scale factor (SFAC) designed to allow image values matching a')
            self.head.add_comment(
                'given data set to have values of order unity.')

        try:
            for i, image in enumerate(data):
                if not isinstance(image, Image):
                    raise DopplerError('Map.__init__: element ' + str(i) +
                                       ' of map is not an Image.')

            self.data = data
        except TypeError, err:
            if not isinstance(data, Image):
                raise DopplerError('Map.__init__: data must be an' +
                                   ' Image or a list of Images')
            self.data = [data,]

        self.tzero  = tzero
        self.period = period
        self.quad   = quad
        self.vfine  = vfine
        self.sfac   = sfac

    @classmethod
    def rfits(cls, fname):
        """
        Reads in a Map from a fits file. The primary HDU's header is
        read followed by Images in the subsequent HDUs. The primary
        HDU header is expected to contain a few standard keywords
        which are stripped out.
        """
        hdul = fits.open(fname)
        if len(hdul) < 2:
            raise DopplerError('Map.rfits: ' + fname + ' had too few HDUs')
        head = hdul[0].header

        # Extract standard values that must be present
        tzero  = head['TZERO']
        period = head['PERIOD']
        quad   = head['QUAD']
        vfine  = head['VFINE']
        sfac   = head['SFAC']

        # Remove from the header
        del head['TZERO']
        del head['PERIOD']
        del head['QUAD']
        del head['VFINE']
        del head['SFAC']

        # Now the data
        data = []
        for hdu in hdul[1:]:
            data.append(Image.fromHDU(hdu))

        # OK, now make the map
        return cls(head, data, tzero, period, quad, vfine, sfac)

    def wfits(self, fname, clobber=True):
        """
        Writes a Map to a file
        """
        # copy the header so we can safely modify it
        head = self.head.copy()
        head['TZERO']  = (self.tzero, 'Zeropoint of ephemeris')
        head['PERIOD'] = (self.period, 'Period of ephemeris')
        head['QUAD']   = (self.quad, 'Quadratic coefficient of ephemeris')
        head['VFINE']  = (self.vfine, 'Fine array spacing, km/s')
        head['SFAC']   = (self.sfac, 'Global scaling factor')
        hdul  = [fits.PrimaryHDU(header=head),]
        for i, image in enumerate(self.data):
            hdul.append(image.toHDU(i+1))
        hdulist = fits.HDUList(hdul)
        hdulist.writeto(fname, clobber=clobber)

    def __repr__(self):
        return 'Map(head=' + repr(self.head) + \
            ', data=' + repr(self.data) + ', tzero=' + repr(self.tzero) + \
            ', period=' + repr(self.period) + ', quad=' + repr(self.quad) + \
            ', vfine=' + repr(self.vfine) + ', sfac=' + repr(self.sfac) + ')'

if __name__ == '__main__':

    # a header
    head = fits.Header()
    head['OBJECT']   = ('IP Peg', 'Object name')
    head['TELESCOP'] = ('William Herschel Telescope', 'Telescope name')

    # create some images
    ny, nx = 100, 100
    x      = np.linspace(-2000.,2000.,nx)
    y      = x.copy()
    vxy    = (x[-1]-x[0])/(nx-1)
    X, Y   = np.meshgrid(x, y)

    data1  = np.exp(-(((X-600.)/200.)**2+((Y-300.)/200.)**2)/2.)
    data1 += np.exp(-(((X+300.)/200.)**2+((Y+500.)/200.)**2)/2.)
    wave1  = np.array((486.2, 434.0))
    gamma1 = np.array((100., 100.))
    def1   = Default.gauss2d(200.)
    scale1 = np.array((1.0, 0.5))
    image1 = Image(data1, vxy, wave1, gamma1, def1, scale1)

    data2  = np.exp(-(((X+300.)/200.)**2+((Y+300.)/200.)**2)/2.)
    wave2  = 468.6
    gamma2 = 150.
    def2   = Default.gauss2d(200.)
    image2 = Image(data2, vxy, wave2, gamma2, def2)

    tzero   =  2550000.
    period  =  0.15
    quad    =  0.0
    vfine   =  10.

    print 'image2.default =',image2.default

    # create the Map
    map = Map(head,[image1,image2],tzero,period,quad,vfine)

    # write to fits
    map.wfits('test.fits')

    # read from fits
    m = Map.rfits('test.fits')

    # print to screen
    print m
