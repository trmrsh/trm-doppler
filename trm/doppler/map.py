"""
Defines the classes needed to represent Doppler maps.
"""

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
        if not isinstance(data, np.ndarray) or data.ndim < 2 or data.ndim > 3:
            raise DopplerError('Image.__init__: data must be a 2D' +
                               ' or 3D numpy array')
        if data.ndim == 3 and vz is None:
            raise DopplerError('Image.__init__: vz must be defined for 3D data')

        self.data = data if data.dtype == np.float32 \
            else data.astype(np.float32)
        self.vxy  = vxy
        self.vz   = vz

        # wavelengths
        if isinstance(wave, np.ndarray):
            if wave.ndim > 1:
                raise DopplerError('Image.__init__: wave can at most' +
                                   ' be one dimensional')
            self.wave = wave
        else:
            self.wave = np.array([float(wave),])

        # systemic velocities
        if isinstance(gamma, np.ndarray):
            if gamma.ndim > 1:
                raise DopplerError('Image.__init__: gamma can at most' +
                                   ' be one dimensional')
            self.gamma = gamma
        else:
            self.gamma = np.array([float(gamma),],dtype=np.float32)

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

        elif len(self.wave) > 1:
            raise DopplerError('Image.__init__: scale must be an array' +
                               ' if wave is')
        else:
            self.scale = None

    def toHDU(self):
        """
        Returns the Image as an astropy.io.fits.ImageHDU. The map is held as
        the main array. All the rest of the information is stored in the
        header.
        """

        # create header which contains all but the actual data array
        head = fits.Header()
        head['TYPE'] = ('doppler.Image', 'Python object type')
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

      period : period of ephemer in the same units as the times of the data

      vfine  : km/s to use for the fine array used to project into before blurring.
               Should be a few times (5x at most) smaller than the km/s used for any
               image.

      vpad   : padding used to extend the fine array beyond the range strictly defined
               by the images. This is a fudge to allow for blurring of the data.
    """

    def __init__(self, head, data, tzero, period, vfine, vpad):
        """
        Creates a Map object

        head : an astropy.io.fits.Header object

        data : an Image or a list of Images

        tzero  : zeropoint of ephemeris in the same units as the times of the data

        period : period of ephemer in the same units as the times of the data

        vfine : km/s to use for the fine array used to project into before
                blurring.  Should be a few times (5x at most) smaller than
                the km/s used for any image.

        vpad : padding used to extend the fine array beyond the range strictly
               defined by the images. This is a fudge to allow for blurring of
               the data.
        """

        # some checks
        if not isinstance(head, fits.Header):
            raise DopplerError('Map.__init__: head' +
                               ' must be a fits.Header object')
        self.head = head

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
        self.vfine  = vfine
        self.vpad   = vpad

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
        vfine  = head['VFINE']
        vpad   = head['VPAD']

        # Remove from the header
        del head['TZERO']
        del head['PERIOD']
        del head['VFINE']
        del head['VPAD']

        # Now the data
        data = []
        for hdu in hdul[1:]:
            data.append(Image.fromHDU(hdu))

        # OK, now make the map
        return cls(head, data, tzero, period, vfine, vpad)

    def wfits(self, fname, clobber=True):
        """
        Writes a Map to a file
        """
        # copy the header so we can safely modify it
        head = self.head.copy()
        head['TZERO']  = (self.tzero, 'Zeropoint of ephemeris')
        head['PERIOD'] = (self.period, 'Period of ephemeris')
        head['VFINE']  = (self.vfine, 'Fine array spacing, km/s')
        head['VPAD']   = (self.vpad, 'Fine array padding, km/s')
        hdul  = [fits.PrimaryHDU(header=head),]
        for image in self.data:
            hdul.append(image.toHDU())
        hdulist = fits.HDUList(hdul)
        hdulist.writeto(fname, clobber=clobber)

    def __repr__(self):
        return 'Map(head=' + repr(self.head) + \
            ', data=' + repr(self.data) + ', tzero=' + repr(self.tzero) + \
            ', period=' + repr(self.period) + ', vfine=' + repr(self.vfine) + \
            ', vpad=' + repr(self.vpad) + ')'

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
    vfine   =  10.
    vpad    =  200.

    print 'image2.default =',image2.default

    # create the Map
    map = Map(head,[image1,image2],tzero,period,vfine,vpad)
    
    # write to fits
    map.wfits('test.fits')

    # read from fits
    m = Map.rfits('test.fits')

    # print to screen
    print m
