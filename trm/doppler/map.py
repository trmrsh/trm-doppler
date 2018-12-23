"""
Defines the classes needed to represent Doppler maps.
"""

import collections
import numpy as np
from astropy.io import fits
import copy

from .core import *

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

    DNAMES = {UNIFORM : 'UNIFORM',
              GAUSS2D : 'GAUSS2D',
              GAUSS3D : 'GAUSS3D'}

    def __init__(self, option, bias, *args):
        """
        Creates a Default.  option contains the default option which
        can be one of UNIFORM, GAUSS2D or GAUSS3D. Each these takes
        a different set of arguments.

        For UNIFORM::

         bias : a factor to pull the default level. Usually it should be
                 set = 1, but values < 1 may be useful. It must always be
                 > 0. A typical use is to support negative regions by having
                 one image of type PUNIT (positive non-modulated), and another
                 of type NUNIT (negative etc). A bias of 1 on the PUNIT image
                 and 0.9 on the negative will tilt the balance to wards the
                 positive overall.

        For GAUSS2D::

          bias : as above

          fwhmxy : FWHM blurr in Vx-Vy-plane

        For GAUSS3D::

          bias : as above

          fwhmxy : as above

          fwhmz : FWHM blurr in Vz

        See also Default.uniform, Default.gauss2D, Default.gauss3d
        """
        self.option = option
        self.bias   = bias
        if option == Default.GAUSS2D and len(args) == 1:
            self.fwhmxy = args[0]
        elif option == Default.GAUSS3D and len(args) == 2:
            self.fwhmxy = args[0]
            self.fwhmz  = args[1]
        elif option != Default.UNIFORM:
            raise DopplerError('Default.__init__: invalid option'
                               ' and/or wrong number of arguments.')

    @classmethod
    def uniform(cls, bias):
        "Returns a uniform Default object"
        return cls(Default.UNIFORM, bias)

    @classmethod
    def gauss2d(cls, bias, fwhmxy):
        "Returns a gaussian Default object for a 2D image"
        return cls(Default.GAUSS2D, bias, fwhmxy)

    @classmethod
    def gauss3d(cls, bias, fwhmxy, fwhmz):
        "Returns a gaussian Default object for a 3D image"
        return cls(Default.GAUSS3D, bias, fwhmxy, fwhmz)

    def __repr__(self):
        """
        Returns a string representation of the Default
        """
        rep = 'Default(option=' + repr(Default.DNAMES[self.option]) + \
              ', bias=' + repr(self.bias)
        if self.option == Default.UNIFORM:
            return rep + ')'
        elif self.option == Default.GAUSS2D:
            return rep + ', fwhmxy=' + repr(self.fwhmxy) + ')'
        elif self.option == Default.GAUSS3D:
            return rep + ', fwhmxy=' + repr(self.fwhmxy) + \
                                     ', fwhmz=' + repr(self.fwhmz) + ')'

# Image types. P = positive, N = negative
# UNIT means    x 1 at all phases
# SINE means    x sin(2*pi*phase)
# COSINE means  x cos(2*pi*phase)
# SINE2 means   x sin(2*2*pi*phase)
# COSINE2 means x cos(2*2*pi*phase)
#
# Together these allow negative region in images
# as well as flux modulation ('modmap')
PUNIT    =  1
NUNIT    =  2
PSINE    =  3
NSINE    =  4
PCOSINE  =  5
NCOSINE  =  6
PSINE2   =  7
NSINE2   =  8
PCOSINE2 =  9
NCOSINE2 = 10

ITNAMES = {
    PUNIT    : 'PUNIT',
    NUNIT    : 'NUNIT',
    PSINE    : 'PSINE',
    NSINE    : 'NSINE',
    PCOSINE  : 'PCOSINE',
    NCOSINE  : 'NCOSINE',
    PSINE2   : 'PSINE2',
    NSINE2   : 'NSINE2',
    PCOSINE2 : 'PCOSINE2',
    NCOSINE2 : 'NCOSINE2'
}

# inverted version
RITNAMES = {v: k for k, v in ITNAMES.items()}

class Image(object):
    """This class contains all the information needed to specify a single image,
    including the wavelength of the line or lines associated with the image,
    systemic velocities, scaling factors, velocity scales of the image,
    default parameters. 2D images have square pixels in velocity space VXY on
    a side. 3D images can be thought of as a series of 2D images spaced by
    VZ.

    The following attributes are set::

      data : float32 ndarray 
         the image data array, 2D or 3D.

      itype : int 
         Image type. Possible values are defined: PUNIT, NUNIT, PSINE, NSINE,
         PCOSINE, NCOSINE, PSINE2, NSINE2, PCOSINE2, NCOSINE2.  Control
         modulation and sign of image. P = +, N = -. UNIT = no modulation
         (i.e. normal doppler tom), sine, cosine etc allow modulation as sine
         / cosine of phase. 2 means at twice orbital frequency.

      vxy : float
         pixel size in Vx-Vy plane, km/s, square.

      wave : float64 1D array
         wavelengths associated with the image

      gamma : 1D array
         systemic velocities, one per wavelength, [km/s]

      default : Default
         object defining how the default is calculated for MEM

      scale : array or None
         scale factors to use if len(wave) > 1 (could be None
         otherwise)

      vz : float
         km/s in vz direction if data.ndim == 3 (will still be defined
         but could be None otherwise)

      group : integer
         for computing optimum scale factors one needs to group images.
         e.g. pairs of matching PUNIT / NUNIT, PSINE/ NSINE images need to be
         scaled by the same factor. Use the group parameter (sequential
         integers) to define such groups. If =0, the image will be
         assumed to be in a group on its own.

      pgroup : integer
         similar to group, this parameters allows you to link N-/P- pairs
         together for plots. The idea is that only the difference between such
         images is of any interest, and this allows you to say which goes with
         which. Unlike group, this should only be used for opposite pairs.

      wgshdu : bool
         If True, the wave, gamma and scale value(s) will be stored on disk as
         a table HDU following the image HDU containing the data. Else they
         will be stored as WAVE1, WAVE2, ..., GAMMA1, GAMMA2, ..., SCALE1,
         SCALE2 etc in the header. For > 999 such values, table HDU storage
         is automatic.

    """

    def __init__(self, data, itype, vxy, wave, gamma, default, scale=None,
                 vz=None, group=0, pgroup=0, wgshdu=False):
        """Defines an Image. Arguments::

           data : ndarray 
              the data array, either 2D or 3D.

           itype : int
              Image type. Possible values have defined names: PUNIT, NUNIT,
              PSINE, NSINE, PCOSINE, NCOSINE, PSINE2, NSINE2, PCOSINE2,
              NCOSINE2.  Control modulation and sign of image. P = +, N =
              -. UNIT = no modulation (i.e. normal doppler tom), sine, cosine
              etc allow modulation as sine / cosine of phase. 2 means at twice
              orbital frequency.

           vxy : float
              the pixel size in the X-Y plane, same in both X and Y, [km/s].

           wave : float | array
               the wavelength(s) associated with this Image. The same image
               can represent multiple lines, in which case a set of scale
               factors must be supplied as well. Can either be a single float
               or an array. Units must match whatever units are used for
               wavelengths in the data files.

           gamma : float | array
               systemic velocity(ies) for the line(s) [km/s]

           default : Default
               how to calculate the default image during MEM iterations.
               This should be a Default object.

           scale : array | None
               if there are multiple lines modelled by this Image (e.g. the
               Balmer series) then you must supply scaling factors to be
               applied for each one as well.  scale must have the same
               dimension as wave in this case. If there is only one line, it
               will set equal to 1 (an array)

           vz : float | None
               if data is 3D then you must supply a z-velocity spacing [km/s].

           group : int
               for computing optimum scale factors one needs to group
               images. e.g.  pairs of matching PUNIT / NUNIT, PSINE/ NSINE
               images need to be scaled by the same factor. Use the group
               parameter (sequential positive integers) to define such
               groups. If group=0, the image will be assumed to be in a group
               on its own.

           pgroup : int
               similar to group, this parameters allows you to link N-/P-
               pairs together for plots. The idea is that only the difference
               between such images is of any interest, and this allows you to
               say which goes with which. Unlike group, this should only be
               used for opposite pairs.

           wgshdu : bool
               If True, the wave, gamma and scale values will be stored in
               a binary HDU following the image, otherwise they will be 
               stored in the header connected to the image.
        """
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim < 2 or self.data.ndim > 3:
            raise DopplerError('Image.__init__: data must be a 2D' +
                               ' or 3D numpy array')
        if self.data.ndim == 3 and vz is None:
            raise DopplerError('Image.__init__: vz must be defined for 3D data')

        self.itype = itype
        self.vxy   = vxy
        self.vz    = vz

        # wavelengths
        self.wave = np.asarray(wave)
        if self.wave.ndim == 0:
            self.wave = np.array([float(wave),])
        elif self.wave.ndim > 1:
            raise DopplerError(
                'Image.__init__: wave can at most be one dimensional'
            )

        # systemic velocities
        self.gamma = np.asarray(gamma, dtype=np.float32)
        if self.gamma.ndim == 0:
            self.gamma = np.array([float(gamma),],dtype=np.float32)
        elif self.gamma.ndim > 1:
            raise DopplerError(
                'Image.__init__: gamma can at most be one dimensional'
            )

        if len(self.gamma) != len(self.wave):
            raise DopplerError(
                'Image.__init__: gamma and wave must match in size'
            )

        # default
        if not isinstance(default, Default):
            raise DopplerError('Image.__init__: default must be a Default object')

        if (default.option == Default.GAUSS2D and data.ndim == 3) or \
                (default.option == Default.GAUSS3D and data.ndim == 2):
            raise DopplerError(
                'Image.__init__: default option must match'
                ' image dimension, e.g. GAUSS2D for 2D images'
            )

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
                raise DopplerError(
                    'Image.__init__: scale can at most' +
                    ' be one dimensional'
                )
            self.scale = scale

            if len(self.scale) != len(self.wave):
                raise DopplerError(
                    'Image.__init__: scale and wave must' +
                    ' match in size'
                )

        elif isinstance(scale, collections.Iterable):
            self.scale = np.array(scale)
            if len(self.scale) != len(self.wave):
                raise DopplerError(
                    'Image.__init__: scale and wave must' +
                    ' match in size'
                )

        elif len(self.wave) > 1:
            raise DopplerError(
                'Image.__init__: scale must be an array' +
                ' if wave is'
            )
        else:
            self.scale = np.array([1.,])

        self.wgshdu = wgshdu

    def toHDU(self, next):
        """
        Returns the Image as an astropy.io.fits.ImageHDU or as an
        astropy.io.fits.ImageHDU and an astropy.io.fits.BinTableHDU.
        The latter is when the wave / gamma / scale values are stored
        as a binary table. In each case these are returned as a tuple
        so they can be added to a list of HDUs.

        Arguments::

            next : int
               a number to append to the EXTNAME header extension names.
        """

        # create header
        head = fits.Header()

        head['ITYPE']  = (ITNAMES[self.itype], 'Image type')

        head['VXY']  = (self.vxy, 'Vx-Vy pixel size, km/s')
        if self.data.ndim == 3:
            head['VZ']  = (self.vz, 'Vz pixel size, km/s')
        head['NWAVE']  = (len(self.wave), 'Number of wavelengths')

        if not self.wgshdu and len(self.wave) < 1000:
            # write wave, gamma, scale values to the header
            n = 1
            for w, g, s in zip(self.wave,self.gamma,self.scale):
                head['WAVE' + str(n)] = (w, 'Central wavelength')
                head['GAMMA' + str(n)] = (g, 'Systemic velocity, km/s')
                head['SCALE' + str(n)] = (s, 'Scaling factor')
                n += 1
            inhead = True

        else:
            # write wave, gamma, scale values to table HDU
            c1 = fits.Column(name='WAVE', array=self.wave, format='D')
            c2 = fits.Column(name='GAMMA', array=self.gamma, format='E')
            c3 = fits.Column(name='SCALE', array=self.scale, format='E')
            thead = fits.Header()
            thead['EXTNAME'] = 'Table' + str(next)
            thdu = fits.BinTableHDU.from_columns([c1, c2, c3], thead)
            inhead = False

        head['DEFAULT'] = (
            Default.DNAMES[self.default.option],'Default option')

        head['BIAS'] = (self.default.bias, 'Bias to steer image')

        if self.default.option == Default.GAUSS2D:
            head['FWHMXY']  = (self.default.fwhmxy, 'Vx-Vy blurring, km/s')
        elif self.default.option == Default.GAUSS3D:
            head['FWHMXY']  = (self.default.fwhmxy, 'Vx-Vy blurring, km/s')
            head['FWHMZ']   = (self.default.fwhmz, 'Vz blurring, km/s')

        head['GROUP']   = (self.group, 'Image group number (0=no group)')
        head['PGROUP']  = (self.pgroup, 'Plot group number (0=no group)')
        head['EXTNAME'] = 'Image' + str(next)

        # define WCS
        head['WCSNAME'] = 'Velocity'
        head['CRPIX1'] = (self.data.shape[-1]+1)/2.
        head['CRPIX2'] = (self.data.shape[-2]+1)/2.
        if self.data.ndim == 3:
            head['CRPIX3'] = (self.data.shape[-3]+1)/2.

        head['CDELT1'] = self.vxy
        head['CDELT2'] = self.vxy
        if self.data.ndim == 3:
            head['CDELT3'] = self.vz

        head['CTYPE1'] = 'Vx'
        head['CUNIT2'] = 'Vy'
        if self.data.ndim == 3:
            head['CUNIT3'] = 'Vz'

        head['CRVAL1'] = 0.
        head['CRVAL2'] = 0.
        if self.data.ndim == 3:
            head['CRVAL3'] = 0.

        head['CUNIT1'] = 'km/s'
        head['CUNIT2'] = 'km/s'
        if self.data.ndim == 3:
            head['CUNIT3'] = 'km/s'
        ihdu = fits.ImageHDU(self.data,head)

        if inhead:
            # ok return with ImageHDU
            return (ihdu,)
        else:
            # ok return with ImageHDU and BinTableHDU
            return (ihdu,thdu)

    @classmethod
    def fromHDU(cls, hdui, hdut=None):
        """
        Create an Image given an HDU or HDUs of the correct nature.
        hdui must be an image HDU containing the data for the image
        along with some associated header items. It may contain
        headers with names like WAVE1, GAMMA1, SCALE1, WAVE2, GAMMA2,
        etc, but if it doesn't, these need to be supplied as a table
        HDU (hdut) with columns called WAVE, GAMMA and SCALE.
        """

        data = hdui.data
        head = hdui.header
        if 'VXY' not in head or 'NWAVE' not in head \
           or 'DEFAULT' not in head or 'BIAS' not in head \
           or 'ITYPE' not in head:
            raise DopplerError(
                'Image.fromHDU: one or more of VXY, NWAVE, DEFAULT, BIAS,'
                ' ITYPE not found in HDU header'
            )

        if ('WAVE1' not in head or 'GAMMA1' not in head) and \
           (hdut is None or not isinstance(hdut,fits.BinTableHDU)):
            raise DopplerError(
                'Image.fromHDU: WAVE1 and/or GAMMA1 not found and no valid BinTableHDU'
            )

        # effectively reverse usual dictionary look up
        itype = RITNAMES[head['ITYPE']]

        vxy   = head['VXY']
        if data.ndim == 3:
            vz = head['VZ']
        else:
            vz = None

        if 'GAMMA1' in head:
            # read from image header
            nwave = head['NWAVE']
            wave  = np.empty((nwave))
            gamma = np.empty((nwave))
            scale = np.empty((nwave))
            for n in range(nwave):
                wave[n]  = head['WAVE' + str(n+1)]
                gamma[n] = head['GAMMA' + str(n+1)]
                if nwave == 1:
                    scale[n] = 1.0
                else:
                    scale[n] = head['SCALE' + str(n+1)]
            wgshdu = False

        else:

            # read from table HDU
            table = hdut.data
            wave = table['WAVE']
            gamma = table['GAMMA']
            scale = table['SCALE']
            wgshdu = True

        if head['DEFAULT'] == 'UNIFORM':
            default = Default.uniform(head['BIAS'])
        elif head['DEFAULT'] == 'GAUSS2D':
            if 'FWHMXY' not in head:
                raise DopplerError('Image.fromHDU: could not find FWHMXY')
            default = Default.gauss2d(head['BIAS'], head['FWHMXY'])
        elif head['DEFAULT'] == 'GAUSS3D':
            if 'FWHMXY' not in head or 'FWHMZ' not in head:
                raise DopplerError('Image.fromHDU: could not find '
                                   'FWHMXY and/or FWHMZ')
            default = Default.gauss3d(head['BIAS'], head['FWHMXY'],
                                      head['FWHMZ'])
        else:
            raise DopplerError(
                'Image.fromHDU: unrecognised default option = ' + head['DEFAULT']
            )

        group  = head['GROUP']
        pgroup = head['PGROUP']

        return cls(
            data, itype, vxy, wave, gamma, default, scale, vz,
            group, pgroup, wgshdu
        )

    def isPositive(self):
        """
        Returns true if all pixels in Image are positive
        """
        return (self.data > 0.).all()

    def log(self):
        """Takes natural log of the Image"""
        self.data = np.log(self.data)

    def sumf(self):
        """Returns sum of image values"""
        return self.data.sum()

    def __itruediv__(self, other):
        """Divides self by other, modifying in place and
        returning self
        """
        if isinstance(other, Image):
            self.data /= other.data
        else:
            self.data /= other
        return self

    def __imul__(self, other):
        """Multiplies self by other, modifying in place and
        returning self
        """
        if isinstance(other, Image):
            self.data *= other.data
        else:
            self.data *= other
        return self

    def __isub__(self, other):
        """Subtracts other from self, modifying in place and
        returning self
        """
        if isinstance(other, Image):
            self.data -= other.data
        else:
            self.data -= other
        return self

    def __iadd__(self, other):
        """Adds other to self, modifying in place and
        returning self
        """
        if isinstance(other, Image):
            self.data += other.data
        else:
            self.data += other
        return self

    def __sub__(self, other):
        """Subtracts other from self returning result as new Image
        """
        nimage = copy.deepcopy(self)
        nimage -= other
        return nimage

    def __repr__(self):
        return \
            'Image(data=' + repr(self.data) + ', itype=' + \
            repr(ITNAMES[self.itype]) + ', vxy=' + repr(self.vxy) + \
            ', wave=' + repr(self.wave) + ', gamma=' + \
            repr(self.gamma) + ', default=' + repr(self.default) + \
            ', scale=' + repr(self.scale) + ', vz=' + repr(self.vz) + \
            ', group=' + repr(self.group) + \
            ', pgroup=' + repr(self.pgroup) + ')'

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

      quad   : quadratic term of ephemeris in the same units as the times of
               the data

      vfine  : km/s to use for the fine array used to project into before
               blurring. Should be a few times (5x at most) smaller than the
               km/s used for any image.

      sfac : scale factor to use when computing data from the map. This is
             designed to allow the map images to take on "reasonable" values
             when matching a set of data. In the old F77 doppler code it was
             set to 0.0001 by default.

    """

    def __init__(self, head, data, tzero, period, quad, vfine, sfac=0.0001):
        """Creates a Map object

        head : an astropy.io.fits.Header object. A copy is taken as it
               is likely to be modified (comments added if keyword VERSION
               is not found)

        data : an Image or a list of Images

        tzero : zeropoint of ephemeris in the same units as the times of the
                data

        period : period of ephemeris in the same units as the times of the data

        quad   : quadratic term of ephemeris in the same units as the times
                 of the data

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
                'Each image must have a defined type (ITYPE) one of: PUNIT, NUNIT,')
            self.head.add_comment(
                'PSINE, NSINE, PCOSINE, NCOSINE, PSINE2, NSINE2, PCOSINE2, NCOSINE2. The')
            self.head.add_comment(
                'type determines the sign and modulation that the images adds in with.')
            self.head.add_comment(
                'P = +ve, N = -ve, UNIT = unmodulated, SINE / COSINE mean the flux of a')
            self.head.add_comment(
                'pixel is modulated on the sin / cosine of orbital phase; SINE2 / COSINE2')
            self.head.add_comment(
                'modulate on the sin / cosine of twice the orbital phase. Images can also')
            self.head.add_comment(
                'be assigned to groups (GROUP) in order to simplify scaling and systemic')
            self.head.add_comment(
                'velocity optimisation. The related PGROUP parameter defines pairs that')
            self.head.add_comment(
                'need differencing for plots. The primary HDU contains parameters that')
            self.head.add_comment(
                'apply to all images. These specify an ephemeris (TZERO, PERIOD, QUAD),')
            self.head.add_comment(
                'a pixel size (VFINE) to be used for an intermediate finely-spaced array')
            self.head.add_comment(
                'during projection and an overall scale factor (SFAC) designed to allow')
            self.head.add_comment(
                'image values matching a given data set to have values of order unity.')

        try:
            for i, image in enumerate(data):
                if not isinstance(image, Image):
                    raise DopplerError(
                        'Map.__init__: element ' + str(i) + ' of map is not an Image.'
                    )
            self.data = data

        except TypeError as err:
            if not isinstance(data, Image):
                raise DopplerError(
                    'Map.__init__: data must be an Image or a list of Images'
                )
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
        tzero = head['TZERO']
        period = head['PERIOD']
        quad = head['QUAD']
        vfine = head['VFINE']
        sfac = head['SFAC']

        # Remove from the header
        del head['TZERO']
        del head['PERIOD']
        del head['QUAD']
        del head['VFINE']
        del head['SFAC']

        # Now the data
        data = []
        for n, ihdu in enumerate(hdul[1:]):
            if isinstance(ihdu, fits.ImageHDU):
                thdu = hdul[n+2] if n+2 < len(hdul) else None
                data.append(Image.fromHDU(ihdu,thdu))

        # OK, now make the map
        return cls(head, data, tzero, period, quad, vfine, sfac)

    def wfits(self, fname, overwrite=True):
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
            hdul += image.toHDU(i+1)
        hdulist = fits.HDUList(hdul)
        hdulist.writeto(fname, overwrite=overwrite)

    def isPositive(self):
        """
        Tests whether all pixels in the Map are positive
        """
        for i, image in enumerate(self.data):
            if not image.isPositive():
                return False
        return True

    def __itruediv__(self, other):
        """Divides Map by other"""
        if isinstance(other, Map):
            for image, oimage in zip(self.data, other.data):
                image /= oimage
        else:
            for image in self.data:
                image /= other
        return self

    def __imul__(self, other):
        """Multiplies Map by other"""
        if isinstance(other, Map):
            for image, oimage in zip(self.data, other.data):
                image *= oimage
        else:
            for image in self.data:
                image *= other
        return self

    def __isub__(self, other):
        """Subtracts other from self"""
        if isinstance(other, Map):
            for image, oimage in zip(self.data, other.data):
                image -= oimage
        else:
            for image in self.data:
                image -= other
        return self

    def __iadd__(self, other):
        """Adds other to self"""
        if isinstance(other, Map):
            for image, oimage in zip(self.data, other.data):
                image += oimage
        else:
            for image in self.data:
                image += other
        return self

    def __add__(self, other):
        """Adds other to self returning result as new Map
        """
        nmap = copy.deepcopy(self)
        nmap += other
        return nmap

    def __sub__(self, other):
        """Subtracts other from self returning result as new Map
        """
        nmap = copy.deepcopy(self)
        nmap -= other
        return nmap

    def __rsub__(self, other):
        """Returns other-self
        """
        nmap = copy.deepcopy(self)*-1
        nmap += other
        return nmap

    def __truediv__(self, other):
        """Divdes self by other returning result as new Map
        """
        nmap = copy.deepcopy(self)
        nmap /= other
        return nmap

    def __mul__(self, other):
        """Multiplies Map by other, returns result as new Map"""
        nmap = copy.deepcopy(self)
        nmap *= other
        return nmap

    def log(self):
        """Takes log of Map"""
        for image in self.data:
            image.log()

    def sumf(self):
        """Returns sum of image values"""
        sum = 0
        for image in self.data:
            sum += image.sumf()
        return sum

    def __repr__(self):
        return 'Map(head=' + repr(self.head) + \
            ', data=' + repr(self.data) + ', tzero=' + repr(self.tzero) + \
            ', period=' + repr(self.period) + ', quad=' + repr(self.quad) + \
            ', vfine=' + repr(self.vfine) + ', sfac=' + repr(self.sfac) + ')'

def inner_product(map1, map2, metric=None, invert=True):
    """Computes the scalar inner product between two Map objects with an optional
    diagonal metric. The metric, if defined, should be a matching Map. It is
    assumed to represent the fully contravariant components of the metric and
    thus to be applicable directly to obtaining the scalar length given two
    one-form inputs. Otherwise it needs to be inverted. Inversion is set to be
    the default so that standardly map1 and map2 are assumed to be vector
    inputs. If they are one-forms, then 'invert' must be set = False. If map1
    and map2 are heterogeneous then no metric is required.

    """

    inner = 0
    if metric is None:
        for image1, image2 in zip(map1.data, map2.data):
            inner += (image1.data*image2.data).sum()
    else:
        for image1, image2, mimage in zip(map1.data, map2.data, metric.data):
            if invert:
                inner += (image1.data*image2.data/mimage.data).sum()
            else:
                inner += (image1.data*image2.data*mimage.data).sum()

    return inner

if __name__ == '__main__':

    # Generates a map, writes it to disc, reads it back, prints it
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
    def1   = Default.gauss2d(1.,200.)
    scale1 = np.array((1.0, 0.5))
    image1 = Image(data1, PUNIT, vxy, wave1, gamma1, def1, scale1)

    data2  = np.exp(-(((X+300.)/200.)**2+((Y+300.)/200.)**2)/2.)
    wave2  = 468.6
    gamma2 = 150.
    def2   = Default.gauss2d(1.,200.)
    image2 = Image(data2, PUNIT, vxy, wave2, gamma2, def2)

    tzero   =  2550000.
    period  =  0.15
    quad    =  0.0
    vfine   =  10.

    print('image2.default =',image2.default)

    # create the Map
    map = Map(head,[image1,image2],tzero,period,quad,vfine)

    # write to fits
    map.wfits('test.fits')

    # read from fits
    m = Map.rfits('test.fits')

    # print to screen
    print(m)
