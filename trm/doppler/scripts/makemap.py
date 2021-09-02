#!/usr/bin/env python

import argparse, os, configparser
import numpy as np
from scipy import ndimage
from astropy.io import fits
from trm import doppler

def makemap(args=None):
    """makemap creates Doppler maps from configuration files to provide starter
    maps and for testing. Use the -w option to write out an example config
    file to start from. config files must end in ".cfg".

    """

    parser = argparse.ArgumentParser(
        description=makemap.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # positional
    parser.add_argument(
        'config', help='configuration file name, output if -w is set')
    parser.add_argument(
        'map', nargs='?', default='mmap.fits', help='name of output map')

    # optional
    parser.add_argument(
        '-w', dest='write', action='store_true',
        help='Will write an example config file rather than read one'
    )
    parser.add_argument(
        '-o', dest='overwrite', action='store_true',
        help='Overwite on output, both config for -w and the FITS file'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    if args.write:
        if not args.overwrite and os.path.exists(doppler.acfg(args.config)):
            print('\nERROR: ',doppler.acfg(args.config),
                  'already exists and will not be overwritten.')
            exit(1)

        if args.map != 'mmap.fits':
            print('\nWARNING: ignoring map output file =',args.map)

        # Example config file
        config = """\
# This is an example of a configuration file needed by makemap.py to create
# starter Doppler maps showing all possible features rather than attempting
# great realism. A few features are optional, as explained below. The config
# file allows you to define one or more images, each of which can be
# associated with one or more atomic lines.  You will need to define the
# dimensions of each image and the default to use during MEM iterations. NB
# You should respect data types, i.e. naturally floating point numbers need a
# decimal point; integers do not.
#
# Main section:
#
# version  : YYYYMMDD version number used to check config file's compatibility
#            with makemap.py. Don't change this.
# target   : what objects this is meant to configure to reduce chances of
#            confusion with other configuration files. Don't change this.
# overwrite : overwrite any existing file of the same name or not
# vfine    : km/s/pixel of fine array
# tzero    : zeropoint of ephemeris
# period   : period of ephemeris
# quad     : quadratic term of ephemeris
# sfac     : global scaling factor to keep image values in a nice range

[main]
version =  {0}
target =  maps
overwrite =  no
vfine =  5.
tzero =  50000.
period =  0.1
quad =  0.0
sfac =  0.0001

# keywords / values for the FITS header. Optional

[fitshead]
ORIGIN = makemap.py
OBJECT = SS433

# image sections. Each image can have the following:
#
# itype  : image type parameter. Possible values are PUNIT, NUNIT, PSINE,
#          NSINE, PCOSINE, NCOSINE, PSINE2, NSINE2, PCOSINE2, NCOSINE2
# group  : An integer assigning the image to a group. 0 implies no group.
#          Image groups are used to define images that should be treated
#          as one when it comes to optimising scaling factors and systemic
#          velocities. This parameter is optional. If not supplied the
#          group will be assumed = 0. The reason for having this is that
#          you may well not want to allow the modulation components to scale
#          arbitrarily since they may make so little difference to the result.
#          This allows you to link them to the main image instead.
# pgroup : if you have defined equivalent P- and N-type images, they should
#          always be differenced for plots. This parameter allows you to
#          identify pairs which need treating in this manner, and functions
#          like group. It is optional, and defaults to 0.
# nxy    : number of pixels on a side in Vx-Vy plane
# nz     : number of Vz slice
# vxy    : km/s/pixel in Vx-Vy plane
# vz     : km/s/slice in Vz direction
# back   : background value to set image to.
# default: default to use (UNIFORM, GAUSS2D, GAUSS3D)
# bias   : bias factor to apply to default. Usually 1 (no bias), but if you
#          want to suppress one image relative to another, a bias < 1 can be
#          useful.
# fwhmxy : if default=GAUSS2D or GAUSS3D, this is FWHM, km/s,to use in X-Y plane
# fwhmz  : if default=GAUSS3D, this is FWHM, km/s, to use in Z
# squeeze : factor from 0 to 1 to use when trying to squeeze a 3D default
#           idea is default will be (1-squeeze)*(usual gaussian blurr) + 
#           squeeze*(version of the gaussian blurr with same mean at each x-y
#           but enforced narrow FWHM).
# sqfwhm : FWHM to use for squeezed part (only if squeeze > 0)
# waves  : list of wavelengths associated with the image
# gammas : list of systemic velocities [km/s] associated with the image
# scales : list of scale factors associated with the image [ignored
#          if only one wavelength]
# wgshdu : True or False for whether to store waves, gammas, scales in a Table
#          HDU (or the image header). If missing, defaults to False.

# this image is a standard one applying to 2 lines
[image1]
itype   = PUNIT
nxy     = 250
nz      = 1
vxy     = 20.
vz      = 0.
back    = 1.e-6
default = GAUSS2D
bias    = 1.
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 486.1 434.0
gammas  = 100. 120.
scales  = 1.0 0.6
wgshdu  = True

# The next set of images define an image that has negative regions near the
# line centre as well as a modulated component.
[image2]
itype   = PUNIT
group   = 1
pgroup  = 1
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = GAUSS2D
bias    = 1.
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 468.6
gammas  = 100.

[image3]
itype   = NUNIT
group   = 1
pgroup  = 1
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = GAUSS2D
bias    = 0.9
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 468.6
gammas  = 100.

[image4]
itype   = PSINE
group   = 1
pgroup  = 2
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = GAUSS2D
bias    = 0.9
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 468.6
gammas  = 100.

[image5]
itype   = NSINE
group   = 1
pgroup  = 2
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = GAUSS2D
bias    = 0.9
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 468.6
gammas  = 100.

[image6]
itype   = PCOSINE
group   = 1
pgroup  = 3
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = GAUSS2D
bias    = 0.9
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 468.6
gammas  = 100.

[image7]
itype   = NCOSINE
group   = 1
pgroup  = 3
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = GAUSS2D
bias    = 0.9
fwhmxy  = 500.
fwhmz   = 0.
squeeze = 0.
sqfwhm  = 100.
waves   = 468.6
gammas  = 100.

# The next sections are entirely optional and should normally be removed if
# you are starting a real map since they are really aimed at creating
# artificial test maps. They allow the addition of spots and discs to make
# something more interesting than a constant, whereas generally one starts
# from a constant when making maps.
#
# Spots: These are gaussian in velocity space. A section like [spot1_2] means
# the second spot for the first image.  Each spot is defined by:
#
# vx     : centre of spot in Vx
# vy     : centre of spot in Vy
# vz     : centre of spot in Vz (ignored if nz == 1)
# fwhm   : FWHM width parameter
# height : height.
#
# NB not every image has to have a spot, so you could just start with spot2_1
# for example. Spots for non-existent images, e.g. spot3_1 here, will be
# ignored, as will non-sequential spots such as an isolated spot2_5

[spot1_1]
vx     = 500
vy     = -300
vz     = 0
fwhm   = 400.
height = 2.0

[spot1_2]
vx     = -700
vy     = -500
vz     = 0
fwhm   = 600.
height = 1.0

[spot2_1]
vx     = 500
vy     = 500
vz     = 0
fwhm   = 500.
height = 1.0

[spot3_1]
vx     = 0
vy     = 0
vz     = 0
fwhm   = 2000.
height = 0.05

# try to make something a bit like an irradiated secondary star
[spot2_2]
vx     = 0
vy     = 300
vz     = 0
fwhm   = 200.
height = 0.1

[spot7_1]
vx     = 0
vy     = 300
vz     = 0
fwhm   = 200.
height = 0.1

# Discs: discs are defined by a centre of symmetry in Vx-Vy, a plane of
# symmetry in Vz, a velocity of peak intensity, the intensity at peak, and
# outer and inner power law exponents to define how the intensity changes away
# from the peak. i.e. for v > vpeak, the intensity scales as
# (v/vpeak)**eout. In the Vz direction the disc is gaussian. The outermost emission
# is linearly tapered to zero between two limits, vout1 and vout2. The following
# conditions must apply vout2 > vout1 > vpeak. Once defined the result is blurred
# in Vx-Vy by an amount defined by fwhmxy. This smooths out the otherwise sharp
# changes
#
# Each disc requires:
#
# vx     : centre of symmetry in Vx
# vy     : centre of symmetry in Vy
# vz     : centre of symmetry in Vz (only if nz > 1)
# fwhmxy : FWHM blurr to apply in Vx-Vy
# fwhmz  : FWHM in Vz (only if nz > 1)
# vpeak  : velocity of peak (outer disc velocity)
# vout1  : velocity at which emission starts being linearly tapered
# vout2  : velocity at which emission is tapered to zero.
# ipeak  : intensity per pixel at peak
# eout   : outer power law exponent
# ein    : inner power law exponent

[disc1]
vx     = 0
vy     = -50.
vz     = 0.
fwhmxy = 200.
fwhmz  = 50.
vpeak  = 450.
vout1  = 2200.
vout2  = 2400.
ipeak  = 1.0
eout   = -2.5
ein    = +3.0

[disc2]
vx     = 0
vy     = -50.
vz     = 0.
fwhmxy = 200.
fwhmz  = 50.
vpeak  = 550.
vout1  = 2200.
vout2  = 2400.
ipeak  = 0.5
eout   = -2.5
ein    = +3.0
"""
        with open(doppler.acfg(args.config),'w') as fout:
            fout.write(config.format(doppler.VERSION))

    else:

        if not args.overwrite and os.path.exists(doppler.afits(args.map)):
            print('\nERROR: ',doppler.afits(args.map),
                  'already exists and will not be overwritten.')
            exit(1)

        config = configparser.RawConfigParser()
        config.read(doppler.acfg(args.config))

        tver = config.getint('main', 'version')
        if tver != doppler.VERSION:
            print('Version number in config file =',tver,
                  'conflicts with version of script =',doppler.VERSION)
            print('Will continue but there may be problems')

        target = config.get('main', 'target')
        if target != 'maps':
            print('Found target =',target,'but expected = maps')
            print('Please check this is the right sort of config file')
            exit(1)

        overwrite = config.getboolean('main', 'overwrite')
        vfine = config.getfloat('main', 'vfine')
        tzero = config.getfloat('main', 'tzero')
        period = config.getfloat('main', 'period')
        quad = config.getfloat('main', 'quad')
        sfac = config.getfloat('main', 'sfac')

        # the header
        mhead = fits.Header()
        if config.has_section('fitshead'):
            for name, value in config.items('fitshead'):
                if len(name) <= 8:
                    mhead[name] = value
                else:
                    print(
                        '\nERROR: Keyword in fitshead section = '
                        + name + ' is too long.'
                    )
                    exit(1)

        # wind through images
        nimage = 1
        images = []
        while True:
            img = 'image' + str(nimage)
            if not config.has_section(img):
                if nimage == 1:
                    print('\nERROR: Could not find section = [image1]')
                    print('ERROR: You must define at least one image')
                    exit(1)
                break

            itype = doppler.RITNAMES[config.get(img,'itype')]

            nxy = config.getint(img,'nxy')
            nz  = config.getint(img,'nz')
            if nz > 1:
                array = np.empty((nz,nxy,nxy))
            else:
                array = np.empty((nxy,nxy))

            vxy = config.getfloat(img,'vxy')
            if nz > 1:
                vz = config.getfloat(img,'vz')
            else:
                vz = None

            # add background
            back = config.getfloat(img,'back')
            array.fill(back)

            # Default
            defop = list(doppler.Default.DNAMES.keys())[
                list(doppler.Default.DNAMES.values()).index(
                    config.get(img,'default'))
            ]
            bias = config.getfloat(img,'bias')
            if defop == doppler.Default.UNIFORM:
                default = doppler.Default.uniform(bias)

            elif defop == doppler.Default.GAUSS2D:
                if nz > 1:
                    print('Cannot use GAUSS2D default for 3D image')
                    print('Probably want GAUSS3D')
                    exit(1)

                fwhmxy = config.getfloat(img,'fwhmxy')
                default = doppler.Default.gauss2d(bias, fwhmxy)

            elif defop == doppler.Default.GAUSS3D:
                if nz == 1:
                    print('Cannot use GAUSS3D default for 2D image')
                    print('Probably want GAUSS2D')
                    exit(1)

                fwhmxy = config.getfloat(img,'fwhmxy')
                fwhmz = config.getfloat(img,'fwhmz')
                squeeze = config.getfloat(img,'squeeze')
                sqfwhm = config.getfloat(img,'sqfwhm')
                default = doppler.Default.gauss3d(
                    bias, fwhmxy, fwhmz, squeeze, sqfwhm
                )

            if config.has_option(img, 'group'):
                group = config.getint(img,'group')
            else:
                group = 0

            wave = [float(f) for f in config.get(img,'waves').split()]
            gamma = [float(f) for f in config.get(img,'gammas').split()]
            if len(wave) > 1:
                scale = [float(f) for f in config.get(img,'scales').split()]
            else:
                scale = None

            wgshdu = config.getboolean(img,'wgshdu',fallback=False)

            # look for spots to add
            sroot = 'spot' + str(nimage) + '_'
            nspot = 1
            while True:
                spot = sroot + str(nspot)
                if not config.has_section(spot):
                    break

                fwhm = config.getfloat(spot,'fwhm')
                height = config.getfloat(spot,'height')
                vx = config.getfloat(spot,'vx')
                vy = config.getfloat(spot,'vy')
                if nz > 1:
                    # need to avoid overwriting the vz
                    # pixel size parameter
                    vzs = config.getfloat(spot,'vz')

                if nspot == 1:
                    # Compute coordinate arrays once
                    if nz == 1:
                        x, y = doppler.meshgrid(nxy, vxy)
                    else:
                        x, y, z = doppler.meshgrid(nxy, vxy, nz, vz)

                # compute distance squared from centre of spot
                if nz == 1:
                    rsq = (x-vx)**2+(y-vy)**2
                else:
                    rsq = (x-vx)**2+(y-vy)**2+(z-vzs)**2

                # Finally add in the spot
                array += height*np.exp(-rsq/((fwhm/doppler.EFAC)**2/2.))

                # move to the next one
                nspot += 1

            # look for discs to add. sigxy is blurring sigma
            # in Vx-Vy in pixels; sigz is in km/s in Vz
            disc = 'disc' + str(nimage)
            if config.has_section(disc):
                vx = config.getfloat(disc,'vx')
                vy = config.getfloat(disc,'vy')
                sigxy = config.getfloat(disc,'fwhmxy')/doppler.EFAC/vxy
                if nz > 1:
                    # need to avoid overwriting the vz
                    # pixel size parameter
                    vzs = config.getfloat(disc,'vz')
                    sigz = config.getfloat(disc,'fwhmz')/doppler.EFAC

                vpeak = config.getfloat(disc,'vpeak')
                vout1 = config.getfloat(disc,'vout1')
                vout2 = config.getfloat(disc,'vout2')
                if vpeak <= 0. or vpeak > vout1 or vout1 >= vout2:
                    print('\nERROR: vpeak, vout1, vout2 =',vpeak,vout1,vout2)
                    print('have invalid values. Must be > 0 and monotonically')
                    print('increase.')
                    exit(1)

                ipeak = config.getfloat(disc,'ipeak')
                eout = config.getfloat(disc,'eout')
                ein = config.getfloat(disc,'ein')
                if ein < 0.:
                    print('\nERROR: inner exponent ein =',ein,'cannot be negative.')
                    exit(1)

                # Compute image in 2D to start to save time as we need to
                # blurr in Vx-Vy
                x, y = doppler.meshgrid(nxy, vxy)

                # cylindrical coord radius for each point
                r = np.sqrt((x-vx)**2+(y-vy)**2)
                twod = np.empty_like(r)

                # Add disc components
                add = r <= vpeak
                twod[add] = ipeak*(r[add]/vpeak)**ein
                add = r > vpeak
                twod[add] = ipeak*(r[add]/vpeak)**eout

                # linear taper the outermost emission
                taper = (r > vout1) & (r < vout2)
                twod[taper] *= (vout2-r[taper])/(vout2-vout1)
                twod[r >= vout2] = 0.

                # blurr
                twod = ndimage.gaussian_filter(twod, sigma=sigxy, mode='constant')

                # Now add in
                if nz == 1:
                    array += twod
                else:
                    twod = np.reshape(twod, (1,twod.shape[0],twod.shape[1]))
                    vzs = config.getfloat(disc,'vz')
                    vzrange = vz*(nz-1)/2.
                    vza = np.linspace(-vzrange,vzrange,nz)
                    zw = np.exp(-((vza-vzs)/sigz)**2/2.)
                    zw = np.reshape(zw, (zw.shape[0],1,1))
                    array += zw*twod

            # create and store image
            images.append(
                doppler.Image(
                    array, itype, vxy, wave, gamma,
                    default, scale, vz, group, wgshdu=wgshdu
                )
            )
            print(
                'Created image number',nimage,', wavelength(s) =',wave,
                'type =',doppler.ITNAMES[itype]
            )
            nimage += 1

        # create the Map
        map = doppler.Map(mhead,images,tzero,period,quad,vfine,sfac)

        # Write to a fits file
        map.wfits(
            doppler.afits(args.map),
            overwrite=(args.overwrite or overwrite)
        )
        print('Written map to',doppler.afits(args.map))
