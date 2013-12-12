#!/usr/bin/env python

from __future__ import print_function

usage = \
"""
makemap creates Doppler maps from configuration files to provide starter maps
and for testing. Use the -w option to write out an example config file to
start from. config files must end in ".cfg".
"""

import argparse, os, ConfigParser
import numpy as np
from astropy.io import fits
from trm import doppler

parser = argparse.ArgumentParser(description=usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# positional
parser.add_argument('config', help='configuration file name, output if -w is set')
parser.add_argument('map', nargs='?', default='mmap.fits', help='name of output map')

# optional
parser.add_argument('-w', dest='write', action='store_true',
                    help='Will write an example config file rather than read one')
parser.add_argument('-c', dest='clobber', action='store_true',
                    help='Clobber output files, both config for -w and the FITS file')

# OK, done with arguments.
args = parser.parse_args()

if args.write:
    if not args.clobber and os.path.exists(doppler.acfg(args.config)):
        print('\nERROR: ',doppler.acfg(args.config),
              'already exists and will not be overwritten.')
        exit(1)

    if args.map != 'mmap.fits':
        print('\nWARNING: ignoring map output file =',args.map)

    # Example config file
    config = """\
# This is an example of a configuration file needed by makemap.py to create
# starter Doppler maps. It allows you to define one or more images, each of
# which can be associated with one or more atomic lines.  You will need to
# define the dimensions of each image and the default to use during MEM
# iterations.
#
# Main section:
#
# version  : YYYYMMDD version number used to check config file's compatibility
#            with makemap.py. Don't change this.
# target   : what objects this is meant to configure to reduce chances of
#            confusion with other configuration files. Don't change this.
# clobber  : overwrite any existing file of the same name or not
# vfine    : km/s/pixel of fine array
# tzero    : zeropoint of ephemeris
# period   : period of ephemeris
# quad     : quadratic term of ephemeris
# sfac     : global scaling factor to keep image values in a nice range

[main]
version  =  {0}
target   =  maps
clobber  =  no
vfine    =  5.
tzero    =  50000.
period   =  0.1
quad     =  0.0
sfac     =  0.0001

# keywords / values for the FITS header. Optional

[fitshead]
ORIGIN = makemap.py
OBJECT = SS433

# image sections. Each image requires the following:
#
# nxy    : number of pixels on a side in Vx-Vy plane
# nz     : number of Vz slice
# vxy    : km/s/pixel in Vx-Vy plane
# vz     : km/s/slice in Vz direction
# back   : background value to set image to.
# default: default to use (Uniform, Gaussian)
# fwhmxy : if Gaussian, this is FWHM, km/s,to use in X-Y plane
# fwhmz  : if Gaussian, this is FWHM, km/s, to use in Z (ignored if nz == 1)
# wave1  : wavelength of first line associated with the image
# gamma1 : systemic velocity, km/s, of first line associated with the image
# scale1 : scale factor of first line associated with the image [ignored
#          if only one wavelength]
# wave2  : wavelength of second line associated with the image
# gamma2 : systemic velocity, km/s, of second line associated with the image
# scale2 : scale factor of second line associated with the image
# wave3  : ... repeat as desired ...

[image1]
nxy     = 250
nz      = 1
vxy     = 20.
vz      = 0.
back    = 1.e-6
default = Gaussian
fwhmxy  = 500.
fwhmz   = 0.
wave1   = 486.1
gamma1  = 100.
scale1  = 1.0
wave1   = 486.1
gamma1  = 100.
scale1  = 1.0
wave2   = 434.0
gamma2  = 120.
scale2  = 0.6

[image2]
nxy     = 250
nz      =  1
vxy     = 20.
vz      =  0.
back    = 1.0e-6
default = Gaussian
fwhmxy  = 500.
fwhmz   = 0.
wave1   = 468.6
gamma1  = 100.

# Next sections are optional. They allow the addition of gaussian
# spots to make something more interesting than a constant. A
# section like [spot1_2] means the second spot for image 1.
# Each spot is defined by:
#
# vx     : centre of spot in Vx
# vy     : centre of spot in Vy
# vz     : centre of spot in Vz (ignored if nz == 1)
# fwhm   : FWHM width parameter
# height : height.

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

"""
    with open(doppler.acfg(args.config),'w') as fout:
        fout.write(config.format(doppler.VERSION))
else:

    if not args.clobber and os.path.exists(doppler.afits(args.map)):
        print('\nERROR: ',doppler.afits(args.map),
              'already exists and will not be overwritten.')
        exit(1)

    config = ConfigParser.RawConfigParser()
    config.read(doppler.acfg(args.config))

    tver   = config.getint('main', 'version')
    if tver != doppler.VERSION:
        print('Version number in config file =',tver,
              'conflicts with version of script =',doppler.VERSION)
        print('Will continue but there may be problems')

    target = config.get('main', 'target')
    if target != 'maps':
        print('Found target =',target,'but expected = maps')
        print('Please check this is the right sort of config file')
        exit(1)

    clobber= config.getboolean('main', 'clobber')
    vfine  = config.getfloat('main', 'vfine')
    tzero  = config.getfloat('main', 'tzero')
    period = config.getfloat('main', 'period')
    quad   = config.getfloat('main', 'quad')
    sfac   = config.getfloat('main', 'sfac')

    # the header
    mhead = fits.Header()
    if config.has_section('fitshead'):
        for name, value in config.items('fitshead'):
            if len(name) <= 8:
                mhead[name] = value
            else:
                print('\nERROR: Keyword in fitshead section = ' + name + ' is too long.')
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
        defop = config.get(img,'default')
        if defop == 'Uniform':
            default = doppler.Default.uniform()
        elif defop == 'Gaussian':
            fwhmxy = config.getfloat(img,'fwhmxy')
            if nz > 1:
                fwhmz = config.getfloat(img,'fwhmz')
                default = doppler.Default.gauss3d(fwhmxy, fwhmz)
            else:
                default = doppler.Default.gauss2d(fwhmxy)

        if config.has_option(img, 'wave2'):
            wave, gamma, scale = [], [], []
            nwave = 1
            while True:
                w = 'wave' + str(nwave)
                if not config.has_option(img,w):
                    break
                g = 'gamma' + str(nwave)
                s = 'scale' + str(nwave)
                wave.append(config.getfloat(img,w))
                gamma.append(config.getfloat(img,g))
                scale.append(config.getfloat(img,s))
                nwave += 1

        else:
            wave  = config.getfloat(img,'wave1')
            gamma = config.getfloat(img,'gamma1')
            scale = None

        # look for spots to add
        sroot = 'spot' + str(nimage) + '_'
        nspot = 1
        while True:
            spot = sroot + str(nspot)
            if not config.has_section(spot):
                    break

            fwhm   = config.getfloat(spot,'fwhm')
            height = config.getfloat(spot,'height')
            vx     = config.getfloat(spot,'vx')
            vy     = config.getfloat(spot,'vy')
            if nz > 1:
                vz = config.getfloat(spot,'vz')

            if nspot == 1:
                # Compute arrays
                vrange = vxy*(nxy-1)/2.

                # Aim in next bit is to return with array of squared
                # distance from the spot centre.
                x = y = np.linspace(-vrange,vrange,nxy)
                if nz == 1:
                    x, y = np.meshgrid(x, y)
                else:
                    nx = ny = nxy
                    vzrange = vz*(nz-1)/2.
                    z = np.linspace(-vzrange,vzrange,nz)

                    # carry out 3D version of meshgrid
                    x = x.reshape(1,1,nx)
                    y = y.reshape(1,ny,1)
                    z = z.reshape(nz,1,1)

                    # now extend in other 2 dimensions
                    x = x.repeat(ny,axis=1)
                    x = x.repeat(nz,axis=0)
                    y = y.repeat(nx,axis=2)
                    y = y.repeat(nz,axis=0)
                    z = z.repeat(nx,axis=2)
                    z = z.repeat(ny,axis=1)


            if nz == 1:
                rsq = (x-vx)**2+(y-vy)**2
            else:
                rsq = (x-vx)**2+(y-vy)**2+(z-vz)**2

            # Finally add in the spot
            array += height*np.exp(-rsq/((fwhm/doppler.EFAC)**2/2.))

            # move to the next one
            nspot += 1

        # create and store image
        images.append(doppler.Image(array, vxy, wave, gamma, default, scale, vz))
        print('Created image number',nimage,', wavelength(s) =',wave)
        nimage += 1

    # create the Map
    map = doppler.Map(mhead,images,tzero,period,quad,vfine,sfac)

    # Write to a fits file
    map.wfits(doppler.afits(args.map),clobber=(args.clobber or clobber))
