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
DAT-OBS

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

# Next sections are optional. They allow the addition of spots and discs to
# make something more interesting than a constant.
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

# Discs: discs are defined by a centre of symmetry in Vx-Vy, a plane of symmetry in
# Vz, a velocity of peak intensity, the intensity at peak, and outer and inner power
# law exponents to define how the intensity changes away from the peak. i.e. for v > vpeak,
# the intensity scales as (v/vpeak)**eout. In the Vz direction the disc  is gaussian.
#
# Each disc requires:
#
# vx    : centre of symmetry in Vx
# vy    : centre of symmetry in Vy
# vz    : centre of symmetry in Vz (only if nz > 1)
# fwhmz : FWHM in Vz (only if nz > 1)
# vpeak : velocity of peak (outer disc velocity)
# ipeak : intensity per pixel at peak
# eout  : outer power law exponent
# ein   : inner power law exponent

[disc1]
vx    = 0
vy    = -50.
vz    = 0.
fwhmz = 50.
vpeak = 450.
ipeak = 1.0
eout  = -2.5
ein   = +3.0

[disc2]
vx    = 0
vy    = -50.
vz    = 0.
fwhmz = 50.
vpeak = 550.
ipeak = 0.5
eout  = -2.5
ein   = +3.0
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

        # look for discs to add
        disc = 'disc' + str(nimage)
        if config.has_section(disc):
            vx     = config.getfloat(disc,'vx')
            vy     = config.getfloat(disc,'vy')
            if nz > 1:
                # need to avoid overwriting the vz
                # pixel size parameter
                vzs  = config.getfloat(spot,'vz')
                sigz = config.getfloat(spot,'fwhmz')/doppler.EFAC

            vpeak  = config.getfloat(disc,'vpeak')
            ipeak  = config.getfloat(disc,'ipeak')
            eout   = config.getfloat(disc,'eout')
            ein    = config.getfloat(disc,'ein')

            if nz == 1:
                x, y = doppler.meshgrid(nxy, vxy)
            else:
                vzs = config.getfloat(spot,'vz')
                x, y, z = doppler.meshgrid(nxy, vxy, nz, vz)

            # cylindrical coord radius for each point
            r = np.sqrt((x-vx)**2+(y-vy)**2)

            # Add low velocity disc
            add = r <= vpeak
            if nz == 1:
                array[add] += ipeak*(r[add]/vpeak)**ein
            else:
                array[add] += ipeak*np.exp(-(z[add]-vzs)**2/(2.*sigz**2))*(r[add]/vpeak)**ein

            # Add high velocity disc
            add = r > vpeak
            if nz == 1:
                array[add] += ipeak*(r[add]/vpeak)**eout
            else:
                array[add] += ipeak*np.exp(-(z[add]-vzs)**2/(2.*sigz**2))*(r[add]/vpeak)**eout

        # create and store image
        images.append(doppler.Image(array, vxy, wave, gamma, default, scale, vz))
        print('Created image number',nimage,', wavelength(s) =',wave)
        nimage += 1

    # create the Map
    map = doppler.Map(mhead,images,tzero,period,quad,vfine,sfac)

    # Write to a fits file
    map.wfits(doppler.afits(args.map),clobber=(args.clobber or clobber))
