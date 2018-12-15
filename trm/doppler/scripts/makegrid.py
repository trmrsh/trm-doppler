#!/usr/bin/env python

import argparse, os, ConfigParser
import numpy as np
from astropy.io import fits
from trm import doppler

def makegrid(args=None):
    """makegrid creates Doppler grids from configuration files to provide starter
    maps and for testing. Use the -w option to write out an example config
    file to start from. config files must end in ".cfg".

    """

    parser = argparse.ArgumentParser(
        description=makegrid.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # positional
    parser.add_argument(
        'config', help='configuration file name, output if -w is set'
    )
    parser.add_argument(
        'grid', nargs='?', default='mgrid.fits', help='name of output grid'
    )

    # optional
    parser.add_argument(
        '-w', dest='write', action='store_true',
        help='Will write an example config file rather than read one'
    )
    parser.add_argument(
        '-c', dest='clobber', action='store_true',
        help='Clobber output files, both config for -w and the FITS file'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    if args.write:
        if not args.clobber and os.path.exists(doppler.acfg(args.config)):
            print('\nERROR: ',doppler.acfg(args.config),
                  'already exists and will not be overwritten.')
            exit(1)

        if args.grid != 'mgrid.fits':
            print('\nWARNING: ignoring map output file =',args.grid)

        # Example config file
        config = """\
# This is an example of a configuration file needed by makegrid.py to create
# Doppler grids.
#
# Main section:
#
# version  : YYYYMMDD version number used to check config file's compatibility
#            with makegrid.py. Don't change this.
# target   : what objects this is meant to configure to reduce chances of
#            confusion with other configuration files. Don't change this.
# clobber  : overwrite any existing file of the same name or not
# ngrid    : number of gaussians on a side of the square grid
# vgrid    : km/s spacing of gaussians
# fratio   : ratio gaussian FWHM / spacing
# tzero    : zeropoint of ephemeris
# period   : period of ephemeris
# quad     : quadratic term of ephemeris
# sfac     : global scaling factor to keep image values in a nice range
# wave1    : wavelength of first line associated with the image
# gamma1   : systemic velocity, km/s, of first line associated with the image
# scale1   : scale factor of first line associated with the image [ignored
#            if only one wavelength]
# wave2    : wavelength of second line associated with the image
# gamma2   : systemic velocity, km/s, of second line associated with the image
# scale2   : scale factor of second line associated with the image
# wave3    : ... repeat as desired ...

[main]
version  =  {0}
target   =  grids
clobber  =  no
ngrid    =  20
vgrid    =  200.
fratio   =  1.0
tzero    =  50000.
period   =  0.1
quad     =  0.0
sfac     =  0.0001
wave1    = 486.1
gamma1   = 100.
scale1   = 1.0
wave2    = 434.0
gamma2   = 120.
scale2   = 0.6

# keywords / values for the FITS header. Optional

[fitshead]
ORIGIN = makegrid.py
OBJECT = SS433
"""
        with open(doppler.acfg(args.config),'w') as fout:
            fout.write(config.format(doppler.VERSION))
    else:

        if not args.clobber and os.path.exists(doppler.afits(args.grid)):
            print('\nERROR: ',doppler.afits(args.grid),
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
        if target != 'grids':
            print('Found target =',target,'but expected = grids')
            print('Please check this is the right sort of config file')
            exit(1)

        clobber = config.getboolean('main', 'clobber')
        ngrid = config.getfloat('main', 'ngrid')
        vgrid = config.getfloat('main', 'vgrid')
        fratio = config.getfloat('main', 'fratio')
        tzero = config.getfloat('main', 'tzero')
        period = config.getfloat('main', 'period')
        quad = config.getfloat('main', 'quad')
        sfac = config.getfloat('main', 'sfac')

        if config.has_option('main', 'wave2'):
            wave, gamma, scale = [], [], []
            nwave = 1
            while True:
                w = 'wave' + str(nwave)
                if not config.has_option('main',w):
                    break
                g = 'gamma' + str(nwave)
                s = 'scale' + str(nwave)
                wave.append(config.getfloat('main',w))
                gamma.append(config.getfloat('main',g))
                scale.append(config.getfloat('main',s))
                nwave += 1

        else:
            wave = config.getfloat('main','wave1')
            gamma = config.getfloat('main','gamma1')
            scale = None

        # the header
        mhead = fits.Header()
        if config.has_section('fitshead'):
            for name, value in config.items('fitshead'):
                if len(name) <= 8:
                    mhead[name] = value
                else:
                    print(
                        '\nERROR: Keyword in fitshead section = ' +
                        name + ' is too long.'
                    )
                    exit(1)

        # create the Grid
        grid = doppler.Grid(
            mhead, np.zeros((ngrid,ngrid)), tzero, period, quad,
            vgrid, fratio, wave, gamma, scale, sfac
        )

        # Write to a fits file
        grid.wfits(
            doppler.afits(args.grid),clobber=(args.clobber or clobber)
        )
