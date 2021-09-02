import argparse, os, configparser
import numpy as np
import pylab as plt
from astropy.io import fits
from trm import doppler

def makedata(args=None):
    """makedata creates a blank data file that can be used as a template for
    creating data files, e.g. using comdat. Like makemap it is driven by a
    configuration file. Use the -w option to write out an example config file
    to start from. config files must end in ".cfg"

    """

    parser = argparse.ArgumentParser(
        description=makedata.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # positional
    parser.add_argument('config', help='configuration file name, output if -w is set')
    parser.add_argument('data', nargs='?', default='mdat.fits', help='name of output data file')

    # optional
    parser.add_argument('-w', dest='write', action='store_true',
                        help='Will write an example config file rather than read one')
    parser.add_argument(
        '-o', dest='overwrite', action='store_true',
        help='Overwrite output files, both config for -w and the FITS file'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    if args.write:
        if not args.overwrite and os.path.exists(doppler.acfg(args.config)):
            print('\nERROR: ',doppler.acfg(args.config),
                  'already exists and will not be overwritten.')
            exit(1)

        if args.data != 'mdat.fits':
            print('\nWARNING: ignoring data output file =',args.data)

        # Example config file
        config = """\
# This is an example of a configuration file needed by makedata.py to create
# data template files for Doppler imaging, mainly for test purposes. It allows
# you to define one or more datasets, which can have different numbers of spectra
# The data arrays are created blank but can be filled in using a map and comdat.
#
# Main section:
#
# version  : YYYYMMDD version number used to check config file's compatibility
#            with makemap.py. Don't change this.
# target   : what objects this is meant to configure to reduce chances of
#            confusion with other configuration files. Don't change this.
# overwrite : overwrite any existing file of the same name or not

[main]
version   =  {0}
target    =  data
overwrite =  no

# keywords / values for the FITS header. Optional

[fitshead]
ORIGIN = makedata.py
OBJECT = SS433

# dataset sections. Each requires the following:
#
# wave1  : start wavelength (units must match those of maps)
# wave2  : end wavelength (>wave1)
# nwave  : number of wavelengths
# wrms   : wavelength jitter, RMS from spectrum to spectrum.
# time1  : time of first spectrum
# time2  : time of last spectrum
# nspec  : number of spectra
# error  : noise level (same for all pixels)
# fwhm   : FWHM resolution, km/s
# nsub   : spectrum sub-division factor (same for all spectra)

[dataset1]
wave1  = 420.
wave2  = 520.
nwave  = 2000
wrms   = 0.02
time1  = 50000.0
time2  = 50000.1
nspec  = 50
error  = 0.1
fwhm   = 150.
nsub   = 1

[dataset2]
wave1  = 480.
wave2  = 500.
nwave  = 100
wrms   = 0.01
time1  = 50000.0
time2  = 50000.1
nspec  = 20
error  = 0.1
fwhm   = 100.
nsub   = 3
"""
        with open(doppler.acfg(args.config),'w') as fout:
            fout.write(config.format(doppler.VERSION))
    else:

        if not args.overwrite and \
           os.path.exists(doppler.afits(args.data)):
            print('\nERROR: ',doppler.afits(args.data),
                  'already exists and will not be overwritten.')
            exit(1)

        config = configparser.RawConfigParser()
        config.read(doppler.acfg(args.config))

        tver   = config.getint('main', 'version')
        if tver != doppler.VERSION:
            print(
                f'Version number in config file = {tver} '
                f'conflicts with version of script = {doppler.VERSION}'
            )
            print('Will continue but there may be problems')

        target = config.get('main', 'target')
        if target != 'data':
            print('Found target =',target,'but expected = data')
            print('Please check that this is the right config file')
            exit(1)

        overwrite = config.getboolean('main', 'overwrite')

        # the header
        dhead = fits.Header()
        if config.has_section('fitshead'):
            for name, value in config.items('fitshead'):
                if len(name) <= 8:
                    dhead[name] = value
                else:
                    print('\nERROR: Keyword in fitshead section = ' + name + ' is too long.')
                    exit(1)

        # wind through the datasets
        ndset = 1
        data  = []
        while True:
            dat = 'dataset' + str(ndset)
            if not config.has_section(dat):
                if ndset == 1:
                    print('\nERROR: Could not find section = [dataset1]')
                    print('ERROR: You must define at least one dataset')
                    exit(1)
                break

            wave1 = config.getfloat(dat,'wave1')
            wave2 = config.getfloat(dat,'wave2')
            if wave2 <= wave1:
                print('ERROR: wave1 (=' + str(wave1) + ') must be < wave2 (=' + str(wave2) + ')')
                exit(1)

            nwave = config.getint(dat,'nwave')
            if nwave < 2:
                print('ERROR: nwave (=' + str(nwave) + ' must be > 1')
                exit(1)

            wrms  = config.getfloat(dat,'wrms')

            time1 = config.getfloat(dat,'time1')
            time2 = config.getfloat(dat,'time2')

            nspec = config.getint(dat,'nspec')
            if nspec < 2:
                print('ERROR: nspec (=' + str(nspec) + ' must be > 1')
                exit(1)

            error = config.getfloat(dat,'error')
            if error <= 0.:
                print('ERROR: error (=' + str(error) + ' must be > 0')
                exit(1)

            fwhm = config.getfloat(dat,'fwhm')
            if fwhm <= 0.:
                print('ERROR: fwhm (=' + str(fwhm) + ' must be > 0')
                exit(1)

            nsb = config.getint(dat,'nsub')
            if nsb < 1:
                print('ERROR: nsub (=' + str(nsb) + ' must be > 0')
                exit(1)

            flux = np.zeros((nspec,nwave),dtype=np.float32)
            ferr = np.empty_like(flux); ferr.fill(error)

            time   = np.linspace(time1,time2,nspec)

            # create wavelength array. Some jiggery-pokery to make it 2D
            wave = np.linspace(wave1,wave2,nwave)
            wave = wave.reshape(1,nwave).repeat(nspec,axis=0)

            # add some ransom wobble onto it
            if wrms > 0.:
                wobble = np.random.normal(scale=wrms, size=nspec).reshape(nspec,1)
                wave += wobble

            etime  = (time[-1]-time[0])/(nspec-1)
            expose = np.empty_like(time); expose.fill(etime)
            nsub   = np.empty_like(time,dtype=np.int)
            nsub.fill(nsb)

            # create & store the Spectra
            data.append(doppler.Spectra(flux,ferr,wave,time,expose,nsub,fwhm))
            print('Created dataset number',ndset,'with',nspec,'spectra of',nwave,'pixels each.')
            ndset += 1

        # create the Data
        data = doppler.Data(dhead,data)

        # Write to a fits file
        data.wfits(
            doppler.afits(args.data),
            overwrite=(args.overwrite or overwrite)
        )
        print('Written data to',doppler.afits(args.data))

