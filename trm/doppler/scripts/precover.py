#!/usr/bin/env python

import argparse
import numpy as np
from astropy.io import fits
from trm import doppler

def precover(args=None):
    """precover is a Monte Carlo routine to test period recovery via SVD. It
    takes a model file (should have just one image) and a data file and
    various other parameters, generates data with noise from the model and
    then tries to locate the best period using the singular value
    decomposition approach. This routine can be very slow so you should start
    with small values of 'nmonte' and frequency search range, a large value of
    'delta' and just a few 'cond' values. All 'nmonte' realisations of the
    data have to be stored at once, an array is needed to store
    nmonte*nfreq*ncond chi**2 values, and an ngrid*ngrid*ndata matrix is
    created, so keep an eye out for memory usage [nfreq = number of
    frequencies searched, ndata = number of data points].

    The parameter 'cond' either takes value(s) < 1 meaning the lowest singular
    value as a fraction of the maximum, or >=1 meaning the integer number of
    the highest singular values to adopt. The frequency of minimum chi**2 will
    be reported for each value of 'cond' for each of the 'nmonte' trials, with
    one line of values per trial.

    """

    # deal with arguments
    parser = argparse.ArgumentParser(description=precover.__doc__)

    # positional
    parser.add_argument('model',  help='model file from which data are computed')
    parser.add_argument('data',   help='template data file')
    parser.add_argument('nmonte', type=int, help='number of monte carlo datasets')
    parser.add_argument('flow',   type=float, help='lowest frequency to search')
    parser.add_argument('fhigh',  type=float, help='highest frequency to search')
    parser.add_argument('delta',  type=float, help='number of cycles change over time base per frequency step')
    parser.add_argument('vgrid',  type=float, help='velocity spacing of grid')
    parser.add_argument('ngrid',  type=int, help='number of gaussians along side of grid')
    parser.add_argument('fratio', type=float, help='ratio FWHM/VGRID')
    parser.add_argument('ofile', help='ASCII file for output')
    parser.add_argument('cond',  type=float, nargs='+', help='SVD condition parameter(s)')

    # optional
    parser.add_argument('-n', dest='ntdiv', type=int,
                        default=11, help='spectrum sub-division factor')

    args = parser.parse_args()

    model  = doppler.Map.rfits(doppler.afits(args.model))
    if len(model.data) != 1:
        print('precover requires just one image in the model file')
        exit(1)

    data   = doppler.Data.rfits(doppler.afits(args.data))
    nmonte = args.nmonte
    flow   = args.flow
    fhigh  = args.fhigh
    delta  = args.delta
    vgrid  = args.vgrid
    ngrid  = args.ngrid
    fratio = args.fratio
    ntdiv  = args.ntdiv
    ofile  = args.ofile
    cond   = args.cond

    # create the grid
    head   = fits.Header()
    dat    = np.empty((ngrid,ngrid))
    tzero  = model.tzero
    period = 1/fhigh
    quad   = 0.
    wave   = model.data[0].wave
    gamma  = model.data[0].gamma
    scale  = model.data[0].scale

    grid = doppler.Grid(head,dat,tzero,period,quad,vgrid,fratio,wave,gamma,scale)

    # computations

    # first the frequency array
    tbase  = data.data[0].time.max() - data.data[0].time.min()
    nfreq  = int((fhigh-flow)*tbase/delta)
    fs     = np.linspace(flow,fhigh,nfreq)

    # compute data without noise
    doppler.comdat(model, data)

    # space for the data with noise
    buffer = []
    nbuff = 0
    for spectra in data.data:
        buffer.append(np.empty((nmonte,
                            spectra.flux.shape[0],spectra.flux.shape[1])))
        nbuff += buffer[-1].size

    # nmonte realisations
    for nd, spectra in enumerate(data.data):
        for nm in range(nmonte):
            buffer[nd][nm] = np.random.normal(spectra.flux, spectra.ferr)

    # allocate space for results as we need to complete all
    # computations before getting answers.
    chisq = np.empty((nfreq,nmonte,len(cond)))

    # outer loop over frequency so SVD need only to be done once per
    # frequency
    for nf, f in enumerate(fs):
        grid.period = 1/f

        # generate the matrix
        A = doppler.genmat(grid, data, ntdiv)

        # Carry out full SVD, returning smallest matrices possible
        u, s, v = np.linalg.svd(A,full_matrices=False)

        # replace u and v by their transposes
        v = np.transpose(v)
        u = np.transpose(u)
        smax = s[0]

        # Go through each value of the conditioning numbers to compute maximum
        # number of SVDs to use
        mok = 0
        for nc, c in enumerate(cond):

            # select the highest singular values with a method
            # determined by the value of the coniditioning number
            if c < 1.0:
                mok = max(mok, (s > c*smax).sum())
            else:
                mok = max(mok, min(len(s), int(round(c))))

        # calculate the worst case up-front to reduce amount of
        # repeated computation in cond loop
        v[:,:mok] /= s[:mok]

        for nm in range(nmonte):
            # update data from buffer
            for nd, spectra in enumerate(data.data):
                spectra.flux = buffer[nd][nm]

            # generate data vector and compute u*b matrix
            # product outside cond loop
            b  = doppler.genvec(data)
            ub = np.dot(u[:mok,:],b)

            # Go through each value of the conditioning numbers
            for nc, c in enumerate(cond):

                # select the highest singular values with a method
                # determined by the value of the coniditioning number
                if c < 1.0:
                    nok = (s > c*smax).sum()
                else:
                    nok = min(len(s), int(round(c)))

                # snew contains the inverses of the largest SVD values
                snew = 1/s[:nok]

                # calculate x = v*diag(snew)*u*b using
                # pre-calculated bits
                x = np.dot(v[:,:nok], ub[:nok])

                # the fit to the data corresponding to x ...
                # this is probably the slowest step overall
                fit = np.dot(A,x)

                # store chi**2
                chisq[nf,nm,nc] = ((b-fit)**2).sum()

    # Save frequencies of minimum chi**2
    with open(ofile, 'w') as fout:
        fout.write('# Output from precover.py\n')
        fout.write('#\n')
        fout.write('# Input arguments were:\n')
        fout.write('#\n')
        fout.write('# model  = ' + str(args.model) + '\n')
        fout.write('# data   = ' + str(args.data) + '\n')
        fout.write('# nmonte = ' + str(nmonte) + '\n')
        fout.write('# flow   = ' + str(flow) + '\n')
        fout.write('# fhigh  = ' + str(fhigh) + '\n')
        fout.write('# delta  = ' + str(delta) + '\n')
        fout.write('# vgrid  = ' + str(vgrid) + '\n')
        fout.write('# ngrid  = ' + str(ngrid) + '\n')
        fout.write('# fratio = ' + str(fratio) + '\n')
        fout.write('# ntdiv  = ' + str(ntdiv) + '\n')
        fout.write('# cond   = ' + str(cond) + '\n')
        fout.write('#\n')
        fout.write('# Memory usage (number of elements):\n')
        fout.write('#\n')
        fout.write('# A       = ' + str(A.size) + '\n')
        fout.write('# chi**2  = ' + str(chisq.size) + '\n')
        fout.write('# MC data = ' + str(nbuff) + '\n')
        fout.write('#\n')

        np.savetxt(fout, fs[np.argmin(chisq, axis=0)], '%.6e')
