#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse
import numpy as np
from trm import doppler, pgram, subs

def sgrec(args=None):
    usage = \
            """ sgrec ("single gaussian recovery") is a comparison routine for
            precover.py. In this case the period is found by measuring
            velocities via cross-correlation with a sigle gaussian.  The
            routine operates like precover otherwise, i.e. Monte Carlo
            datasets are created and the best period from each is measured and
            reported. In this case just a single column of periods is sent to
            the output file.  """

    # deal with arguments
    parser = argparse.ArgumentParser(description=usage)

    # positional
    parser.add_argument('model', help='model file from which data are computed')
    parser.add_argument('data', help='template data file')
    parser.add_argument(
        'nmonte', type=int, help='number of monte carlo datasets'
    )
    parser.add_argument('flow', type=float, help='lowest frequency to search')
    parser.add_argument('fhigh', type=float, help='highest frequency to search')
    parser.add_argument(
        'delta', type=float,
        help='number of cycles change over time base per frequency step'
    )
    parser.add_argument(
        'fwhm',   type=float,
        help='FWHM of gaussian for cross-correlation (pixels)'
    )
    parser.add_argument('ofile', help='ASCII file for output')

    args = parser.parse_args()

    model  = doppler.Map.rfits(doppler.afits(args.model))
    if len(model.data) != 1:
        print('precover requires just one image in the model file')
        exit(1)

    data = doppler.Data.rfits(doppler.afits(args.data))
    nmonte = args.nmonte
    flow = args.flow
    fhigh = args.fhigh
    delta = args.delta
    fwhm = args.fwhm
    ofile = args.ofile

    # computations

    # first the frequency array
    tbase = data.data[0].time.max() - data.data[0].time.min()
    nfreq = int((fhigh-flow)*tbase/delta)
    fs = np.linspace(flow,fhigh,nfreq)

    # compute data without noise
    doppler.comdat(model, data)

    with open(ofile, 'w') as fout:
        fout.write('# Output from voverr.py\n')
        fout.write('#\n')
        fout.write('# Input arguments were:\n')
        fout.write('#\n')
        fout.write('# model  = ' + str(args.model) + '\n')
        fout.write('# data   = ' + str(args.data) + '\n')
        fout.write('# nmonte = ' + str(nmonte) + '\n')
        fout.write('# flow   = ' + str(flow) + '\n')
        fout.write('# fhigh  = ' + str(fhigh) + '\n')
        fout.write('# delta  = ' + str(delta) + '\n')
        fout.write('# fwhm   = ' + str(fwhm) + '\n')
        fout.write('#\n')

        for nm in range(nmonte):

            # add noise and accumulate V/R data
            times = []
            vs = []
            verrs = []
            for spectra in data.data:
                dat = np.random.normal(spectra.flux, spectra.ferr)
                vel = doppler.CKMS*(spectra.wave-model.data[0].wave[0])/model.data[0].wave[0]-model.data[0].gamma[0]
                wcen = model.data[0].wave
                gam = model.data[0].gamma

                for f,e,t,w in zip(dat,spectra.ferr,spectra.time,spectra.wave):
                    xcen, xerr = subs.centroid((len(f)-1)/2., fwhm, f, True, e)
                    icen = int(round(xcen))
                    if icen >=0 and icen < len(w):
                        dw = (w[-1]-w[0])/(len(w)-1)
                        v = doppler.CKMS*(w[icen] + (xcen-icen)*dw - wcen)/wcen - gam
                        dv = doppler.CKMS*xerr*dw/wcen
                        vs.append(v)
                        verrs.append(dv)
                        times.append(t)

                l = len(vs)
                vs = np.array(vs).reshape((l))
                verrs = np.array(verrs).reshape((l))
                times = np.array(times).reshape((l))

                # compute periodogram on same set of frequencies as
                p = pgram.wmls(times, vs, verrs, fs)

                # output best frequency
                fout.write(str(fs[p.argmax()]) + '\n')
