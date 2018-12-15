#!/usr/bin/env python

import argparse
import numpy as np
from trm import doppler, pgram

def vrrec(args=None):
    usage = \
            """ vrrec ("V over R") is a comparison routine for precover.py. In this case
            the period is found by the traditional violet over red (V/R) flux
            ratio method in which one simply computes the ratio of blue to
            red-shifted flux. The routine operates like precover otherwise,
            i.e. Monte Carlo datasets are created and the best period from
            each is measured and reported. In this case just a single column
            of periods is sent to the output file.  """

    # deal with arguments
    parser = argparse.ArgumentParser(description=usage)

    # positional
    parser.add_argument('model', help='model file from which data are computed')
    parser.add_argument('data', help='template data file')
    parser.add_argument(
        'nmonte', type=int,
        help='number of monte carlo datasets'
    )
    parser.add_argument('flow', type=float, help='lowest frequency to search')
    parser.add_argument('fhigh', type=float, help='highest frequency to search')
    parser.add_argument(
        'delta', type=float,
        help='number of cycles change over time base per frequency step'
    )
    parser.add_argument(
        'vmax',   type=float, help='maximum velocity to include in V/R sums'
    )
    parser.add_argument('ofile', help='ASCII file for output')

    args = parser.parse_args()

    model = doppler.Map.rfits(doppler.afits(args.model))
    if len(model.data) != 1:
        print('precover requires just one image in the model file')
        exit(1)

    data = doppler.Data.rfits(doppler.afits(args.data))
    nmonte = args.nmonte
    flow = args.flow
    fhigh = args.fhigh
    delta  = args.delta
    vmax = args.vmax
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
        fout.write('# vmax   = ' + str(vmax) + '\n')
        fout.write('#\n')

        for nm in range(nmonte):

            # add noise and accumulate V/R data
            times  = []
            ratios = []
            rerrs  = []
            for spectra in data.data:
                dat = np.random.normal(spectra.flux, spectra.ferr)
                vel = doppler.CKMS*(spectra.wave-model.data[0].wave[0])/model.data[0].wave[0]-model.data[0].gamma[0]

                for f,e,v,t in zip(dat,spectra.ferr,vel,spectra.time):
                    bpix = (v > -vmax) & (v < 0)
                    blue = f[bpix].sum()
                    bvar = (e[bpix]**2).sum()
                    rpix = (v < vmax) & (v >= 0)
                    red = f[rpix].sum()
                    rvar = (e[rpix]**2).sum()
                    rat = blue/red
                    rerr = np.abs(rat)*np.sqrt(bvar/blue**2+rvar/red**2)
                    ratios.append(rat)
                    rerrs.append(rerr)
                    times.append(t)

                ratios = np.array(ratios)
                rerrs = np.array(rerrs)
                times = np.array(times)

                # compute periodogram on same set of frequencies as
                p = pgram.wmls(times, ratios, rerrs, fs)

                # output best frequency
                fout.write(str(fs[p.argmax()]) + '\n')
