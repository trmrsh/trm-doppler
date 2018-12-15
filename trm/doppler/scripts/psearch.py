#!/usr/bin/env python

import sys, argparse
import numpy as np
from trm import doppler

def psearch(args=None):
    """Carries out a period search by computing chi**2 versus period, or more
    accurately frequency since one can step uniformly in frequency. Results
    are printed to stdout. This routine uses single value decomposition to
    match a grid of gaussians to data. The parameter cond either takes
    value(s) < 1 meaning the lowest singular value as a fraction of the
    maximum, or >=1 meaning the integer number of the highest singular values
    to adopt. Be warned: this is slow.

    One line per frequency will be printed consisting of the frequency, the
    period, the chi**2 values [one per value of cond], reduced chi**2 [one per
    cond], and then information on the conditioning [one per cond]. If a value
    of cond < 1, then the number of SVs used will be reported; if cond > 1
    then the largest SV used as a fraction of the maximum will be reported.

    """

    parser = argparse.ArgumentParser(description=psearch.__doc__)

    # positional
    parser.add_argument('grid',  help='name of grid file')
    parser.add_argument('data',  help='name of data file')
    parser.add_argument('flow',  type=float, help='Lower frequency limit, cycles / unit time')
    parser.add_argument('fhigh', type=float, help='Upper frequency limit, cycles / unit time')
    parser.add_argument('delta', type=float, help='Number of cycles change over time base for one frequency step')
    parser.add_argument('cond',  type=float, nargs='+', help='SVD condition parameter(s)')

    # optional
    parser.add_argument('-n', dest='ntdiv', type=int, default=11,
                        help='spectrum sub-division factor for finite exposures')

    # OK, done with arguments.
    args = parser.parse_args()

    # a few checks
    if args.flow <= 0 or args.fhigh <= args.flow:
        print('Upper frequency must excedd the lower, and both muts be > 0')
        exit(1)

    if args.delta <= 0 or args.delta >=1:
        print('Cycle change per frequency step must lie between 0 and 1.')
        exit(1)

    # load map and data
    grid = doppler.Grid.rfits(doppler.afits(args.grid))
    data = doppler.Data.rfits(doppler.afits(args.data))

    tbase = data.data[0].time.max() - data.data[0].time.min()
    nf = int((args.fhigh-args.flow)*tbase/args.delta)
    fs = np.linspace(args.flow,args.fhigh,nf)

    # a bit of header
    print('# Output from the SVD routine, psearch.py')
    print('#')
    print('# Grid file =',args.grid)
    print('# Data file =',args.data)
    print('# Frequency range, number of frequencies =',args.flow,args.fhigh,nf)
    print('# Cycle change per frequency =',args.delta)
    print('# Condition values =',args.cond)
    print('#')

    for f in fs:
        grid.period = 1/f
        chisq, cred, sing, s = doppler.svd(grid, data, args.cond, args.ntdiv)
        print(f, 1/f, ' '.join([str(c) for c in chisq]), \
              ' '.join([str(c) for c in cred]), ' '.join([str(sng) for sng in sing]))
        sys.stdout.flush()
