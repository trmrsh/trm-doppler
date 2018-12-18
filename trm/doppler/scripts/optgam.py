#!/usr/bin/env python

import argparse
import numpy as np
from scipy.optimize import minimize_scalar, bracket
from trm import doppler
import copy

__all__ = ['optgam',]

def optgam(args=None):
    """
    optgam computes optimum values of gamma. Currently just handles
    a single global value.
    """

    parser = argparse.ArgumentParser(description=optgam.__doc__)

    # positional
    parser.add_argument('map',   help='name of the input map')
    parser.add_argument('data',  help='data file')
    parser.add_argument('output',  help='output map with optimized gamma')

    # optional
    parser.add_argument(
        '-s', dest='step', type=float, default=50.,
        help='initial step size for bracketing the minimum [km/s]'
    )
    # OK, done with arguments.
    args = parser.parse_args()

    # load map and data
    dmap = doppler.Map.rfits(doppler.afits(args.map))
    data = doppler.Data.rfits(doppler.afits(args.data))

    # create object for minimising.
    csq = Chisq(dmap, data)

    # Take first gamma as start
    gamma = dmap.data[0].gamma[0]

    # bracket minimum
    ga,gb,gc,ca,cb,cc,neval = bracket(csq, gamma-args.step, gamma+args.step)

    # bracket minimum
    res = minimize_scalar(csq,(ga,gb,gc))

    print('Optimum (global) gamma =',res.x,'km/s')

    # Write to a fits file
    dmap.wfits(doppler.afits(args.output))

class Chisq:
    """Function object returning chi**2/n given a value of gamma"""
    def __init__(self, dmap, data):
        self.dmap = dmap
        self.data = data

    def __call__(self, gamma):
        # set the gamma values
        for image in self.dmap.data:
            image.gamma.fill(gamma)

        # copy data
        dmodel = copy.deepcopy(self.data)

        # compute model data
        doppler.comdat(self.dmap, dmodel)

        # calculate chisq
        chisq, ndata = doppler.chisquared(self.data,dmodel)

        return chisq/ndata
