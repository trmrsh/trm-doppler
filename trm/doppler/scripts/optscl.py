#!/usr/bin/env python

import argparse
import numpy as np
from scipy import linalg
from trm import doppler
import copy

__all__ = ['optscl',]

def retarr(data):
    """Returns data and errors (>0) in a Data as two 1D arrays

    """
    dat = np.array([])
    err = np.array([])
    for spectra in data.data:
        ok = spectra.ferr > 0.
        dat = np.concatenate([dat,spectra.flux[ok].flatten()])
        err = np.concatenate([err,spectra.ferr[ok].flatten()])
    return (dat,err)

def optscl(args=None):
    """
    optscl computes optimum scaling and in useful when starting to get into
    the right ballpark and thereby save iterations.
    """

    parser = argparse.ArgumentParser(description=optscl.__doc__)

    # positional
    parser.add_argument('map',   help='name of the input map')
    parser.add_argument('data',  help='data file')
    parser.add_argument('scaled',  help='scaled output map')

    # optional
    parser.add_argument(
        '-i', dest='iscale', action='store_true',
        help='individual scaling (else a single global scale)'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    # load map and data
    dmap = doppler.Map.rfits(doppler.afits(args.map))
    data = doppler.Data.rfits(doppler.afits(args.data))

    nscale = 0
    for image in dmap.data:
        nscale += len(image.wave)

    if args.iscale and nscale > 1:
        # in this option we try to individually scale the images
        mtemp = copy.deepcopy(dmap)
        flux, ferr = retarr(data)
        wgt = np.empty_like(ferr)
        ok = ferr > 0
        wgt[ok] = 1./ferr[ok]**2

        # create indices to access the scale factors
        # save old scale factors
        sindices = []
        osfacs = []
        for ni, image in enumerate(dmap.data):
            for ns in range(len(image.scale)):
                sindices.append((ni,ns))
                osfacs.append(image.scale[ns])
                image.scale[ns] = 0.

        # compute a set of data vectors with each scale factor
        # set to 1 with all others set = 0, one by one
        dvecs = []
        for ni, ns in sindices:
            dmap.data[ni].scale[ns] = 1.0

            # compute data equivelent to data
            dvec = copy.deepcopy(data)
            doppler.comdat(dmap, dvec)
            dvecs.append(retarr(dvec)[0])

            dmap.data[ni].scale[ns] = 0.0

        # compile least-squares matrix & right-hand vector
        nvec = len(dvecs)
        A = np.empty((nvec,nvec))
        b = np.empty((nvec))
        for j in range(nvec):
            b[j] = (wgt[ok]*dvecs[j]*flux[ok]).sum()
            for i in range(j+1):
                A[j][i] = (wgt[ok]*dvecs[j]*dvecs[i]).sum()
                A[i][j] = A[j][i]

        nsfacs = linalg.solve(A,b)
        ocalc = np.zeros_like(flux)
        ncalc = np.zeros_like(flux)
        for j in range(nvec):
            ocalc += osfacs[j]*dvecs[j]
            ncalc += nsfacs[j]*dvecs[j]

        ndata = flux.size
        cold = (wgt*(flux-ocalc)**2).sum()/ndata
        cnew = (wgt*(flux-ncalc)**2).sum()/ndata
        print('Chi**2/ndata (before) =',cold,' (after) =',cnew)

        # set the new scale factors in place
        i = 0
        for ni, ns in sindices:
            dmap.data[ni].scale[ns] = nsfacs[i]
            i += 1

        # set the new scale factors in place
        i = 0
        for ni, ns in sindices:
            dmap.data[ni].scale[ns] = nsfacs[i]
            i += 1

        # set the singleton scale factors = 1 by
        # re-scaling the corresponding images instead.
        for ni, image in enumerate(dmap.data):
            if len(image.scale) == 1:
                image.data *= image.scale[0]
                image.scale[0] = 1

    else:
        # compute data equivalent to data
        dcalc = copy.deepcopy(data)
        doppler.comdat(dmap, dcalc)

        # compute optimum scale factor.
        sum0  = 0.
        sum1  = 0.
        sum2  = 0.
        ndata = 0
        for cspec, dspec in zip(dcalc.data, data.data):
            ok     = dspec.ferr > 0.
            sum0  += ((dspec.flux[ok]/dspec.ferr[ok])**2).sum()
            sum1  += ((cspec.flux[ok]/dspec.ferr[ok])*(dspec.flux[ok]/dspec.ferr[ok])).sum()
            sum2  += ((cspec.flux[ok]/dspec.ferr[ok])**2).sum()
            ndata += dspec.ferr.size

        scale = sum1 / sum2
        cold = cnew = 0
        for cspec, dspec in zip(dcalc.data, data.data):
            ok    = dspec.ferr > 0.
            cold += (((dspec.flux[ok]-cspec.flux[ok])/dspec.ferr[ok])**2).sum()
            cnew += (((dspec.flux[ok]-scale*cspec.flux[ok])/dspec.ferr[ok])**2).sum()

        print('ndata =',ndata)
        print('Optimum scale factor =',scale)
        print('Chi**2/ndata (before) =',cold/ndata,' (after) =',cnew/ndata)

        # scale images and write out
        for image in dmap.data:
            image.data *= scale

    # Write to a fits file
    dmap.wfits(doppler.afits(args.scaled))
