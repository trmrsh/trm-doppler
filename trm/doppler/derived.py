#!/usr/bin/env python

"""
Routines that requires 2 or more of the other sub-packages
"""
from __future__ import absolute_import

import numpy as np
from scipy import linalg

from .data  import *
from .grid  import *

def genmat(grid, data, ntdiv):
    """
    Computes matrix A and right-hand vector b when representing
    Doppler image problem by A x = b

    Returns (A,b) where::

      A : NxM matrix where N is number of data and M is number of grid
          points

      b : Nx1 vector

      ntdiv : sub-division factor to spread the model within exposures using
              trapezoidal averaging. Note that only a single value can be specified
              for technical reasons. One tends to get oscillations in chi**2 for small
              ntdiv, so values of order 10 are recommended.
    """

    ndata, ngrid = data.size, grid.size

    nside = grid.data.shape[0]

    # Reserve space for matrix and vector (will solve
    # Ax = b in least-squares sense)
    A = np.zeros((ngrid, ndata))
    b = np.empty((ndata))

    vgrid = grid.vgrid

    nrow = 0
    for iy in xrange(nside):
        vy = vgrid*(iy-(nside-1)/2.)
        for ix in xrange(nside):
            vx = vgrid*(ix-(nside-1)/2.)
            noff = 0
            for nd in xrange(len(data.data)):
                dat = data.data[nd]
                for nt in xrange(ntdiv):
                    # compute phases
                    t = dat.time+dat.expose*(float(nt)-float(ntdiv-1)/2.)/ \
                        max(ntdiv-1,1)
                    phase = (t-grid.tzero)/grid.period
                    for nc in xrange(2):
                        corr   = grid.tzero + \
                            phase*(grid.period + grid.quad*phase) - t
                        deriv  = grid.period + 2.*grid.quad*phase
                        phase -= corr/deriv

                    phase *= 2.*np.pi
                    cosp   = np.cos(phase)
                    sinp   = np.sin(phase)
                    voff   = vx*cosp + vy*sinp
                    voff   = np.reshape(voff,(len(voff),1))
                    nwave = len(grid.wave)
                    for nim in xrange(nwave):
                        w = grid.wave[nim]
                        g = grid.gamma[nim]
                        if nwave == 1:
                            s = 1.
                        else:
                            s = grid.scale[nim]
                        vel = (CKMS/w)*(data.data[nd].wave-w)-g-voff

                        # vel contains velocities of each pixel in current
                        # dataset. Now calculate gaussian, adding in with
                        # correct weight.
                        sigma = grid.vgrid*grid.fratio/EFAC
                        if ntdiv > 1 and (nt == 0 or nt == ntdiv - 1):
                            weight = (np.sqrt(2*np.pi)*sigma*grid.sfac*s)/(2*(ntdiv-1))
                        else:
                            weight = (np.sqrt(2*np.pi)*sigma*grid.sfac*s)/max(1,ntdiv-1)

                        A[nrow,noff:noff+vel.size] += weight * \
                            (np.exp(-(vel/sigma)**2/2) / data.data[nd].ferr).flat

                noff += vel.size
            nrow += 1

    noff = 0
    for nd in xrange(len(data.data)):
        dat = data.data[nd]
        b[noff:noff+vel.size] = (dat.flux / dat.ferr).flat
        noff += dat.flux.size

    # have matrices; beat into shape, compute Monroe-Penrose
    # pseudo-inverse (Api)
    A   = np.transpose(A)
    b   = np.reshape(b, (ndata,1))
    return (A,b)

def pinv(grid, data, cond, ntdiv):
    """
    Carries out SVD-based least-squares fit of a Grid to a Data object. See
    'svd' for a version of this allowing multiple values for the 'cond'
    parameter

    Arguments::

      grid : Grid object defining the fit model

      data : the data to fit to.

      cond : smallest singular value to keep, as ratio of largest
              see numpy.linalg.pinv for more. It should be < 1

      ntdiv : sub-division factor to spread the model within exposures using
              trapezoidal averaging

    Returns (coeff, chisq, cpn, rchisq) where:

      coeff : square array of fit coeffients identical in form
              to the data array of grid

      chisq : the chi**2 of the fit

      cpn   : chi**2 divided by the number of data points

      ndata  : number of data
    """
    if cond >= 1:
        raise DopplerError('trm.doppler.pinv: cond >= 1')

    A, b = genmat(grid, data, ntdiv)

    # compute Monroe-Penrose pseudo-inverse (Api)
    Api = np.linalg.pinv(A,cond)

    # x is the solution we want. 'fit' is fit to the data
    # based upon this fit.
    x   = np.dot(Api,b)
    fit = np.dot(A,x)

    # compute chi**2
    ndata = data.size
    chisq  = ((b-fit)**2).sum()
    cpn    = chisq/ndata
    nside = grid.data.shape[0]
    return (np.reshape(x,(nside,nside)), chisq, cpn, ndata)

def svd(grid, data, cond, ntdiv):
    """
    Carries out SVD-based least-squares fit of a Grid to a Data object. This
    acts much like pinv but in this case 'cond' can have multiple values and
    no grid is returned. This should be faster than calling pinv multiple
    times as the initial svd part is only carried out once. It is very
    comparable to pinv for just one value, perhaps even a tiny bit faster
    although there is little in it.

    Arguments::

      grid : Grid object defining the fit model

      data : the data to fit to.

      cond : value or series of values to condition the sngular values. If the
             values are < 1 they will be taken to indicate the smallest
             singular value to include as a ratio of the largest. If they are
             >= 1 they will be taken to indicate the number of the highest
             singular values to keep.

      ntdiv : sub-division factor to spread the model within exposures using
              trapezoidal averaging

    Returns (chisq, sing, s) where:

      chisq : chi**2 of the fit for each value of 'cond'.  This will be
              an array, even if 'cond' is a single float

      sing :  either the number of singular values used in each case (if
              corresponding cond < 1) or the smallest singular value used as a
              ratio of the largest (if corresponding cond >= 1)

      s     : the singular values.
    """

    # generate the matrices
    A, b = genmat(grid, data, ntdiv)

    # carry out full SVD. Return smallest matrices possible
    u, s, v = np.linalg.svd(A,full_matrices=False)
    v = np.transpose(v)
    u = np.transpose(u)

    # force cond to be an array
    cs = np.asarray(cond)
    if cs.ndim == 0:
        cs = np.array([float(cond),])
    smax  = abs(s[0])
    chisq = np.empty_like(cs)
    cred  = np.empty_like(cs)
    sing  = np.empty_like(cs)
    ndata = data.size

    # Go through each value of the conditioning numbers
    # calculate the Penrose-Monroe inverse, the fit coefficients
    # equivalent to this and finally the chi-squared.
    for i, c in enumerate(cs):

        # select the highest singular values with a method
        # determined by the value of the coniditioning number
        if c < 1.:
            nok     = (np.abs(s) > c*smax).sum()
            sing[i] = nok
        else:
            nok     = min(len(s), int(round(c)))
            sing[i] = s[nok-1]/s[0]

        snew = 1/s[:nok]

        # next line can be slow. snew*v is equivalent to
        # post-multiplying v by a diagonal matrix
        Api = np.dot(snew*v[:,:nok],u[:nok,:])

        # x is the solution we want. 'fit' is the fit based upon x
        x   = np.dot(Api,b)
        fit = np.dot(A,x)

        # compute chi**2 and reduced chi**2
        chisq[i] = ((b-fit)**2).sum()
        cred[i]  = chisq[i] / (ndata - nok)

    return (chisq, cred, sing, s)


