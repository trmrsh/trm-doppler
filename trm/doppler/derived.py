#!/usr/bin/env python

"""
Routines that requires 2 or more of the other sub-packages
"""

from __future__ import absolute_import

import sys
import numpy as np
from scipy import linalg

from .data import *
from .map import *
from .grid import *

def genmat(grid, data, ntdiv):
    """Computes the matrix A when representing the Doppler image problem by A x = b

    Arguments::

      grid : the Grid defining the fit model

      data : the Data defining the data

      ntdiv : sub-division factor to spread the model within exposures using
              trapezoidal averaging. Note that only a single value can be
              specified for technical reasons. One tends to get oscillations
              in chi**2 for small ntdiv, so values of order 10 are
              recommended.

    Returns::

      A : NxM matrix where N is number of data and M is number of grid
          points

    """

    ndata, ngrid = data.size, grid.size

    nside = grid.data.shape[0]

    # Reserve space for matrix
    A = np.empty((ngrid, ndata))

    vgrid = grid.vgrid

    nrow = 0
    for iy in xrange(nside):
        vy = vgrid*(iy-(nside-1)/2.)
        for ix in xrange(nside):
            vx = vgrid*(ix-(nside-1)/2.)
            noff = 0
            for nd in xrange(len(data.data)):
                dat = data.data[nd]
                tflx = np.zeros_like(dat.flux)
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

                        # compute velocities of each pixel in current
                        # dataset, scale by gaussian RMS, folding in the
                        wv    = data.data[nd].wave
                        vel   = (CKMS/w)*(wv-w)-(g-voff)
                        sigma = grid.vgrid*grid.fratio/EFAC
                        vel  /= sigma

                        # compute sub-division weight
                        if ntdiv > 1 and (nt == 0 or nt == ntdiv - 1):
                            weight = (np.sqrt(2*np.pi)*sigma*grid.sfac*s)/(2*(ntdiv-1))
                        else:
                            weight = (np.sqrt(2*np.pi)*sigma*grid.sfac*s)/max(1,ntdiv-1)

                        # add into temporary array, with a restriction to < 6 sigma
                        # to speed things a little
                        ok = np.abs(vel) < 6.
                        tflx[ok] += weight*np.exp(-vel[ok]**2/2.)

                A[nrow,noff:noff+vel.size] = (tflx / data.data[nd].ferr).flat

                noff += vel.size
            nrow += 1

    # have matrices. beat into shape and return
    A   = np.transpose(A)

    return A

def genvec(data):
    """
    Computes the vector b when representing the Doppler image problem by A x = b

    Arguments::

      data : the Data defining the data

    Returns::

      b : column vector dimensions Nx1 matrix where N is number of data

    """

    ndata = data.size

    # Reserve space for vector
    b = np.empty((ndata))

    noff = 0
    for nd, spectra in enumerate(data.data):
        b[noff:noff+spectra.flux.size] = (spectra.flux / spectra.ferr).flat
        noff += spectra.flux.size

    # beat into shape and return
    b   = np.reshape(b, (ndata,1))
    return b

def svd(grid, data, cond, ntdiv, full_output=False):
    """Carries out SVD-based least-squares fit of a Grid to a Data object
    returning chi**2 values for each of several possible values of the
    parameter 'cond' which determines how many singular values are retained.

    Arguments::

      grid : Grid object defining the fit model

      data : the data to fit to.

      cond : value or series of values to condition the sngular values. If the
             values are < 1 they will be taken to indicate the smallest
             singular value to include as a ratio of the largest. If they are
             >= 1 they will be taken to indicate the number of the highest
             singular values to keep (rounded to nearest integer in this
             case).  'cond' will be limited to a maximum set by the number of
             grid points.

      ntdiv : sub-division factor to spread the model within exposures using
              trapezoidal averaging. This reduces a tendency of chi**2
              oscillating when plotted against period.

      full_output : if True, the best fit vectors are also returned, see 'x'
                    below.

    Returns (chisq, cred, sing, s, [x]) where:

      chisq : chi**2 of the fit for each value of 'cond'.  This will be an
              array, even if 'cond' is a single float

      cred : reduced chi**2 where number of degrees of freedom = ndata -
             number of singular values used.

      sing : either the number of singular values used in each case (if the
             corresponding cond value is < 1) or the smallest singular value
             used as a ratio of the largest (if corresponding cond >= 1).

      s     : the singular values.

      x : Optional list of best-fit vectors for each value of cond. Only
          returned if the full_output flag is set.

    """

    # generate the matrix and vector
    A = genmat(grid, data, ntdiv)
    b = genvec(data)

    if A.shape[0] < A.shape[1]:
        raise DopplerError('ERROR: trm.doppler.svd -- more grid points than data')

    # carry out full SVD. Return smallest matrices possible
    # This is the slowest step of the program. scipy version
    # a tiny bit faster than numpy's
    u, s, v = linalg.svd(A,full_matrices=False)

    # we need the transposes later
    v = np.transpose(v)
    u = np.transpose(u)

    # force cond to be an array
    cs = np.asarray(cond)
    if cs.ndim == 0:
        cs = np.array([float(cond),])
    smax  = s[0]
    chisq = np.empty_like(cs)
    cred  = np.empty_like(cs)
    sing  = np.empty_like(cs)
    ndata = data.size
    nside = grid.data.shape[0]

    # optional return of best-fit vectors
    if full_output: xs = []

    # Go through each value of the conditioning numbers
    for i, c in enumerate(cs):

        # select the highest singular values with a method
        # determined by the value of the coniditioning number
        if c < 1.:
            nok     = (s > c*smax).sum()
            sing[i] = nok
        else:
            nok     = min(len(s), int(round(c)))
            sing[i] = s[nok-1]/s[0]

        # snew contains the inverses of the largest SVD values
        snew = 1/s[:nok]

        # we now want to calculate x = v*diag(snew)*u*b
        # We calculate this as (v*diag(snew))*(u*b)
        # for speed.
        x   = np.dot(snew*v[:,:nok],np.dot(u[:nok,:],b))

        # the fit to the data corresponding to x ...
        fit = np.dot(A,x)

        # compute chi**2 and reduced chi**2, save the grid
        # image if wanted.
        chisq[i] = ((b-fit)**2).sum()
        cred[i]  = chisq[i] / (ndata - nok)

        if full_output:
            xs.append(np.reshape(x,(nside,nside)))

    if full_output:
        return (chisq, cred, sing, s, xs)
    else:
        return (chisq, cred, sing, s)
