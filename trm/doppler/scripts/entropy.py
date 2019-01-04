import argparse
from trm import doppler
import copy
import numpy as np

def entropy(args=None):
    """
    entropy -- iterates to minimum chi**2 for a fixed entropy
    experimental!
    """

    parser = argparse.ArgumentParser(description=entropy.__doc__)

    # positional
    parser.add_argument('imap',  help='name of the input map')
    parser.add_argument('data',  help='data file')
    parser.add_argument('niter', type=int, help='number of iterations')
    parser.add_argument('saim',  type=float, help='entropy to aim for')
    parser.add_argument('omap',  help='name of the output map')

    # optional
    parser.add_argument('-r', dest='rmax', type=float,
                        default=0.2, help='maximum change')
    parser.add_argument(
        '-t', dest='tlim', type=float,
        default=1.e-4, help='test limit for stopping iterations'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    # load map and data
    dmap = doppler.map.Map.rfits(doppler.afits(args.imap))
    data = doppler.data.Data.rfits(doppler.afits(args.data))

    # metric with both indices in contra-variant "up" positions for taking
    # one-forms to vectors
    umetric = dmap

    # copy the data, compute model data [in mdata]
    mdata = copy.deepcopy(data)
    doppler.comdat(dmap, mdata)

    # subtract data
    resid = mdata - data

    # calculate chi**2/ndata
    C = doppler.data.inner_product(resid,resid)/2
    print('C =',C)

    # normalise by variances, *2/ndata
    resid.vnorm()

    # apply tropus to the result to get chi**2 gradient one-form,
    # factor 2 needed for square in chi**2
    gradc = copy.deepcopy(dmap)
    doppler.datcom(resid, gradc)

    # compute its length-squared
    gradc_lsq = doppler.map.inner_product(
        gradc, gradc, umetric, False
    )

    # copy the map, compute default [in grads]
    grads = copy.deepcopy(dmap)
    doppler.comdef(grads)

    # take log of default / map -- this is entropy gradient one-form
    grads /= dmap
    grads.log()
    S = (dmap*(grads+1)).sumf()
    print('S =',S)

    # compute its length-squared
    grads_lsq = doppler.map.inner_product(
        grads, grads, umetric, False
    )

    # take difference between length-normalised one-forms
    diff = gradc/np.sqrt(gradc_lsq) - grads/np.sqrt(grads_lsq)

    # report TEST
    print('TEST =',doppler.map.inner_product(diff,diff,umetric,False)/2)

    # Three basic search directions which are the map itself, and the vector
    # forms of the gradients. The map itself is already a vector and needs no
    # conversion.
    search_dirns = [dmap, umetric*grads, umetric*gradc]
    ndirn = len(search_dirns)

    # entropy sub-space curvatures
    sdd = np.empty((ndirn,ndirn))

    # calculate inner products of search directions with metric
    # since the metric is grad(grad(S))
    for i, sdi in enumerate(search_dirns):
        for j, sdj in enumerate(search_dirns[:i+1]):
            sdd[j,i] = sdd[i,j] = doppler.map.inner_product(
                sdi, sdj, umetric
            )

    print(sdd)

    # chi**2/N sub-space curvatures
    cdd = np.empty((ndirn,ndirn))

    # compute first data space search directions (first already
    # calculated)
    data_search_dirns = [mdata,]
    for sdi in search_dirns[1:]:
        tdata = copy.deepcopy(data)
        doppler.comdat(sdi, tdata)
        data_search_dirns.append(tdata)

    # inner products
    for i, dsdi in enumerate(data_search_dirns):
        for j, dsdj in enumerate(data_search_dirns[:i+1]):
            cdd[j,i] = cdd[i,j] = \
                       doppler.data.inner_product(dsdi, dsdj)

    print(cdd)

    print('gsd0 =',sdd[:,1])
    print('gcd0 =',sdd[:,2])

    # By this point we have calculated the sub-space quadratic model
    # parameters for both S and C.

    # make matrix versions
    msdd = np.matrix(sdd)
    mcdd = np.matrix(cdd)

    diag_sdd = np.diagflat(1/np.sqrt(np.diag(msdd)))
    print(msdd,diag_sdd)

    # normalise
    nsdd = diag_sdd*msdd*diag_sdd
    ncdd = diag_sdd*mcdd*diag_sdd

    print('nsdd =',nsdd)
    svals, svecs = np.linalg.eigh(nsdd)
    print('S eigen values  =',svals)
    print('S eigen vectors =',svecs)

    nsrch = len(svals)
    svalmx = 3.e-5*svals[-1]

    # 'lowest' is the lowest eigenvalue to retain, and
    # corresponds to 'l' in meml3
    lowest = np.searchsorted(svals, svalmx)
    print('lowest =',lowest)
    ndim = nsrch - lowest

    # rotate cdd (lines 2378-2397 in memsys)
    # rcdd corresponds to w1
    rcdd = svecs[:,lowest:].T*ncdd*svecs[:,lowest:]
    print(rcdd)

    # squeeze rcdd to isotropise the now-diagonal sdd
    diag_sval = np.diagflat(1/np.sqrt(svals[lowest:]))
    rcdd = diag_sval*rcdd*diag_sval

    # equiv "meml33(ndim,w1,cval,w2);" in memsys
    # returns eigenvalues and vectors of rotated cdd matrix.
    cvals, cvec = np.linalg.eigh(rcdd)

    # equiv to lines following "complete squeeze of w2 back to 
    # sdd eigenvector space" in memsys [cdd standing in for 'w2']
    rcdd = rcdd*diag_sval

    print(rcdd)

