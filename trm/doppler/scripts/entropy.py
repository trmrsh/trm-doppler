import argparse
from trm import doppler
import copy
import numpy as np

def eigpd(mat):
    """
    Given symmetric input matrix mat, this first adds a diagonal
    matrix to make it positive definite, then returns eigenvalues
    and eigenvectors as a two element tuple. See meml33 in memsys.
    I have merely copied what is done there without understanding
    the precise details.
    """
    EPS=3.0e-16

    print('\n   input matrix =\n',mat)

    a = np.trace(mat)
    b = np.square(mat).sum()
    t = mat.shape[0]
    c = max(b-a**2/t,0.)
    c = a/t-np.sqrt(c)-EPS*np.sqrt(b)
    print('   a,b,c,t =',a,b,c,t)
    print('   eigenvalues and vectors =\n')

    # z is transformed matrix which should be positive definite
    z = mat - c*np.identity(mat.shape[0])

    # return z's eigenvalues and vectors
    vals, vecs = np.linalg.eigh(z)
    for n in range(z.shape[0]):
        print('   ',n,vals[n],'{',[float(v) for v in vecs[:,n]],'}')
    vls, vcs = np.linalg.eigh(mat)
    vls += c
    print(vls,vcs)

    return (vals, vecs)


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

    # entropy metric with both indices in contra-variant "up" positions for taking
    # one-forms to vectors, and its covariant partner. 'u' and 'd' are used repeatedly
    # later on to distinguish vector and one-form components.
    guu = dmap
    gdd = 1/dmap

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
    # "dgradc" ('d' for down covariant index). factor 2 needed for
    # square in chi**2
    dgradc = copy.deepcopy(dmap)
    doppler.datcom(resid, dgradc)

    # compute equivalent gradient (uses diagonal nature of metric)
    ugradc = guu*dgradc

    # compute length-squared
    gradc_lsq = doppler.map.inner_product(ugradc, dgradc)

    # copy the map, compute default
    defmap = copy.deepcopy(dmap)
    doppler.comdef(defmap)

    # take log(default / map) -- this is entropy gradient one-form
    dgrads = defmap / dmap
    dgrads.log()

    # entropy, as defined in memsys
    S = (dmap*(dgrads+1)).sumf()/defmap.sumf()-1
    print('S =',S)

    # compute its length-squared
    ugrads = guu*dgrads
    grads_lsq = doppler.map.inner_product(ugrads, dgrads)

    # take difference between length-normalised one-forms
    ddiff = dgradc/np.sqrt(gradc_lsq) - dgrads/np.sqrt(grads_lsq)

    # report TEST
    udiff = guu*ddiff
    print('TEST =',doppler.map.inner_product(udiff,ddiff)/2)

    # Three basic search directions which are the map itself and the gradients.
    # Store both vector and one-form components
    usearch_dirns = [dmap, ugrads, ugradc]
    dsearch_dirns = [gdd*dmap, dgrads, dgradc]
    ndirn = len(usearch_dirns)

    # entropy sub-space curvatures
    sdd = np.empty((ndirn,ndirn))

    # calculate inner products of search directions to develop quadratic
    # part of entropy model. Works because grad(grad(S)) is our metric.
    for i, usdi in enumerate(usearch_dirns):
        for j, dsdj in enumerate(dsearch_dirns[:i+1]):
            sdd[j,i] = sdd[i,j] = doppler.map.inner_product(usdi, dsdj)
    print('sdd =\n',sdd)

    # chi**2/N sub-space curvatures
    cdd = np.empty((ndirn,ndirn))

    # compute first data space search directions
    # (first already calculated)
    data_search_dirns = [mdata,]
    for sdi in usearch_dirns[1:]:
        tdata = copy.deepcopy(data)
        doppler.comdat(sdi, tdata)
        data_search_dirns.append(tdata)

    # inner products, no messing with metrics here
    for i, sdi in enumerate(data_search_dirns):
        for j, sdj in enumerate(data_search_dirns[:i+1]):
            cdd[j,i] = cdd[i,j] = doppler.data.inner_product(sdi, sdj)
    print('cdd =\n',cdd)

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

    print('\nS matrix')
    svals, svecs = eigpd(nsdd)

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

    # squeeze rcdd to isotropise the now-diagonal sdd
    diag_sval = np.diagflat(1/np.sqrt(svals[lowest:]))
    rcdd = diag_sval*rcdd*diag_sval
    print('squeezed w1 =\n',rcdd,'\n')

    # equiv "meml33(ndim,w1,cval,w2);" in memsys
    # returns eigenvalues and vectors of rotated cdd matrix.
    print('\nSqueezed rotated CDD matrix')
    cvals, cvec = eigpd(rcdd)

    # equiv to lines following "complete squeeze of w2 back to 
    # sdd eigenvector space" in memsys [cdd standing in for 'w2']
    cvec = cvec*diag_sval
    print('squeezed w2 =\n',cvec,'\n')

