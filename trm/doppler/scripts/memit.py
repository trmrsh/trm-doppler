import argparse
from trm import doppler

def memit(args=None):
    """
    memit carries out MEM iterations on an image.
    """

    parser = argparse.ArgumentParser(description=memit.__doc__)

    # positional
    parser.add_argument('imap',  help='name of the input map')
    parser.add_argument('data',  help='data file')
    parser.add_argument('niter', type=int, help='number of iterations')
    parser.add_argument('caim',  type=float, help='reduced chi**2 to aim for')
    parser.add_argument('omap',  help='name of the output map')

    # optional
    parser.add_argument('-r', dest='rmax', type=float,
                        default=0.2, help='maximum change')
    parser.add_argument('-t', dest='tlim', type=float,
                        default=1.e-4, help='test limit for stopping iterations')

    # OK, done with arguments.
    args = parser.parse_args()

    # load map and data
    dmap = doppler.Map.rfits(doppler.afits(args.imap))
    data = doppler.Data.rfits(doppler.afits(args.data))

    # mem iterations
    doppler.memit(dmap, data, args.niter, args.caim, args.tlim, args.rmax)

    # write to fits file
    dmap.wfits(doppler.afits(args.omap))
