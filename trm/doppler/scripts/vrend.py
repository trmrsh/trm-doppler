#!/usr/bin/env python

import argparse
import numpy as np
from trm import doppler
import copy
from s2plot import *

def vrend(args=None):
    """vrend uses the s2plot 3D plotting package to provide a volume rendering of
    a 3D image. This will only work if s2plot and the python extension are
    installed of course

    """

    parser = argparse.ArgumentParser(description=vrend.__doc__)

    # positional
    parser.add_argument('map', help='name of the input map')
    parser.add_argument('vmax', type=float, help='maximum velocity (km/s)')

    # optional
    parser.add_argument(
        '-n', dest='nimage', type=int, default=1,
        help='number of the image to plot'
    )
    parser.add_argument(
        '-s', dest='step', type=int, default=1,
        help='pixel step to use. Make > 1 if the rendering seems slow and clunky'
    )

    # OK, done with arguments.
    args = parser.parse_args()

    dmap = doppler.Map.rfits(doppler.afits(args.map))
    img = dmap.data[args.nimage-1]

    # have to copy the array [perhaps to allow modification?]
    data = img.data.T[::args.step,::args.step,::args.step].copy()
    nx, ny, nz = data.shape

    # transform array
    tr = np.empty((12))
    tr[1], tr[6], tr[11] = args.step*img.vxy, args.step*img.vxy, \
                           args.step*img.vz
    tr[2] = tr[3] = tr[5] = tr[7] = tr[9] = tr[10] = 0

    tr[0] = -tr[1]*(nx-1)/2
    tr[4] = -tr[6]*(ny-1)/2
    tr[8] = -tr[11]*(nz-1)/2

    x1, x2 = tr[0], -tr[0]
    y1, y2 = tr[4], -tr[4]
    z1, z2 = tr[8], -tr[8]

    s2opendo("/s2mono")
    s2swin(-args.vmax, args.vmax, -args.vmax, args.vmax, -args.vmax, args.vmax)
    s2box("BCDETMNOQ",0,0,"BCDETMNOQ",0,0,"BCDETMNOQ",0,0)
    s2lab("Vx","Vy","Vz","")

    def cb(t, kc):
        """A dynamic callback"""
        global vro
        # Draw the volume render object
        ds2dvr(vro, 1)

    amb = {'r':0.8, 'g':0.8, 'b':0.}   # ambient light
    dmin = 0.0
    dmax = 1.0
    amin = 0.0
    amax = 0.8
    trans = 't'

    # Set colour range, install colour map
    s2scir(1000,2000)
    s2icm("mgreen",1000,2000)

    # Create the volume render object
    vro = ns2cvr(
        data, nx, ny, nz, 0, nx-1, 0, ny-1, 0, nz-1,
        tr, trans, dmin, dmax, amin, amax
    )

    # Install dynamic callback
    cs2scb(cb)

    # flat shading
    ss2srm(SHADE_FLAT)

    # ambient lighting
    ss2sl(amb, 0, None, None, 0)

    s2funuva(
        lambda u, v: u, lambda u, v: v, lambda u, v: 0, lambda u, v: 0.5,
         't', lambda u, v: 0.5, x1, x2, 100, y1, y2, 100
    )

    s2show(1)


