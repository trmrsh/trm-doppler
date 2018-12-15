#!/usr/bin/env python

import argparse
import numpy as np
from trm import doppler

def grid2map(args=None):
    usage = \
            """
            computes a map corresponding to a grid.
            """

    parser = argparse.ArgumentParser(description=usage)

    # positional
    parser.add_argument('grid',  help='name of the grid')
    parser.add_argument('tmap',  help='template map file')
    parser.add_argument('map',   help='output map')

    # OK, done with arguments.
    args = parser.parse_args()

    # load grid and data
    grid = doppler.Grid.rfits(doppler.afits(args.grid))
    tmap = doppler.Map.rfits(doppler.afits(args.tmap))

    nside = grid.data.shape[0]

    for image in tmap.data:
        dmap = image.data
        if dmap.ndim != 2:
            print('ERROR: Map images must be 2D for this routine.')
            exit(1)
        nvy, nvx = dmap.shape
        vx = np.linspace(-image.vxy*(nvx-1)/2., image.vxy*(nvx-1)/2., nvx)
        vy = np.linspace(-image.vxy*(nvy-1)/2., image.vxy*(nvy-1)/2., nvy)
        VX, VY = np.meshgrid(vx, vy)
        dmap[:] = 0
        efac = 1./(grid.vgrid*grid.fratio/doppler.EFAC)**2/2.
        for iy in range(nside):
            vy = grid.vgrid*(iy-(nside-1)/2.)
            for ix in range(nside):
                vx = grid.vgrid*(ix-(nside-1)/2.)
                dmap += grid.data[iy,ix]*np.exp(-efac*((VX-vx)**2+(VY-vy)**2))

    tmap.wfits(doppler.afits(args.map))

