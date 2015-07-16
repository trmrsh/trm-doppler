'''
Subroutines for new Doppler tomography code 
'''

import os
import sys
import subprocess
from trm import doppler, dnl
from trm.dnl import molly
import numpy as np
import matplotlib.pyplot as plt

# Plot a trailed spectrogram

def plot_trail(FITS_datafile, cmap=plt.cm.binary):
 # Read data in
 HDU_data = doppler.Data.rfits(FITS_datafile)
 flux = HDU_data.data[0].flux
 wave = HDU_data.data[0].wave[0,:]
 extent = (wave[0],wave[-1],0.5,flux.shape[0]+0.5)
 plt.imshow(flux,origin='lower', aspect='auto',interpolation='nearest',extent=extent,cmap=cmap)
 plt.xlabel(r'Wavelength', fontsize = 18)
 plt.ylabel(r'Spectrum', fontsize = 18) 
 plt.show()


# Plot a PUNIT map

def plotmap_PUNIT(mapfile, im = 0, cmap=plt.cm.jet, plot=True):
 '''
 Plot a positive 2D image  
 '''
 dmap = doppler.Map.rfits(mapfile)
 if im > len(dmap.data)-1:
  print 'not a valid image...'
  return
 vxy = dmap.data[0].vxy
 nxy = dmap.data[0].data.shape[0]
 Pimage = dmap.data[im].data
 vmax=(nxy-1)*vxy/2. 
 range=[-vmax-vxy/2.,vmax+vxy/2.,-vmax-vxy/2.,vmax+vxy/2.]
 if plot:
  plt.imshow(Pimage, extent=range, interpolation='nearest',aspect=1,cmap=cmap, origin='lower')
  plt.xlabel('$\mathrm{V_{x}}$ $\mathrm{(km/s)}$', fontsize = 18)
  plt.ylabel('$\mathrm{V_{y}}$ $\mathrm{(km/s)}$', fontsize = 18)
  plt.title(mapfile, fontsize = 16, y=1.02)
  plt.show() 
 print 'vpix = ', vxy
 print 'npix = ', nxy
 return Pimage 
 

# Compute a difference map

def diff_map(map1, map2, diffmap):
 '''
 Write out a difference map
 '''
 inmap1 = doppler.Map.rfits(map1)
 inmap2 = doppler.Map.rfits(map2)
 for d1, d2 in zip(inmap1.data, inmap2.data):
  d1.data -= d2.data
 
 inmap1.wfits(diffmap) 


# Make bootstrap data

def boot(originalData, bootData, multiplier=-1):
 '''
 Make a bootstrap copy of data
 '''
 print 'Generating boostrap data...'
 # Read data in
 data_o = doppler.Data.rfits(originalData)
 for dat in data_o.data:
  ndata = dat.ferr.size
  pick = np.random.multinomial(ndata, [1/float(ndata)]*ndata, size=1).reshape(dat.ferr.shape) 
  # Change uncertainties on data points selected more than once
  chg = (pick > 1) & (dat.ferr > 0.)
  dat.ferr[chg] /= np.sqrt(pick[chg])
  # change errors on any not selected to negative
  mask = (pick == 0) & (dat.ferr > 0.)
  dat.ferr[mask] *= multiplier
 
 # Write out the data
 data_o.wfits(bootData) 


# Run a doppler2/makemap process

def dop2_makemap(startmconfig, startm, logfile):
 '''
 Make a starting image
 '''
 if os.path.exists(startm):
  print 'Filename exists...'
  return
 else: 
  f = open(logfile,'a')
  print 'Making start image...'
  p = subprocess.Popen('makemap.py '+startmconfig+' '+startm+"\n", shell=True, stdout=f,stderr=subprocess.STDOUT)
  p.wait()
  f.close()


# Run a doppler2/optscl process

def dop2_optscl(startmap, data, optsclmap, logfile):
 '''
 Scale the starting image
 '''
 f = open(logfile,'a')
 print 'Scaling of starting image...'
 p = subprocess.Popen('optscl.py '+startmap+' '+data+' '+optsclmap+"\n", shell=True, stdout=f,stderr=subprocess.STDOUT)
 p.wait()
 f.close()


# Run a doppler2/memit process

def dop2_memit(inmap, data, niter, caim, reconmap, logfile):
 '''
 Carry out MEM iterations
 '''
 f = open(logfile,'a')
 print 'Running memit...'
 p = subprocess.Popen('memit.py -t 1.e-20 '+inmap+' '+data+' '+str(niter)+' '+str(caim)+' '+reconmap+"\n", shell=True, stdout=f,stderr=subprocess.STDOUT)
 p.wait()
 f.close()

  
