// The hard work of Doppler imaging is done by the code here.

#include <stdlib>
#include <iostream>
#include "trm/subs.h"
#include "trm/constants.h"

/*
 * map   -- multiple image data, stored in a 1D array for compatibility with mem routines.
 * nxyz  -- dimensions of each image.
 * wave  -- wavelengths for each image.
 * gamma -- systemic velocities for each wavelength
 * scale -- scaling factors for each wavelength
 */

struct XYZ {
    size_t nx, ny, nz;
};

void op(const float map[], const vector<XYZ>& nxyz, const vector<vector<double> >& wave, 
        const vector<vector<double> >& gamma, const vector<vector<double> >& scale, 
        const vector<double>& vxy, 
        float vpix, 
        float fwhm, int ndiv, int ntdiv, int npixd, int nspec, 
        float vpixd, double waved, const Subs::Array1D<double>& time, 
        const Subs::Array1D<float>& expose, double tzero, double period, float data[]){

  int i, j, l, off;
  
  const int nfine = ndiv*npixd;   // number of pixels in fine pixel buffer.
  double fine[nfine], tfine[nfine]; // fine buffers

  // blurr array stuff
  const int nblurr = int(3.*ndiv*fwhm/vpixd);
  const int nbtot  = 2*nblurr+1;
  float blurr[nbtot], sigma = fwhm/Constants::EFAC;
  float efac = Subs::sqr(vpixd/ndiv/sigma)/2.;  
  double sum=0.;
  int k;
  for(k = -nblurr; k<= nblurr; k++)
    sum += (blurr[nblurr+k] = exp(-efac*k*k));

  for(k=0; k< nbtot; k++) 
    blurr[k] /= sum;


  float scale  = ndiv*vpix/vpixd; // scale factor map/fine
  float pxscale, pyscale;         // projected scale factors
  double phase, cosp, sinp;       // phase, cosine and sine.
  float fpcon;                    // fine pixel offset

  size_t yp, xp;
  int np;
  float fpoff, weight;

  // Loop through spectra
  for(int ns=0, doff=0; ns<nspec; ns++){

    // This initialisation is needed per spectrum
    for(k=0; k<nfine; k++) fine[k] = 0.;

    for(int nt=0; nt<ntdiv; nt++){

      // This initialisation is needed per sub-spectrum
      for(k=0; k<nfine; k++) tfine[k] = 0.;

      // Compute phase over uniformly spaced set from start to end of exposure. Times assumed
      // to be mid-exposure
      phase = (time[ns]+expose[ns]*(float(nt)-float(ntdiv-1)/2.)/std::max(ntdiv-1,1)-tzero)/period;
      cosp  = cos(Constants::TWOPI*phase);
      sinp  = sin(Constants::TWOPI*phase);

      pxscale = -scale*cosp;
      pyscale =  scale*sinp;

      // Loop over images
      for(int nwave=0, moff=0; nwave<wave.size(); nwave++){
	for(int ngamma=0; ngamma<gamma.size(); ngamma++){
	  
	  // Compute fine pixel offset factor. This shows where
	  // to add in to the fine pixel array. C = speed of light
	  // Two other factor account for the centres of the arrays
	  
	  fpcon = ndiv*((npixd-1)/2. + gamma[ngamma]/vpixd + 
			Constants::C*1.e-3*(1.-waved/wave[nwave]))
	    -scale*(-cosp+sinp)*(nside-1)/2. + 0.5;
	  
	  // Finally carry out projection
	  for(yp=0; yp<nside; yp++, fpcon += pyscale){
	    for(xp=0, fpoff=fpcon; xp<nside; xp++, moff++, fpoff+=pxscale){
	      np  = int(floor(fpoff));
	      if(np >= 0 && np < nfine) tfine[np] += map[moff];
	    }
	  }    
	}  
      }

      // Now add in with correct weight to fine buffer
      // The xpix squared factor is to give a similar intensity
      // regardless of the pixel size. i.e. the pixel values are
      // per 10^4 (km/s)**2
      if(ntdiv > 1 && (nt == 0 || nt == ntdiv - 1)){
	weight = Subs::sqr(vpix/100.)/(2*std::max(1,ntdiv-1));
      }else{
	weight = 2.*Subs::sqr(vpix/100.)/(2*std::max(1,ntdiv-1));
      }
      for(k=0; k<nfine; k++) fine[k] += weight*tfine[k];
    }

    // Blurr and bin into output spectrum
    for(i=0; i<npixd; i++){
      sum = 0.;
      off = ndiv*i;
      for(l=off; l<off+ndiv; l++){
	for(k = 0, j=l-nblurr; k<nbtot; k++, j++)
	  if(j >= 0 && j < nfine) sum += blurr[k]*fine[j];
      }
      data[doff++] = sum;
    }
  }
}

void Tomog::tr(const float data[], const Subs::Array1D<double>& wave, 
		 const Subs::Array1D<float>& gamma, size_t nside, float vpix, 
		 float fwhm, int ndiv, int ntdiv, int npixd, int nspec, float vpixd, 
		 double waved, const Subs::Array1D<double>& time, const Subs::Array1D<float>& expose, 
		 double tzero, double period, float map[]){
  
  
  int i, j, l, off;

  const int nfine = ndiv*npixd;   // number of pixels in fine pixel buffer.
  double fine[nfine], tfine[nfine]; // fine buffers
  
  // blurr array stuff
  const int nblurr = int(3.*ndiv*fwhm/vpixd);
  const int nbtot = 2*nblurr+1;
  float blurr[nbtot], sigma = fwhm/Constants::EFAC;
  float efac = Subs::sqr(vpixd/ndiv/sigma)/2., add;
  
  int k;
  double sum = 0.;
  for(k = -nblurr; k<= nblurr; k++)
    sum += (blurr[nblurr+k] = exp(-efac*k*k));

  for(k=0; k< nbtot; k++)
    blurr[k] /= sum;


  // Loop through spectra
  float scale  = ndiv*vpix/vpixd; // scale factor map/fine
  float pxscale, pyscale;         // projected scale factors
  double phase, cosp, sinp;       // phase, cosine and sine.
  float fpcon;             // fine pixel offset

  size_t xp, yp;
  int np;
  float fpoff, weight;

  for(int nwave=0, moff=0; nwave<wave.size(); nwave++){
    for(int ngamma=0; ngamma<gamma.size(); ngamma++){
      for(xp=0; xp<nside; xp++){
	for(yp=0; yp<nside; yp++){
	  map[moff++] = 0.;
	}
      }
    }
  }
  
  for(int ns=0, doff=0; ns<nspec; ns++){

    // This initialisation is needed per spectrum
    for(k=0; k<nfine; k++) fine[k] = 0.;

    // Transpose of blurr and bin section
    for(i=0; i<npixd; i++){
      add = data[doff++];
      off = ndiv*i;
      for(l=off; l<off+ndiv; l++){
	for(k = 0, j=l-nblurr; k<nbtot; k++, j++)
	  if(j >= 0 && j < nfine) fine[j] += blurr[k]*add;
      }
    }

    // Now finite exposure loop
    for(int nt=0; nt<ntdiv; nt++){

      // Compute phase over uniformly spaced set from start to end of exposure. Times assumed
      // to be mid-exposure
      phase = (time[ns]+expose[ns]*(float(nt)-float(ntdiv-1)/2.)/std::max(ntdiv-1,1)-tzero)/period;

      cosp  = cos(Constants::TWOPI*phase);
      sinp  = sin(Constants::TWOPI*phase);

      pxscale = -scale*cosp;
      pyscale =  scale*sinp;

      // Now add in with correct weight to fine buffer
      // The xpix squared factor is to give a similar intensity
      // regardless of the pixel size. i.e. the pixel values are
      // per 10^4 (km/s)**2
      if(ntdiv > 1 && (nt == 0 || nt == ntdiv - 1)){
	weight = Subs::sqr(vpix/100.)/(2*std::max(1,ntdiv-1));
      }else{
	weight = 2.*Subs::sqr(vpix/100.)/(2*std::max(1,ntdiv-1));
      }

      for(k=0; k<nfine; k++) tfine[k] = weight*fine[k];

      // Transpose of projection section
      for(int nwave=0, moff=0; nwave<wave.size(); nwave++){
	for(int ngamma=0; ngamma<gamma.size(); ngamma++){
	  
	  // Compute fine pixel offset factor. This shows where
	  // to add in to the fine pixel array. C = speed of light
	  // Two other factor account for the centres of the arrays
	  
	  fpcon = ndiv*((npixd-1)/2. + gamma[ngamma]/vpixd + 
			Constants::C*1.e-3*(1.-waved/wave[nwave]))
	    -scale*(-cosp+sinp)*(nside-1)/2. + 0.5;
	  
	  for(yp=0; yp<nside; yp++, fpcon+=pyscale){
	    for(xp=0, fpoff=fpcon; xp<nside; xp++, moff++, fpoff+=pxscale){
	      np  = int(floor(fpoff));
	      if(np >= 0 && np < nfine) map[moff] += tfine[np];
	    }
	  }
	}      
      }
    }
  }
}

// Computes gaussian default image. This blurrs by fwhm pixels
// in the x and y directions and gfwhm in the z (gamma) direction.
// It uses FFT/inverse-FFTs to carry out the blurring.

void Tomog::gaussdef(const float input[], size_t nwave, size_t ngamma, 
		       size_t nside, float fwhm, float gfwhm, float output[]){

  // copy input to output
  size_t npix = nside*nside;
  for(size_t i=0; i<nwave*ngamma*npix; i++) 
    output[i] = input[i];

  // Work out buffer size needed for image FFTs
  size_t naddi = 2*int(3.*fwhm+1.); 
  size_t ntoti = nside + naddi;
  ntoti  = size_t(pow(2.,int(log(float(ntoti))/log(2.))+1));

  // Work out buffer size needed for gamma FFTs
  size_t naddg = 2*int(3.*gfwhm+1.); 
  size_t ntotg = ngamma + naddg;
  ntotg  = size_t(pow(2.,int(log(float(ntotg))/log(2.))+1));

  // grab space for larger of two 
  float* work1 = new float[std::max(ntoti,ntotg)];
  float* work2 = new float[std::max(ntoti,ntotg)];

  // Prepare transform of gaussian convolution function for
  // image FFTs
  float efaci   = Subs::sqr(Constants::EFAC/fwhm)/2.;
  size_t i;
  for(i=0; i<ntoti; i++) work2[i] = 0.;
  float z, sum;
  work2[0] = sum = 1.;
  for(i=1; i<naddi/2; i++){
    z = exp(-efaci*i*i);
    work2[i] = work2[ntoti-i] = z;
    sum += 2.*z;
  }
  sum *= ntoti/2;
  for(i=0; i<ntoti; i++) work2[i] /= sum;
  Subs::fftr(work2,ntoti,1);

  size_t nim, off, toff, ttoff, ix, iy, n1, i1, i2, ind;
  float a, b, c, d;

  // Blurr in X
  n1 = (ntoti-nside)/2;
  for(nim=0, off=0; nim<nwave*ngamma; nim++, off += npix){

    for(iy=0, toff=off; iy<nside; iy++, toff += nside){
	
      // Load up data extending at ends by end values
      for(i=0; i<ntoti; i++){
	if(i > n1){
	  ind = std::min(i-n1, nside-1);
	}else{
	  ind = 0;
	}
	work1[i] = output[toff+ind];
      }
      
      // FFT
      Subs::fftr(work1,ntoti,1);
      
      // Multiply by convolving FFT
      work1[0] *= work2[0];
      work1[1] *= work2[1];
      for(i=0; i<ntoti/2-1; i++){
	i1 = 2*i+2;
	i2 = i1+1;
	a  = work2[i1];
	b  = work2[i2];
	c  = work1[i1];
	d  = work1[i2];
	work1[i1] = a*c-b*d;
	work1[i2] = a*d-b*c;
      }
      
      // inverse FFT, store in output
      Subs::fftr(work1,ntoti,-1);
      
      for(i=n1,ttoff=toff; i<nside+n1; i++,ttoff++) 
	output[ttoff] = work1[i];
    }
  }

  // Blurr in Y
  for(nim=0, off=0; nim<nwave*ngamma; nim++, off += npix){

    for(ix=0, toff=off; ix<nside; ix++, toff++){
      for(i=0; i<ntoti; i++){
	if(i > n1){
	  ind = std::min(i-n1, nside-1);
	}else{
	  ind = 0;
	}
	work1[i] = output[toff+nside*ind];
      }
      
      // FFT
      Subs::fftr(work1,ntoti,1);
      
      // Multiply by convolving FFT
      work1[0] *= work2[0];
      work1[1] *= work2[1];
      for(i=0; i<ntoti/2-1; i++){
	i1 = 2*i+2;
	i2 = i1+1;
	a  = work2[i1];
	b  = work2[i2];
	c  = work1[i1];
	d  = work1[i2];
	work1[i1] = a*c-b*d;
	work1[i2] = a*d-b*c;
      }
      
      // inverse FFT, store in output
      Subs::fftr(work1,ntoti,-1);
      
      for(i=n1,ttoff=toff; i<nside+n1; i++,ttoff+=nside) 
	output[ttoff] = work1[i];
    }
  }

  // Blurr in gamma
  if(ngamma > 1){

    // Prepare transform of gaussian convolution function for
    // gamma FFTs
    float efacg   = Subs::sqr(Constants::EFAC/gfwhm)/2.;
    
    for(i=0; i<ntotg; i++) work2[i] = 0.;
    work2[0] = sum = 1.;
    for(i=1; i<naddg/2; i++){
      z = exp(-efacg*i*i);
      work2[i] = work2[ntotg-i] = z;
      sum += 2.*z;
    }
    sum *= ntotg/2;
    for(i=0; i<ntotg; i++) work2[i] /= sum;
    Subs::fftr(work2,ntotg,1);
    
    // carry out gamma FFTs
    n1 = (ntotg-ngamma)/2;
    size_t nw, ip;
    for(nw=0, off=0; nw<nwave; nw++, off += npix*ngamma){  
      for(ip=0, toff=off; ip<npix; ip++, toff++){
	for(i=0; i<ntotg; i++){
	  if(i > n1){
	    ind = std::min(i-n1, ngamma-1);
	  }else{
	    ind = 0;
	  }
	  work1[i] = output[toff+npix*ind];
	}
	  
	// FFT
	Subs::fftr(work1,ntotg,1);
	
	// Multiply by convolving FFT
	work1[0] *= work2[0];
	work1[1] *= work2[1];
	for(i=0; i<ntotg/2-1; i++){
	  i1 = 2*i+2;
	  i2 = i1+1;
	  a  = work2[i1];
	  b  = work2[i2];
	  c  = work1[i1];
	  d  = work1[i2];
	  work1[i1] = a*c-b*d;
	  work1[i2] = a*d-b*c;
	}
	
	// inverse FFT, store in output
	Subs::fftr(work1,ntotg,-1);
	
	for(i=n1, ttoff=toff; i<ngamma+n1; i++, ttoff += npix) 
	  output[ttoff] = work1[i];

      }
    }
  }

  // clean up
  delete[] work1;
  delete[] work2;
}

    




