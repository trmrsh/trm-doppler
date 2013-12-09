// The hard work of Doppler imaging is done by the code here. Much of code
// (~1000 lines or so) is unedifying "boilerplate" for interfacing Python and
// C++, and in particular to the C++ mem routines. The minimal number of
// routines needed to do this is provided. Most manipulations of the objects
// are better left to Python. The main routines that do the hard work in mem
// are op and tr that carry out the image to data projection and its
// transpose.
//
// The data, structure and routines in this file are as follows::
//  
//  CKMS        : speed of light in km/s (data)
//  EFAC        : ratio of fwhm/sigma for a gaussian (data)
//  Nxyz        : structure for image dimensions (struct)
//  op          : image to data transform routine. (func)
//  tr          : data to image (transpose of op) routine. (func)
//  npix_map    : calculates the number of pixels needed for the images (func)
//  read_map    : makes internal contents of a Map object accessible from C++ (func)
//  update_data : overwrites image array(s) of a Map object. (func)
//  npix_data   : calculates the number of pixels needed for fluxes, errors,
//                wave (func)
//  read_data   : makes internal contents of a Data object accessible from C++ (func)
//  update_data : overwrites flux array(s) of a Data object. (func)
//  doppler_comdat : compute data equivalent to an image (func)

#include <Python.h>
#include <iostream>
#include <vector>
#include "numpy/arrayobject.h"
#include <cmath>
#include <complex.h>
#include <fftw3.h>
#include "trm/memsys.h"

// Speed of light in km/s
const double CKMS = 299792.458;

// Ratio FWHM/sigma for a gaussian
const double EFAC = 2.354820045;

// simple structure for containing image dimensions
struct Nxyz{
    Nxyz(size_t nx, size_t ny, size_t nz=1) : nx(nx), ny(ny), nz(nz) {}
    size_t ntot() const {return nx*ny*nz;}

    size_t nx, ny, nz;
};

/* op computes the image to data transform that is the heart of Doppler
 * tomography. i.e it projects the image(s) to compute data corresponding to
 * each spectrum.
 *
 * Arguments::
 *
 *  image  : multiple images, stored in a 1D array for compatibility with
 *           mem routines (input)
 *  nxyz   : dimensions of each image, one/image, (input)
 *  vxy    : Vx-Vy pixel size(s), one/image, (input)
 *  vz     : Vz pixel size(s), one/image, (input)
 *  wavel  : central wavelengths for each image, vector/image, (input)
 *  gamma  : systemic velocities for each wavelength, vector/image, (input)
 *  scale  : scaling factors for each wavelength, vector/image, (input)
 *  tzero  : ephemeris zero point, same units as the times (input)
 *  period : ephemeris period, same units as the times (input)
 *  vfine  : km/s to use for internal fine array (input)
 *  vpad   : km/s to allow at each end of fine array beyond maximum extent
 *           implied by images. (input)
 *  data   : computed data, again as a 1D array for mem routines (output)
 *  wave   : wavelength array, matching the data array, (input)
 *  nwave  : number of wavelengths, one/dataset [effectively an X dimension],
 *           (input)
 *  nspec  : number of spectra, one/dataset [effectively a Y dimension], (input)
 *  time   : central times for each spectrum (input)
 *  expose : exposure lengths for each spectrum (input)
 *  ndiv   : sub-division factors for each spectrum (input)
 *  fwhm   : FWHM resolutions, kms/s, one/dataset, (input)
 *
 * Note the arguments 'image' through to 'vpad' are associated with the
 * Doppler map file, while the rest are associated with the data. 'data' is
 * the only output argument.
 */

void op(const float* image, const std::vector<Nxyz>& nxyz,
        const std::vector<double>& vxy, const std::vector<double>& vz,
        const std::vector<std::vector<double> >& wavel,
        const std::vector<std::vector<float> >& gamma,
        const std::vector<std::vector<float> >& scale,
        double tzero, double period, double vfine, double vpad,
        float* data, const double* wave,
        const std::vector<size_t>& nwave, const std::vector<size_t>& nspec,
        const std::vector<std::vector<double> >& time,
        const std::vector<std::vector<float> >& expose,
        const std::vector<std::vector<int> >& ndiv,
        const std::vector<double>& fwhm){

    // Each image is first projected onto a finely-spaced array which is later
    // blurred. This should be faster than blurring each image pixel since
    // multiple image pixels end in the same fine array pixel provided it is
    // not made too fine. Start by computing the size of the fine array
    size_t nimage = nxyz.size(), nx, ny, nz;
    double vmax = 0.;
    for(size_t nim=0; nim<nimage; nim++){
        nx = nxyz[nim].nx;
        ny = nxyz[nim].ny;
        nz = nxyz[nim].nz;
        if(nz > 1){
            vmax = std::max(std::sqrt(std::pow(vxy[nim],2)
                                      *(std::pow(nx,2)+std::pow(ny,2))+
                                      std::pow(vz[nim]*nz,2)), vmax);
        }else{
            vmax = std::max(std::sqrt(std::pow(vxy[nim],2)
                                      *(std::pow(nx,2)+std::pow(ny,2))), vmax);
        }
    }
    vmax += 2.*vpad;

    // NFINE is the number of pixels needed for the fine array.
    const int nfine = std::max(5,int(std::ceil(vmax/vfine)));

    // adjust vmax to be the maximum velocity of the fine array (from zero not
    // full extent)
    vmax = nfine*vfine/2.;

    // Now we need to know how many pixels at most are needed for the blurring
    // function. Loop over the data sets. Also take chance to compute number
    // of data pixels
    size_t nblurr = 0, ndpix = 0;
    double psigma;
    for(size_t nd=0; nd<nspec.size(); nd++){
        // sigma of blurring in terms of fine array pixels
        psigma = fwhm[nd]/vfine/EFAC;

        // total blurr width will be 2*nblurr-1
        nblurr = std::max(nblurr,size_t(round(6.*psigma)+1));
        ndpix += nwave[nd]*nspec[nd];
    }

    // The blurring is carried out with FFTs requiring zero padding.  Thus the
    // actual number of pixels to grab is the nearest power of 2 larger than
    // nfine+2*nblurr-1. Call this NFINE.
    const size_t NFINE =
        size_t(std::pow(2,int(std::ceil(std::log(nfine+2*nblurr-1)/
                                        std::log(2.)))));

    // grab memory for blurring array and associated FFTs
    double *blurr = new double[NFINE];

    // this for the FFT of the blurring array. This could in fact be done in
    // place, but the amount of memory required here is small, so it is better
    // to have an explicit dedicated array
    const size_t NFFT = NFINE/2+1;
    fftw_complex *bfft = (fftw_complex*) 
        fftw_malloc(sizeof(fftw_complex) * NFFT);

    // this for the FFT of the fine pixel array
    fftw_complex *fpfft = (fftw_complex*) 
        fftw_malloc(sizeof(fftw_complex) * NFFT);

    // Grab space for fine arrays
    double *fine  = new double[NFINE];
    double *tfine = new double[NFINE];

    // create plans for blurr, fine pixel and final inverse FFTs
    fftw_plan pblurr = fftw_plan_dft_r2c_1d(NFINE, blurr, bfft, 
                                            FFTW_ESTIMATE);
    fftw_plan pforw  = fftw_plan_dft_r2c_1d(NFINE, fine, fpfft, 
                                            FFTW_ESTIMATE);
    fftw_plan pback  = fftw_plan_dft_c2r_1d(NFINE, fpfft, fine, 
                                            FFTW_ESTIMATE);

    // various local variables
    size_t k, m, n;
    double norm, prf, fp1, fp2, v1, v2, sum;
    double wv, sc;
    float gm;

    // pixel index in fine array
    int nf, ifp1, ifp2;

    // phase, cosine, sine, weight factor
    double phase, cosp, sinp, weight;

    // image indices
    size_t ix, iy, iz;

    // image velocity steps xy and z
    double vxyi, vzi = 0.;

    // steps and offsets in fine pixel space
    double pxstep, pystep, pzstep, pxoff, pyoff, pzoff;

    // temporary image & data pointers
    const float *iptr, *iiptr;
    float *dptr;
    const double *wptr;

    // large series of nested loops coming up.
    // First zero the data array
    memset(data, 0, ndpix*sizeof(float));

    // Loop over each data set
    for(size_t nd=0; nd<nspec.size(); nd++){

        // compute blurring array (specific to the dataset
        // so can't be done earlier). End by FFT-ing it in order
        // to perform the convolution later.
        memset(blurr, 0, NFINE*sizeof(double));
        norm = blurr[0] = 1.;
        psigma = fwhm[nd]/vfine/EFAC;
        for(m=1, n=NFINE-1; m<nblurr; m++, n--){
            prf = std::exp(-std::pow(m/psigma,2)/2.);
            blurr[m] = blurr[n] = prf;
            norm += 2.*prf;
        }

        // Next line gets overall FFT/iFFT normalisation right
        // cutting out any need to normalise by NFINE later
        norm *= NFINE;

        // normalise
        for(m=0;m<NFINE;m++)
            blurr[m] /= norm;
        
        // take FFT
        fftw_execute(pblurr);

        // Loop over each image. iptr updated at end of loop
        iptr = image;
        for(size_t ni=0; ni<nxyz.size(); ni++){

            // Calculate whether we can skip this particular
            // dataset / image combination ??
            // ...
            
            // extract image dimensions and velocity scales
            nx   = nxyz[ni].nx;
            ny   = nxyz[ni].ny;
            nz   = nxyz[ni].nz;
            vxyi = vxy[ni];
            if(nz > 1) vzi = vz[ni];

            // loop through each spectrum of the data set
            dptr = data;
            wptr = wave;
            for(size_t ns=0; ns<nspec[nd]; ns++, dptr+=nwave[nd], wptr+=nwave[nd]){

                // This initialisation is needed per spectrum
                memset(fine, 0, NFINE*sizeof(double));

                // Loop over sub-spectra to simulate finite exposures
                int ntdiv = ndiv[nd][ns];
                for(int nt=0; nt<ntdiv; nt++){

                    // This initialisation is needed per sub-spectrum
                    memset(tfine, 0, NFINE*sizeof(double));

                    // Compute phase over uniformly spaced set from start to
                    // end of exposure. Times are assumed to be mid-exposure
                    phase = (time[nd][ns]+expose[nd][ns]*
                             (float(nt)-float(ntdiv-1)/2.)/
                             std::max(ntdiv-1,1)-tzero)/period;
                    cosp  = cos(2.*M_PI*phase);
                    sinp  = sin(2.*M_PI*phase);

                    // Inner loops are coming, so time to pre-compute some
                    // stuff for speed.

                    // If nz > 1 (3D tomog), assume equal spaced placed in Vz
                    // space spaced by vz symmetrically around 0 (there is in
                    // addition an overall gamma that is added later). We work
                    // in the fine array pixel space (denoted by p) to avoid
                    // too many divisions by vfine in the inner loops. A pixel
                    // with zero projected velocity should fall on the centre
                    // of the fine array at (NFINE-1)/2. Overall systemic
                    // velocity shifts are added later when the fine array is
                    // added into the data array.

                    pxstep = vxyi/vfine*cosp;
                    pystep = vxyi/vfine*sinp;
                    if(nz > 1){
                        pzoff  = double(nfine-1)/2. - vzi*double(nz-1)/2./vfine;
                        pzstep = vzi/vfine;
                    }else{
                        pzoff  = double(nfine-1)/2.;
                        pzstep = 0.;
                    }

                    // We are about to loop through the image so set the image
                    // pointer at the start of the current image. This is
                    // incremented in the innermost loop
                    iiptr = iptr;
                    
                    // Project onto the fine array. These are the innermost
                    // loops which need to be as fast as possible. Each loop
                    // initialises a pixel index and a pixel offset that are
                    // both added to as the loop progresses. All the
                    // projection has to do is calculate where to add into the
                    // tfine array, check that it lies within range and add the
                    // value in if it is. Per cycle of the innermost loop there
                    // are 4 additions (2 integer, 2 double), 1 rounding, 
                    // 1 conversion to an integer and 2 comparisons.
                    for(iz=0; iz<nz; iz++, pzoff+=pzstep){
                        for(iy=0, pyoff=pzoff-pystep*double(ny-1)/2.;
                            iy<ny; iy++, pyoff+=pystep){
                            for(ix=0, pxoff=pyoff-pxstep*double(nx-1)/2.;
                                ix<nx; ix++, pxoff+=pxstep, iiptr++){
                                nf  = int(round(pxoff));
                                if(nf >= 0 && nf < nfine) tfine[nf] += *iiptr;
                            }
                        }
                    }

                    // Now add in with correct weight (trapezoidal) to the
                    // fine buffer The vxyi squared factor is to give a
                    // similar intensity regardless of the pixel
                    // size. i.e. the pixel intensities should be thought of
                    // as being per (km/s)**2. The normalisation also ensures
                    // relative independence with respect to the value of
                    // ntdiv
                    if(ntdiv > 1 && (nt == 0 || nt == ntdiv - 1)){
                        weight = std::pow(vxyi,2)/(2*(ntdiv-1));
                    }else{
                        weight = std::pow(vxyi,2)/std::max(1,ntdiv-1);
                    }
                    for(nf=0; nf<nfine; nf++) fine[nf] += weight*tfine[nf];
                }

                // At this point 'fine' contains the projection of the current
                // image for the current spectrum. We now applying the blurring.

                // Take FFT of fine array
                fftw_execute(pforw);

                // multiply the FFT by the FFT of the blurring (in effect a
                // convolution)
                for(k=0; k<NFFT; k++)
                    fpfft[k] *= bfft[k];

                // Take the inverse FFT
                fftw_execute(pback);

                // We now need to add the blurred array into the spectrum once
                // for each wavelength associated with the current image. Do
                // this by calculating the start and end velocities of each
                // data pixel relative to the projection. The first and last
                // pixels of the data array are ignored because we don't have
                // the surrounding pixels needed to compute their extent.

                // loop over each line associated with this image
                for(k=0; k<wavel[ni].size(); k++){
                    wv = wavel[ni][k];
                    gm = gamma[ni][k];
                    sc = wavel[ni].size() > 1 ? scale[ni][k] : 1.;

                    // left-hand side of first pixel
                    v1 = CKMS*((wptr[0]+wptr[1])/2./wv)-gm;

                    // loop over every pixel in the spectrum. Could be made
                    // more efficient by narrowing on region overlapping with
                    // projection array although its probably not the CPU hog
                    for(m=1; m<nwave[nd]-1; m++){
                        // velocity of right-hand side of pixel
                        v2 = CKMS*((wptr[m]+wptr[m+1])/2./wv-1.)-gm;

                        if(v1 < vmax && v2 > -vmax){

                            // fp1, fp2 -- start and end limits of data pixel
                            // in fine array pixels 
                            fp1  = v1/vfine + double(nfine-1)/2.;
                            fp2  = v2/vfine + double(nfine-1)/2.;

                            // ifp1, ifp2 -- fine pixel range fully inside the data
                            // in traditional C form (i.e. ifp1<= <ifp2)
                            ifp1 = std::min(nfine,std::max(0,int(std::ceil(fp1+0.5))));
                            ifp2 = std::min(nfine,std::max(0,int(std::floor(fp2+0.5))));

                            // add full pixels
                            sum = 0.;
                            for(nf=ifp1; nf<ifp2; nf++) sum += fine[nf];

                            // add partial pixels
                            if(ifp1 > 0) sum += (ifp1-0.5-fp1)*fine[ifp1-1];
                            if(ifp2 < nfine) sum += (fp2-ifp2+0.5)*fine[ifp2];

                            // finally add it in
                            dptr[m] += sc*sum;
                        }

                        // move on a click
                        v1 = v2;
                    }
                }
            }

            // advance the image pointer
            iptr += nz*ny*nx;
        }
         
        // advance the data and wavelength pointers
        data += nspec[nd]*nwave[nd];
        wave += nspec[nd]*nwave[nd];
    }

    // cleanup in reverse order of allocation
    fftw_destroy_plan(pback);
    fftw_destroy_plan(pforw);
    fftw_destroy_plan(pblurr);
    delete[] tfine;
    delete[] fine;
    fftw_free(fpfft);
    fftw_free(bfft);
    delete[] blurr;
}

/* 'tr' is the counterpart of op calculating its transpose.
 *
 * Arguments::
 *
 *  image  : multiple images, stored in a 1D array for compatibility with
 *           mem routines (output)
 *  nxyz   : dimensions of each image, one/image, (input)
 *  vxy    : Vx-Vy pixel size(s), one/image, (input)
 *  vz     : Vz pixel size(s), one/image, (input)
 *  wavel  : central wavelengths for each image, vector/image, (input)
 *  gamma  : systemic velocities for each wavelength, vector/image, (input)
 *  scale  : scaling factors for each wavelength, vector/image, (input)
 *  tzero  : ephemeris zero point, same units as the times (input)
 *  period : ephemeris period, same units as the times (input)
 *  vfine  : km/s to use for internal fine array (input)
 *  vpad   : km/s to allow at each end of fine array beyond maximum extent
 *           implied by images. (input)
 *  data   : computed data, again as a 1D array for mem routines (input)
 *  wave   : wavelength array, matching the data array, (input)
 *  nwave  : number of wavelengths, one/dataset [effectively an X dimension],
 *           (input)
 *  nspec  : number of spectra, one/dataset [effectively a Y dimension], (input)
 *  time   : central times for each spectrum (input)
 *  expose : exposure lengths for each spectrum (input)
 *  ndiv   : sub-division factors for each spectrum (input)
 *  fwhm   : FWHM resolutions, kms/s, one/dataset, (input)
 *
 * Note the arguments 'image' through to 'vpad' are associated with the
 * Doppler map file, while the rest are associated with the data. 'data' is
 * the only output argument.
 */

void tr(float* image, const std::vector<Nxyz>& nxyz,
        const std::vector<double>& vxy, const std::vector<double>& vz,
        const std::vector<std::vector<double> >& wavel,
        const std::vector<std::vector<float> >& gamma,
        const std::vector<std::vector<float> >& scale,
        double tzero, double period, double vfine, double vpad,
        const float* data, const double* wave,
        const std::vector<size_t>& nwave, const std::vector<size_t>& nspec,
        const std::vector<std::vector<double> >& time,
        const std::vector<std::vector<float> >& expose,
        const std::vector<std::vector<int> >& ndiv,
        const std::vector<double>& fwhm){

    // See op for what is going on. This routine computes a transposed version
    // which is hard to explain except by saying here we carry out the transpose
    // of what happens in op. The actual code ends up looking very similar, with
    // the main change being a reversal of ordering. First change occurs here
    // where we tot up the total number of image pixels for a later zeroing
    // compare with op where this is done for the data instead.
    // In the rest of the code "[cf op]" indicates a difference wrt op
    size_t nimage = nxyz.size(), nx, ny, nz, nipix=0;
    double vmax = 0.;
    for(size_t nim=0; nim<nimage; nim++){
        nx = nxyz[nim].nx;
        ny = nxyz[nim].ny;
        nz = nxyz[nim].nz;
        if(nz > 1){
            vmax = std::max(std::sqrt(std::pow(vxy[nim],2)
                                      *(std::pow(nx,2)+std::pow(ny,2))+
                                      std::pow(vz[nim]*nz,2)), vmax);
        }else{
            vmax = std::max(std::sqrt(std::pow(vxy[nim],2)
                                      *(std::pow(nx,2)+std::pow(ny,2))), vmax);
        }
        nipix += nx*ny*nz;
    }
    vmax += 2.*vpad;

    // NFINE is the number of pixels needed for the fine array.
    const int nfine = std::max(5,int(std::ceil(vmax/vfine)));

    // adjust vmax to be the maximum velocity of the fine array (from zero not
    // full extent)
    vmax = nfine*vfine/2.;

    // Now we need to know how many pixels at most are needed for the blurring
    // function. Loop over the data sets. Also take chance to compute number
    // of data pixels
    size_t nblurr = 0;
    double psigma;
    for(size_t nd=0; nd<nspec.size(); nd++){
        // sigma of blurring in terms of fine array pixels
        psigma = fwhm[nd]/vfine/EFAC;

        // total blurr width will be 2*nblurr-1
        nblurr = std::max(nblurr,size_t(round(6.*psigma)+1));
    }

    // The blurring is carried out with FFTs requiring zero padding.  Thus the
    // actual number of pixels to grab is the nearest power of 2 larger than
    // nfine+2*nblurr-1. Call this NFINE.
    const size_t NFINE =
        size_t(std::pow(2,int(std::ceil(std::log(nfine+2*nblurr-1)/
                                        std::log(2.)))));

    // grab memory for blurring array and associated FFTs
    double *blurr = new double[NFINE];

    // this for the FFT of the blurring array. This could in fact be done in
    // place, but the amount of memory required here is small, so it is better
    // to have an explicit dedicated array
    const size_t NFFT = NFINE/2+1;
    fftw_complex *bfft = (fftw_complex*) 
        fftw_malloc(sizeof(fftw_complex) * NFFT);

    // this for the FFT of the fine pixel array
    fftw_complex *fpfft = (fftw_complex*) 
        fftw_malloc(sizeof(fftw_complex) * NFFT);

    // Grab space for fine arrays
    double *fine  = new double[NFINE];
    double *tfine = new double[NFINE];

    // create plans for the blurring, fine pixeland final inverse FFTs
    fftw_plan pblurr = fftw_plan_dft_r2c_1d(NFINE, blurr, bfft, 
                                            FFTW_ESTIMATE);
    fftw_plan pforw  = fftw_plan_dft_r2c_1d(NFINE, fine, fpfft, 
                                            FFTW_ESTIMATE);
    fftw_plan pback  = fftw_plan_dft_c2r_1d(NFINE, fpfft, fine, 
                                            FFTW_ESTIMATE);

    // various local variables
    size_t k, m, n;
    double norm, prf, fp1, fp2, v1, v2;
    double wv, sc, add;
    float gm;

    // pixel index in fine array
    int nf, ifp1, ifp2;

    // phase, cosine, sine, weight factor
    double phase, cosp, sinp, weight;

    // image indices
    size_t ix, iy, iz;

    // image velocity steps xy and z
    double vxyi, vzi = 0.;

    // steps and offsets in fine pixel space
    double pxstep, pystep, pzstep, pxoff, pyoff, pzoff;

    // temporary image & data pointers
    float *iptr, *iiptr;
    const float *dptr;
    const double *wptr;

    // large series of nested loops coming up.
    // First zero the image array [cf op]
    memset(image, 0, nipix*sizeof(float));

    // Loop over each data set
    for(size_t nd=0; nd<nspec.size(); nd++){

        // compute blurring array (specific to the dataset
        // so can't be done earlier). End by FFT-ing it in order
        // to perform the convolution later.
        memset(blurr, 0, NFINE*sizeof(double));
        norm = blurr[0] = 1.;
        psigma = fwhm[nd]/vfine/EFAC;
        for(m=1, n=NFINE-1; m<nblurr; m++, n--){
            prf = std::exp(-std::pow(m/psigma,2)/2.);
            blurr[m] = blurr[n] = prf;
            norm += 2.*prf;
        }

        // Next line gets overall FFT/iFFT normalisation right
        // cutting out any need to normalise by NFINE later
        norm *= NFINE;

        // normalise
        for(m=0; m<NFINE; m++)
            blurr[m] /= norm;

        // take FFT
        fftw_execute(pblurr);

        // Loop over each image. iptr updated at end of loop
        iptr = image;
        for(size_t ni=0; ni<nxyz.size(); ni++){

            // Calculate whether we can skip this particular
            // dataset / image combination ??
            // ...
            
            // extract image dimensions and velocity scales
            nx   = nxyz[ni].nx;
            ny   = nxyz[ni].ny;
            nz   = nxyz[ni].nz;
            vxyi = vxy[ni];
            if(nz > 1) vzi = vz[ni];

            // loop through each spectrum of the data set
            dptr = data;
            wptr = wave;
            for(size_t ns=0; ns<nspec[nd]; ns++, dptr+=nwave[nd], wptr+=nwave[nd]){

                // This initialisation is needed per spectrum
                memset(fine, 0, NFINE*sizeof(double));

                // [cf op]. This is the final step in op, here it comes the first

                // loop over each line associated with this image
                for(k=0; k<wavel[ni].size(); k++){
                    wv = wavel[ni][k];
                    gm = gamma[ni][k];
                    sc = wavel[ni].size() > 1 ? scale[ni][k] : 1.;

                    // left-hand side of first pixel
                    v1 = CKMS*((wptr[0]+wptr[1])/2./wv)-gm;

                    // loop over every pixel in the spectrum. Could be made
                    // more efficient by narrowing on region overlapping with
                    // projection array although its probably not the CPU hog
                    for(m=1; m<nwave[nd]-1; m++){
                        // velocity of right-hand side of pixel
                        v2 = CKMS*((wptr[m]+wptr[m+1])/2./wv-1.)-gm;

                        if(v1 < vmax && v2 > -vmax){

                            // fp1, fp2 -- start and end limits of data pixel
                            // in fine array pixels 
                            fp1  = v1/vfine + double(nfine-1)/2.;
                            fp2  = v2/vfine + double(nfine-1)/2.;

                            // ifp1, ifp2 -- fine pixel range fully inside the data
                            // in traditional C form (i.e. ifp1<= <ifp2)
                            ifp1 = std::min(nfine,std::max(0,int(std::ceil(fp1+0.5))));
                            ifp2 = std::min(nfine,std::max(0,int(std::floor(fp2+0.5))));

                            // [cf op]
                            add = sc*dptr[m];
                            for(nf=ifp1; nf<ifp2; nf++) fine[nf] += add;

                            // add partial pixels
                            if(ifp1 > 0) fine[ifp1-1]   += (ifp1-0.5-fp1)*add;
                            if(ifp2 < nfine) fine[ifp2] += (fp2-ifp2+0.5)*add;

                        }

                        // move on a click
                        v1 = v2;
                    }
                }

                // next section, blurring of fine array, stays same cf op

                // Take FFT of fine array
                fftw_execute(pforw);

                // multiply the FFT by the FFT of the blurring (in effect a
                // convolution)
                for(k=0; k<NFFT; k++)
                    fpfft[k] *= bfft[k];

                // Take the inverse FFT
                fftw_execute(pback);

                // Loop over sub-spectra to simulate finite exposures
                int ntdiv = ndiv[nd][ns];
                for(int nt=0; nt<ntdiv; nt++){

                    // Compute phase over uniformly spaced set from start to
                    // end of exposure. Times are assumed to be mid-exposure
                    phase = (time[nd][ns]+expose[nd][ns]*
                             (float(nt)-float(ntdiv-1)/2.)/
                             std::max(ntdiv-1,1)-tzero)/period;
                    cosp  = cos(2.*M_PI*phase);
                    sinp  = sin(2.*M_PI*phase);

                    // Inner loops are coming, so time to pre-compute some
                    // stuff for speed.

                    // If nz > 1 (3D tomog), assume equal spaced placed in Vz
                    // space spaced by vz symmetrically around 0 (there is in
                    // addition an overall gamma that is added later). We work
                    // in the fine array pixel space (denoted by p) to avoid
                    // too many divisions by vfine in the inner loops. A pixel
                    // with zero projected velocity should fall on the centre
                    // of the fine array at (NFINE-1)/2. Overall systemic
                    // velocity shifts are added later when the fine array is
                    // added into the data array.

                    pxstep = vxyi/vfine*cosp;
                    pystep = vxyi/vfine*sinp;
                    if(nz > 1){
                        pzoff  = double(nfine-1)/2. - vzi*double(nz-1)/2./vfine;
                        pzstep = vzi/vfine;
                    }else{
                        pzoff  = double(nfine-1)/2.;
                        pzstep = 0.;
                    }

                    // We are about to loop through the image so set the image
                    // pointer at the start of the current image. This is
                    // incremented in the innermost loop
                    iiptr = iptr;
                    
                    // Project onto the fine array. These are the innermost
                    // loops which need to be as fast as possible. Each loop
                    // initialises a pixel index and a pixel offset that are
                    // both added to as the loop progresses. All the
                    // projection has to do is calculate where to add into the
                    // tfine array, check that it lies within range and add the
                    // value in if it is. Per cycle of the innermost loop there
                    // are 4 additions (2 integer, 2 double), 1 rounding, 
                    // 1 conversion to an integer and 2 comparisons.

                    // [cf op]
                    if(ntdiv > 1 && (nt == 0 || nt == ntdiv - 1)){
                        weight = std::pow(vxyi,2)/(2*(ntdiv-1));
                    }else{
                        weight = std::pow(vxyi,2)/std::max(1,ntdiv-1);
                    }
                    for(nf=0; nf<nfine; nf++) tfine[nf] = weight*fine[nf];

                    for(iz=0; iz<nz; iz++, pzoff+=pzstep){
                        for(iy=0, pyoff=pzoff-pystep*double(ny-1)/2.;
                            iy<ny; iy++, pyoff+=pystep){
                            for(ix=0, pxoff=pyoff-pxstep*double(nx-1)/2.;
                                ix<nx; ix++, pxoff+=pxstep, iiptr++){
                                nf  = int(round(pxoff));
                                if(nf >= 0 && nf < nfine) *iiptr += tfine[nf];
                            }
                        }
                    }

                }
            }

            // advance the image pointer
            iptr += nz*ny*nx;
        }
         
        // advance the data and wavelength pointers
        data += nspec[nd]*nwave[nd];
        wave += nspec[nd]*nwave[nd];
    }

    // cleanup in reverse order of allocation
    fftw_destroy_plan(pback);
    fftw_destroy_plan(pforw);
    fftw_destroy_plan(pblurr);
    delete[] tfine;
    delete[] fine;
    fftw_free(fpfft);
    fftw_free(bfft);
    delete[] blurr;
}

/* gaussdef computes a gaussian default image by blurring in all three directions.
 * The blurring is carried out using FFTs. This routine applies the blurring to one image
 *
 * input  : input image (input)
 * nxyz   : dimensions. Only axes with dimensions > 1 are blurred. (input)
 * fwhmx  : FWHM blurr in X (pixels), <=0 to ignore. (input)
 * fwhmy  : FWHM blurr in Y (pixels), <=0 to ignore. (input)
 * fwhmz  : FWHM blurr in Z (pixels), <=0 to ignore. (input)
 * output : output (blurred) image (output)
 */

void gaussdef(const float *input, const Nxyz& nxyz, double fwhmx, double fwhmy, double fwhmz, float *output){

  // Work out buffer sizes needed for FFTs
  size_t naddx = 2*int(3.*fwhmx+1.); 
  size_t ntotx = nxyz.nx + naddx;
  const size_t NTOTX =
      size_t(std::pow(2,int(std::ceil(std::log(ntotx)/std::log(2.)))));

  size_t naddy = 2*int(3.*fwhmy+1.); 
  size_t ntoty = nxyz.ny + naddy;
  const size_t NTOTY =
      size_t(std::pow(2,int(std::ceil(std::log(ntoty)/std::log(2.)))));

  size_t naddz = 2*int(3.*fwhmz+1.); 
  size_t ntotz = nxyz.nz + naddz;
  const size_t NTOTZ =
      size_t(std::pow(2,int(std::ceil(std::log(ntotz)/std::log(2.)))));

  const size_t NTOT = std::max(std::max(NTOTX,NTOTY),NTOTZ);
  const size_t NFFT = NTOT/2+1;

  // Get memory

  // input workspace
  double *array = new double[NTOT];

  // blurring array
  double *blurr = new double[NTOT];

  // FFT of blurring array
  fftw_complex *bfft = (fftw_complex*) 
      fftw_malloc(sizeof(fftw_complex) * NFFT);

  // FFT of the array to be blurred
  fftw_complex *afft = (fftw_complex*) 
      fftw_malloc(sizeof(fftw_complex) * NFFT);

  // create FFT plans for the blurr, work and final inverse
  fftw_plan pblurr = fftw_plan_dft_r2c_1d(NTOT, blurr, bfft, 
                                          FFTW_ESTIMATE);
  fftw_plan pforw = fftw_plan_dft_r2c_1d(NTOT, array, afft, 
                                         FFTW_ESTIMATE);
  fftw_plan pback = fftw_plan_dft_c2r_1d(NTOT, afft, array, 
                                         FFTW_ESTIMATE);

  // some repeatedly used variables
  size_t ix, iy, iz, nstep, k, m, n;
  float *iptr, *iiptr, *optr, *ooptr;
  double norm, sigma, prf;

  // copy input to output
  memcpy(output, input, nxyz.ntot()*sizeof(float));

  // Blurr in X
  if(nxyz.nx > 1 && fwhmx >= 0.){

      norm  = blurr[0] = 1.;
      sigma = fwhmx/EFAC;
      for(m=1, n=NTOT-1; m<naddx/2; m++, n--){
          prf = std::exp(-std::pow(m/sigma,2)/2.);
          blurr[m] = blurr[n] = prf;
          norm += 2.*prf;
      }

      // zero the middle part
      memset(blurr, 0, NTOT*sizeof(double));
      
      // Next line gets the FFT/inverse-FFT pair
      // normalisation right.
      norm *= NTOT;
      
      // normalise
      for(m=0;m<NTOT;m++)
          blurr[m] /= norm;
      
      // FFT the blurring array
      fftw_execute(pblurr);
      
      iptr  = output;
      optr  = output;
      nstep = nxyz.nx;
      
      for(iz=0; iz<nxyz.nz; iz++){
          for(iy=0; iy<nxyz.ny; iy++, iptr+=nstep, optr+=nstep){
              
              // transfer data to double work array
              for(ix=0; ix<nxyz.nx; ix++)
                  array[ix] = double(iptr[ix]);
              
              // zeropad
              memset(array+nxyz.nx, 0, (NTOT-nxyz.nx)*sizeof(double));
              
              // FFT
              fftw_execute(pforw);
              
              // multiply by the FFT of the blurr 
              for(k=0; k<NFFT; k++)
                  afft[k] *= bfft[k];
              
              // inverse FFT
              fftw_execute(pback);
              
              // transfer result to output
              for(ix=0; ix<nxyz.nx; ix++)
                  optr[ix] = float(array[ix]);
          }
      }
  }

  // Blurr in Y
  if(nxyz.ny > 1 && fwhmy >= 0.){

      memset(blurr, 0, NTOT*sizeof(double));
      norm  = blurr[0] = 1.;
      sigma = fwhmy/EFAC;
      for(m=1, n=NTOT-1; m<naddy/2; m++, n--){
          prf = std::exp(-std::pow(m/sigma,2)/2.);
          blurr[m] = blurr[n] = prf;
          norm += 2.*prf;
      }
      
      // Next line gets the FFT/inverse-FFT pair
      // normalisation right.
      norm *= NTOT;
      
      // normalise
      for(m=0;m<NTOT;m++)
          blurr[m] /= norm;
      
      // FFT the blurring array
      fftw_execute(pblurr);
      nstep = nxyz.nx;
      for(iz=0; iz<nxyz.nz; iz++){
          iiptr = output + nxyz.nx*nxyz.nz*iz;
          ooptr = output + nxyz.nx*nxyz.nz*iz;

          for(ix=0; ix<nxyz.nx; ix++, iiptr++, ooptr++){

              // transfer data to double work array
              iptr = iiptr;
              for(iy=0; iy<nxyz.ny; iy++, iptr+=nstep)
                  array[iy] = double(*iptr);

              // zeropad
              memset(array+nxyz.ny, 0, (NTOT-nxyz.ny)*sizeof(double));
              
              // FFT
              fftw_execute(pforw);
              
              // multiply by the FFT of the blurr 
              for(k=0; k<NFFT; k++)
                  afft[k] *= bfft[k];
              
              // inverse FFT
              fftw_execute(pback);
              
              // transfer result to output
              optr = ooptr;
              for(iy=0; iy<nxyz.ny; iy++, optr+=nstep)
                  *optr = float(array[iy]);
          }
      }
  }

  // Blurr in Z
  if(nxyz.nz > 1 && fwhmz >= 0.){

      memset(blurr, 0, NTOT*sizeof(double));
      norm  = blurr[0] = 1.;
      sigma = fwhmz/EFAC;
      for(m=1, n=NTOT-1; m<naddz/2; m++, n--){
          prf = std::exp(-std::pow(m/sigma,2)/2.);
          blurr[m] = blurr[n] = prf;
          norm += 2.*prf;
      }
      
      // Next line gets the FFT/inverse-FFT pair
      // normalisation right.
      norm *= NTOT;
      
      // normalise
      for(m=0;m<NTOT;m++)
          blurr[m] /= norm;
      
      // FFT the blurring array
      fftw_execute(pblurr);

      nstep = nxyz.nx*nxyz.ny;

      for(iy=0; iy<nxyz.ny; iy++){
          iiptr = output + nxyz.nx*iy;
          ooptr = output + nxyz.nx*iy;
          for(ix=0; ix<nxyz.nx; ix++, iiptr++, ooptr++){

              // transfer data to double work array
              iptr = iiptr;
              for(iz=0; iz<nxyz.nz; iz++, iptr+=nstep)
                  array[iy] = double(*iptr);

              // zeropad
              memset(array+nxyz.nz, 0, (NTOT-nxyz.nz)*sizeof(double));
              
              // FFT
              fftw_execute(pforw);
              
              // multiply by the FFT of the blurr 
              for(k=0; k<NFFT; k++)
                  afft[k] *= bfft[k];
              
              // inverse FFT
              fftw_execute(pback);
              
              // transfer result to output
              optr = ooptr;
              for(iz=0; iz<nxyz.nz; iz++, optr+=nstep)
                  *optr = float(array[iz]);
          }
      }
  }
      
  // Get memory back
  fftw_destroy_plan(pback);
  fftw_destroy_plan(pforw);
  fftw_destroy_plan(pblurr);
  fftw_free(afft);
  fftw_free(bfft);
  delete[] blurr;
  delete[] array;
}

// Wrap globals to get through to opus 
// and tropus inside a namespace

namespace Dopp {

    // For Map objects 
    std::vector<Nxyz> nxyz;
    std::vector<int> def;
    std::vector<double> vxy, vz, fwhmxy, fwhmz;
    std::vector<std::vector<double> > wavel;
    std::vector<std::vector<float> > gamma, scale;
    double tzero, period, vfine, vpad;

    // For Data objects
    std::vector<size_t> nwave, nspec;
    std::vector<std::vector<double> > time;
    std::vector<std::vector<float> > expose;
    std::vector<std::vector<int> > ndiv;
    std::vector<double> fwhm;
    double *wave;
}

/* The opus routine needed for memsys
 */
void Mem::opus(const int j, const int k){

    std::cerr << "    OPUS " << j+1 << " ---> " << k+1 << std::endl;
    
    op(Mem::Gbl::st+Mem::Gbl::kb[j], Dopp::nxyz, Dopp::vxy, Dopp::vz,
       Dopp::wavel, Dopp::gamma, Dopp::scale, Dopp::tzero, Dopp::period,
       Dopp::vfine, Dopp::vpad, 
       Mem::Gbl::st+Mem::Gbl::kb[k], Dopp::wave, Dopp::nwave, Dopp::nspec,
       Dopp::time, Dopp::expose, Dopp::ndiv, Dopp::fwhm);
}

/* The tropus routine needed for memsys
 */
void Mem::tropus(const int k, const int j){

    std::cerr << "  TROPUS " << j+1 << " <--- " << k+1 << std::endl;
    
    tr(Mem::Gbl::st+Mem::Gbl::kb[j], Dopp::nxyz, Dopp::vxy, Dopp::vz,
       Dopp::wavel, Dopp::gamma, Dopp::scale, Dopp::tzero, Dopp::period,
       Dopp::vfine, Dopp::vpad, 
       Mem::Gbl::st+Mem::Gbl::kb[k], Dopp::wave, Dopp::nwave, Dopp::nspec,
       Dopp::time, Dopp::expose, Dopp::ndiv, Dopp::fwhm);
}

/* npix_data -- returns with the number of pixels needed to allocate memory
 * for the images in a Doppler Map.
 *
 * Arguments::
 *
 *  Map     :  the Doppler image (input)
 *  npix    :  the total number of pixels (output)
 *
 * Returns true/false according to whether it has succeeded. If false
 * it sets an exception so you don't need to set another.
 */

bool
npix_map(PyObject *Map, size_t& npix)
{

    // set status code
    bool status = false;

    if(Map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.npix_map: Map is NULL");
        return status;
    }

    npix = 0;

    // initialise attribute pointers
    PyObject *data=NULL, *image=NULL, *idata=NULL;

    // get the data attribute.
    data  = PyObject_GetAttrString(Map, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.npix_map: Map has no attribute called data");
        goto failed;
    }

    // data should be a list of Image, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.npix_map: Map.data is not a sequence");
        goto failed;
    }

    // define scope to avoid compilation error
    {
        // work out the number of Images
        Py_ssize_t nimage = PySequence_Size(data);

        // loop through the Image objects
        for(Py_ssize_t i=0; i<nimage; i++){

            image = PySequence_GetItem(data, i);
            if(!image){
                PyErr_SetString(PyExc_ValueError, "doppler.npix_map:"
                                " failed to extract an Image from Map.data");
                goto failed;
            }

            // get data attribute
            idata = PyObject_GetAttrString(image, "data");

            if(idata && PyArray_Check(idata) &&
               (PyArray_NDIM(idata) == 2 || PyArray_NDIM(idata) == 3)){

                npix += PyArray_SIZE(idata);

                // clear temporaries
                Py_CLEAR(idata);
                Py_CLEAR(image);

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.npix_map:"
                                " failed to locate Image.data attribute or"
                                " it was not an array or had wrong dimension");
                goto failed;
            }
        }
    }

    // made it!!
    status = true;

 failed:

    Py_XDECREF(idata);
    Py_XDECREF(image);
    Py_XDECREF(data);

    return status;
}

/* read_map -- reads the data contained in a PyObject representing a Doppler
 * map returning the contents in a single C-array for the image data and a
 * series of vectors for the rest of the associated information defining the
 * map. Map objects can have multiple images, and each image can have multiple
 * associated wavelengths, systemic velocities and scaling factors. This leads
 * to the vector and vector of vectors objects below. The multiple images are
 * concatenated into a single C-array for compatibility with the memsys
 * routines.
 *
 * Arguments::
 *
 *  map     :  the Doppler map (input)
 *  images  :  C-style array with enough space for the image array (output), 
 *  nxyz    :  vector of image dimensions (output)
 *  vxy     :  vector of VX-VY pixels sizes for each image (output)
 *  vz      :  vector of VZ spacings for each image (output)
 *  wave    :  vector of vectors of wavelengths for each image (output)
 *  gamma   :  vector of vectors of systemic velocities for each image (output)
 *  scale   :  vector of vectors of scale factors for each image (output)
 *  def     :  default options for each image (output) [see map.py]
 *  fwhmxy  :  gaussian defaults, FWHM blurr in km/s in Vx-Vy(output)
 *  fwhmz   :  gaussian defaults, FWHM blurr in km/s in Vz (output)
 *  tzero   :  zeropoint of ephemeris (output)
 *  period  :  period of ephemeris (output)
 *  vfine   :  km/s for fine projection array (output)
 *  vpad    :  km/s extra padding for fine array (output)
 *
 *  Returns true/false according to success. If false, the outputs above
 *  will be useless. If false, a Python exception is raised and you should
 *  return NULL from the calling routine. In this case any memory allocated
 *  internally will have been deleted before exiting and you should not attempt
 *  to free it in the calling routine.
 */

bool
read_map(PyObject *map, float* images, std::vector<Nxyz>& nxyz,
         std::vector<double>& vxy, std::vector<double>& vz,
         std::vector<std::vector<double> >& wave,
         std::vector<std::vector<float> >& gamma,
         std::vector<std::vector<float> >& scale,
         std::vector<int>& def, std::vector<double>& fwhmxy,
         std::vector<double>& fwhmz, double& tzero, 
         double& period, double& vfine, double& vpad)
{

    bool status = false;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.read_map: map is NULL");
        return status;
    }

    float *timages=images;

    // initialise attribute pointers
    PyObject *data=NULL, *image=NULL, *idata=NULL, *iwave=NULL;
    PyObject *igamma=NULL, *ivxy=NULL, *iscale=NULL, *ivz=NULL;
    PyObject *itzero=NULL, *iperiod=NULL, *ivfine=NULL, *ivpad=NULL;
    PyObject *idef=NULL, *doption=NULL, *dfwhmxy=NULL, *dfwhmz=NULL;
    PyArrayObject *darray=NULL, *warray=NULL, *garray=NULL;
    PyArrayObject *sarray=NULL;

    // clear the output vectors
    nxyz.clear();
    vxy.clear();
    vz.clear();
    wave.clear();
    gamma.clear();
    def.clear();
    fwhmxy.clear();
    fwhmz.clear();
    scale.clear();

    // get attributes.
    itzero  = PyObject_GetAttrString(map, "tzero");
    iperiod = PyObject_GetAttrString(map, "period");
    ivfine  = PyObject_GetAttrString(map, "vfine");
    ivpad   = PyObject_GetAttrString(map, "vpad");
    data    = PyObject_GetAttrString(map, "data");

    if(!itzero || !iperiod || !ivfine || !ivpad || !data){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.read_map: one or more of "
                        "data, tzero, period, vfine, vpad is missing");
        goto failed;
    }

    // data should be a list of Images, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.read_map: map.data is not a sequence");
        goto failed;
    }

    // store values for easy ones
    tzero  = PyFloat_AsDouble(itzero);
    period = PyFloat_AsDouble(iperiod);
    vfine  = PyFloat_AsDouble(ivfine);
    vpad   = PyFloat_AsDouble(ivpad);

    // define scope to avoid compilation error
    {
        // work out the number of Images
        Py_ssize_t nimage = PySequence_Size(data);

        // loop through Images
        for(Py_ssize_t i=0; i<nimage; i++){

            image  = PySequence_GetItem(data, i);
            if(!image){
                PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                " failed to access element in map.data");
                goto failed;
            }

            // get attributes of the image
            idata  = PyObject_GetAttrString(image, "data");
            iwave  = PyObject_GetAttrString(image, "wave");
            igamma = PyObject_GetAttrString(image, "gamma");
            idef   = PyObject_GetAttrString(image, "default");
            iscale = PyObject_GetAttrString(image, "scale");
            ivxy   = PyObject_GetAttrString(image, "vxy");
            ivz    = PyObject_GetAttrString(image, "vz");

            if(idata && iwave && igamma && ivxy && idef && iscale && ivz){

                darray = (PyArrayObject*)                               \
                    PyArray_FromAny(idata, PyArray_DescrFromType(NPY_FLOAT),
                                    2, 3, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!darray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                    " failed to extract array from data");
                    goto failed;
                }
                int nddim = PyArray_NDIM(darray);
                npy_intp *ddim = PyArray_DIMS(darray);
                memcpy (timages, PyArray_DATA(darray), PyArray_NBYTES(darray));
                timages += PyArray_SIZE(darray);
                if(nddim == 2){
                    nxyz.push_back(Nxyz(ddim[1],ddim[0]));
                }else{
                    nxyz.push_back(Nxyz(ddim[2],ddim[1],ddim[0]));
                }

                // copy over wavelengths
                warray = (PyArrayObject*) \
                    PyArray_FromAny(iwave, PyArray_DescrFromType(NPY_DOUBLE),
                                    1, 1, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!warray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                    " failed to extract array from wave");
                    goto failed;
                }
                npy_intp *wdim = PyArray_DIMS(warray);
                double *wptr = (double*) PyArray_DATA(warray);
                std::vector<double> waves(wptr,wptr+PyArray_SIZE(warray));
                wave.push_back(waves);

                // copy over systemic velocities
                garray = (PyArrayObject*) \
                    PyArray_FromAny(igamma, PyArray_DescrFromType(NPY_FLOAT),
                                    1, 1, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!garray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                    " failed to extract array from gamma");
                    goto failed;
                }
                npy_intp *gdim = PyArray_DIMS(garray);
                if(gdim[0] != wdim[0]){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                    " systemic velocities & wavelengths do not"
                                    " have matching sizes");
                    goto failed;
                }
                float *gptr = (float*) PyArray_DATA(garray);
                std::vector<float> gammas(gptr,gptr+PyArray_SIZE(garray));
                gamma.push_back(gammas);

                // get attributes of Default object
                doption = PyObject_GetAttrString(idef, "option");
                if(!doption){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                    " failed to locate an Image.default.option");
                    goto failed;
                } 
                int option = int(PyInt_AsLong(doption));
                def.push_back(option);
                if(option == 2 || option == 3){
                    dfwhmxy = PyObject_GetAttrString(idef, "fwhmxy");
                    if(!dfwhmxy){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                        " failed to locate an Image.default.fwhmxy");
                        goto failed;
                    }
                    fwhmxy.push_back(PyFloat_AS_DOUBLE(dfwhmxy));
                }
                if(option == 3){
                    dfwhmz = PyObject_GetAttrString(idef, "fwhmz");
                    if(!dfwhmz){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                        " failed to locate an Image.default.fwhmz");
                        goto failed;
                    }
                    fwhmz.push_back(PyFloat_AS_DOUBLE(dfwhmz));
                }

                // store the Vx-Vy pixel size
                vxy.push_back(PyFloat_AsDouble(ivxy));

                // store the Vz spacing
                if(nddim == 3) vz.push_back(PyFloat_AsDouble(ivz));

                // scale factors, if needed
                if(wdim[0] > 1){
                    sarray = (PyArrayObject*)                           \
                        PyArray_FromAny(iscale,
                                        PyArray_DescrFromType(NPY_FLOAT),
                                        1, 1, NPY_IN_ARRAY | NPY_FORCECAST,
                                        NULL);
                    if(!sarray){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                        " failed to extract array from scale");
                        goto failed;
                    }
                    npy_intp *sdim = PyArray_DIMS(sarray);
                    if(sdim[0] != wdim[0]){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                        " scale factors & wavelengths do not"
                                        " have matching sizes");
                        goto failed;
                    }
                    float *sptr = (float*) PyArray_DATA(sarray);
                    std::vector<float> scales(sptr,sptr+PyArray_SIZE(sarray));
                    scale.push_back(scales);
                }

                // clear temporaries
                Py_CLEAR(sarray);
                Py_CLEAR(garray);
                Py_CLEAR(warray);
                Py_CLEAR(darray);
                Py_CLEAR(dfwhmz);
                Py_CLEAR(dfwhmxy);
                Py_CLEAR(doption);
                Py_CLEAR(ivz);
                Py_CLEAR(ivxy);
                Py_CLEAR(iscale);
                Py_CLEAR(idef);
                Py_CLEAR(igamma);
                Py_CLEAR(iwave);
                Py_CLEAR(idata);
                Py_CLEAR(image);

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.read_map:"
                                " failed to locate an Image attribute");
                goto failed;
            }
        }
    }

    // made it!!
    status = true;

 failed:

    Py_XDECREF(sarray);
    Py_XDECREF(garray);
    Py_XDECREF(warray);
    Py_XDECREF(darray);
    Py_XDECREF(ivz);
    Py_CLEAR(dfwhmz);
    Py_CLEAR(dfwhmxy);
    Py_CLEAR(doption);
    Py_XDECREF(ivxy);
    Py_XDECREF(iscale);
    Py_CLEAR(idef);
    Py_XDECREF(igamma);
    Py_XDECREF(iwave);
    Py_XDECREF(idata);
    Py_XDECREF(image);

    Py_XDECREF(data);
    Py_XDECREF(ivpad);
    Py_XDECREF(ivfine);
    Py_XDECREF(iperiod);
    Py_XDECREF(itzero);

    return status;
}

/* update_map -- copies image data into a Map object for output. The idea is
 * you have created an image inside a C-array that was originally defined by
 * the structure of a Map object using read_map. You now copy it over into the
 * Map object. The very limited nature of the modification here is because
 * most manipulations should be carried out in Python.
 *
 * Arguments::
 *
 *  images : pointer to the images to load into the Map object. Contiguous array
 *         of floats. It must match the total size of the image arrays in map
 *         (input)
 *
 *  map    : the Doppler map (output)
 *
 *  Returns True/False according to success. If False, the outputs above
 *  will not be correctly assigned. If False, a Python exception is raised
 *  and you should return NULL from the calling routine.
 */

bool
update_map(float const* images, PyObject* map)
{

    // set status code
    bool status = false;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.update_map: map is NULL");
        return status;
    }

    // initialise attribute pointers
    PyObject *data=NULL, *image=NULL, *idata=NULL;
    PyArrayObject *array=NULL;

    // get the data attribute.
    data  = PyObject_GetAttrString(map, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.update_map: map has no attribute"
                        " called data");
        goto failed;
    }

    // data should be a list of Image objects, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.update_map: map.data is not a sequence");
        goto failed;
    }

    {
        // work out the number of Image objects
        Py_ssize_t nimage = PySequence_Size(data);

        // loop through the Image objects
        for(Py_ssize_t i=0; i<nimage; i++){
            image = PySequence_GetItem(data, i);
            if(!image){
                PyErr_SetString(PyExc_ValueError, "doppler.update_map:"
                                " failed to access element in map.data");
                goto failed;
            }

            // get data attribute of the Image
            idata = PyObject_GetAttrString(image, "data");

            if(idata){

                // get images in form that allows easy modification
                array = (PyArrayObject*) \
                    PyArray_FromAny(idata, PyArray_DescrFromType(NPY_FLOAT),
                                    2, 3, NPY_INOUT_ARRAY | NPY_FORCECAST,
                                    NULL);
                if(!array){
                    PyErr_SetString(PyExc_ValueError, "doppler.update_map:"
                                    " failed to extract array from data");
                    goto failed;
                }
                memcpy (PyArray_DATA(array), images, PyArray_NBYTES(array));
                images += PyArray_SIZE(array);

                // clear temporaries
                Py_CLEAR(array);
                Py_CLEAR(idata);
                Py_CLEAR(image);

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.update_map:"
                                " failed to locate an Image attribute");
                goto failed;
            }
        }
    }

    // made it!!
    status = true;

 failed:

    Py_XDECREF(array);
    Py_XDECREF(idata);
    Py_XDECREF(image);
    Py_XDECREF(data);
    return status;
}

/* npix_data -- returns with the number of pixels needed to allocate memory
 * for the fluxes, errors and wavelengths of a Data object.
 *
 * Arguments::
 *
 *  Data    :  the Doppler data (input)
 *  npix    :  the total number of pixels (output)
 *
 * Returns true/false according to whether it has succeeded. If false
 * it sets an exception so you don't need to set another.
 */

bool
npix_data(PyObject *Data, size_t& npix)
{

    // set status code
    bool status = false;

    if(Data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.npix_data: Data is NULL");
        return status;
    }

    npix = 0;

    // initialise attribute pointers
    PyObject *data=NULL, *spectra=NULL, *sflux=NULL;

    // get the data attribute.
    data  = PyObject_GetAttrString(Data, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.npix_data: Data has no attribute called data");
        goto failed;
    }

    // data should be a list of Spectra, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.npix_data: Data.data is not a sequence");
        goto failed;
    }

    // define scope to avoid compilation error
    {
        // work out the number of Images
        Py_ssize_t nspectra = PySequence_Size(data);

        // loop through the Spectra objects
        for(Py_ssize_t i=0; i<nspectra; i++){
            spectra = PySequence_GetItem(data, i);
            if(!spectra){
                PyErr_SetString(PyExc_ValueError, "doppler.npix_data:"
                                " failed to extract a Spectra from Data.data");
                goto failed;
            }

            // get flux attribute
            sflux   = PyObject_GetAttrString(spectra, "flux");

            if(sflux && PyArray_Check(sflux) && PyArray_NDIM(sflux) == 2){

                npix += PyArray_SIZE(sflux);

                // clear temporaries
                Py_CLEAR(sflux);
                Py_CLEAR(spectra);

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.npix_data:"
                                " failed to locate Spectra.flux attribute or"
                                " it was not an array or had wrong dimension");
                goto failed;
            }
        }
    }

    // made it!!
    status = true;

 failed:

    Py_XDECREF(sflux);
    Py_XDECREF(spectra);
    Py_XDECREF(data);

    return status;
}

/* read_data -- reads the data contained in a PyObject representing Doppler
 * imaging data, returning the contents in memory pointed to by three input
 * pointers, and a series of vectors containing remaining information per
 * spectrum.
 *
 * Arguments::
 *
 *  Data     :  the Doppler data (input)
 *  flux     :  pointer to a single 1D array where the flux data will be
 *              stored (output). Note this may involve multiple different
 *              data sets which are written sequentially. This must have
 *              enough space to hold the data of course.
 *  ferr     :  pointer to 1D array where flux errors will be stored (output)
 *              Same remarks about memory for flux apply to ferr.
 *  wave     :  pointer to 1D array where wavelengths will be stored (output)
 *              Same remarks about memory for flux apply to wave.
 *  nwave    :  number of wavelengths for each dataset
 *  nspec    :  number of spectra in each dataset
 *  time     :  vector of vector<double> of times for each spectrum (output).
 *  expose   :  vector of vector<float> of exposure times for each spectrum
 *              (output).
 *  ndiv     :  vector of vector<int> of sub-division factors for each
 *              spectrum (output).
 *  fwhm     :  vector of doubles, fwhm in pixels for each data set (output)
 *
 *  Returns true/false according to success. If false, the outputs above
 *  will be meaningless. Also, a Python exception is raised and you should
 *  return NULL from the calling routine. Any memory assigned to flux, ferr
 *  and wave will be deleted in this case so there is no need to handle this
 *  outside the routine in case of error.
 */

bool
read_data(PyObject *Data, float* flux, float* ferr, double* wave,
          std::vector<size_t>& nwave, std::vector<size_t>& nspec,
          std::vector<std::vector<double> >& time,
          std::vector<std::vector<float> >& expose,
          std::vector<std::vector<int> >& ndiv,
          std::vector<double>& fwhm)
{

    // set status code
    bool status = false;

    if(Data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.read_data: Data is NULL");
        return status;
    }

    // temporary versions which can be modified safely
    float  *tflux=flux;
    float  *tferr=ferr;
    double *twave=wave;

    // clear the output vectors
    nwave.clear();
    nspec.clear();
    time.clear();
    expose.clear();
    ndiv.clear();
    fwhm.clear();

    // initialise attribute pointers
    PyObject *data=NULL, *spectra=NULL, *sflux=NULL, *sferr=NULL;
    PyObject *swave=NULL, *stime=NULL, *sexpose=NULL, *sndiv=NULL;
    PyObject *sfwhm=NULL;
    PyArrayObject *farray=NULL, *earray=NULL, *warray=NULL;
    PyArrayObject *tarray=NULL, *xarray=NULL, *narray=NULL;

    // get the data attribute.
    data  = PyObject_GetAttrString(Data, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.read_data: Data has no attribute called data");
        goto failed;
    }

    // data should be a list of Spectra, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.read_data: Data.data is not a sequence");
        goto failed;
    }

    // define scope to avoid compilation error
    {
        // work out the number of Images
        Py_ssize_t nspectra = PySequence_Size(data);

        // loop through the Spectra objects
        for(Py_ssize_t i=0; i<nspectra; i++){
            spectra = PySequence_GetItem(data, i);
            if(!spectra){
                PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                " failed to access element in Data.data");
                goto failed;
            }

            // get attributes of the Spectra
            sflux   = PyObject_GetAttrString(spectra, "flux");
            sferr   = PyObject_GetAttrString(spectra, "ferr");
            swave   = PyObject_GetAttrString(spectra, "wave");
            stime   = PyObject_GetAttrString(spectra, "time");
            sexpose = PyObject_GetAttrString(spectra, "expose");
            sndiv   = PyObject_GetAttrString(spectra, "ndiv");
            sfwhm   = PyObject_GetAttrString(spectra, "fwhm");

            // some basic checks
            if(sflux && sferr && swave && stime && sexpose && sndiv && sfwhm){

                // transfer fluxes)
                farray = (PyArrayObject*) \
                    PyArray_FromAny(sflux, PyArray_DescrFromType(NPY_FLOAT),
                                    2, 2, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!farray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " failed to extract array from flux");
                    goto failed;
                }
                npy_intp *fdim = PyArray_DIMS(farray);
                memcpy (tflux, PyArray_DATA(farray), PyArray_NBYTES(farray));
                tflux += PyArray_SIZE(farray);
                nspec.push_back(fdim[0]);
                nwave.push_back(fdim[1]);

                // transfer flux errors
                earray = (PyArrayObject*) \
                    PyArray_FromAny(sferr, PyArray_DescrFromType(NPY_FLOAT),
                                    2, 2, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!earray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " failed to extract array from ferr");
                    goto failed;
                }
                npy_intp *edim = PyArray_DIMS(earray);
                for(int j=0; j<2; j++){
                    if(fdim[j] != edim[j]){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                        " flux and ferr have incompatible"
                                        " dimensions");
                        goto failed;
                    }
                }
                memcpy (tferr, PyArray_DATA(earray), PyArray_NBYTES(earray));
                tferr += PyArray_SIZE(earray);

                // transfer wavelengths
                warray = (PyArrayObject*) \
                    PyArray_FromAny(swave, PyArray_DescrFromType(NPY_DOUBLE),
                                    2, 2, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!warray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " failed to extract array from wave");
                    goto failed;
                }
                npy_intp *wdim = PyArray_DIMS(warray);
                for(int j=0; j<2; j++){
                    if(fdim[j] != wdim[j]){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                        " flux and wavelength have incompatible"
                                        " dimensions");
                        goto failed;
                    }
                }
                memcpy (twave, PyArray_DATA(warray), PyArray_NBYTES(warray));
                twave += PyArray_SIZE(warray);

                // get times
                tarray = (PyArrayObject*) \
                    PyArray_FromAny(stime, PyArray_DescrFromType(NPY_DOUBLE),
                                    1, 1, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!tarray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " failed to extract array from time");
                    goto failed;
                }
                npy_intp *tdim = PyArray_DIMS(tarray);
                if(fdim[0] != tdim[0]){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " flux and time have incompatible"
                                    " dimensions");
                    goto failed;
                }
                double *tptr = (double*) PyArray_DATA(tarray);
                std::vector<double> times(tptr,tptr+PyArray_SIZE(tarray));
                time.push_back(times);

                // get exposures
                xarray = (PyArrayObject*) \
                    PyArray_FromAny(sexpose, PyArray_DescrFromType(NPY_FLOAT),
                                    1, 1, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!xarray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " failed to extract array from expose");
                    goto failed;
                }
                npy_intp *exdim = PyArray_DIMS(xarray);
                if(fdim[0] != exdim[0]){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " flux and expose have incompatible"
                                    " dimensions");
                    goto failed;
                }
                float *exptr = (float*) PyArray_DATA(xarray);
                std::vector<float> exposes(exptr,exptr+PyArray_SIZE(xarray));
                expose.push_back(exposes);

                // get ndiv factors
                narray = (PyArrayObject*) \
                    PyArray_FromAny(sndiv, PyArray_DescrFromType(NPY_INT),
                                    1, 1, NPY_IN_ARRAY | NPY_FORCECAST, NULL);
                if(!narray){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " failed to extract array from ndiv");
                    goto failed;
                }
                npy_intp *ndim = PyArray_DIMS(narray);
                if(fdim[0] != ndim[0]){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                    " flux and ndiv have incompatible"
                                    " dimensions");
                    goto failed;
                }
                int *nptr = (int*) PyArray_DATA(narray);
                std::vector<int> ndivs(nptr,nptr+PyArray_SIZE(narray));
                ndiv.push_back(ndivs);

                // store the FWHM
                fwhm.push_back(PyFloat_AsDouble(sfwhm));

                // clear temporaries
                Py_CLEAR(narray);
                Py_CLEAR(xarray);
                Py_CLEAR(tarray);
                Py_CLEAR(warray);
                Py_CLEAR(earray);
                Py_CLEAR(farray);
                Py_CLEAR(sfwhm);
                Py_CLEAR(sexpose);
                Py_CLEAR(swave);
                Py_CLEAR(stime);
                Py_CLEAR(sferr);
                Py_CLEAR(sflux);
                Py_CLEAR(spectra);

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.read_data:"
                                " failed to locate a Spectra attribute");
                goto failed;
            }
        }
    }

    // made it!!
    status = true;

 failed:

    Py_XDECREF(narray);
    Py_XDECREF(xarray);
    Py_XDECREF(tarray);
    Py_XDECREF(warray);
    Py_XDECREF(earray);
    Py_XDECREF(farray);
    Py_XDECREF(sfwhm);
    Py_XDECREF(sexpose);
    Py_XDECREF(stime);
    Py_XDECREF(sferr);
    Py_XDECREF(swave);
    Py_XDECREF(sflux);
    Py_XDECREF(spectra);
    Py_XDECREF(data);

    if(!status){
        delete[] flux;
        delete[] ferr;
        delete[] wave;
    }
    return status;
}

/* update_data -- copies flux data into a Data object for output. The idea is
 * you have made some data e.g. using one of the opus-related routines to
 * generate data from a model. This goes into some array that was defined by
 * the structure of a Data object. You now copy it over into the Data
 * object. The very limited nature of the modification here is because most
 * manipulations should be carried out in Python.
 *
 * Arguments::
 *
 *  flux : pointer to the flux to load into the Data object. Contiguous array
 *         of floats. It must match the total size of the flux arrays in Data
 *         (input)
 *
 *  Data : the Doppler data (output)
 *
 *  Returns True/False according to success. If False, the outputs above
 *  will not be correctly assigned. If False, a Python exception is raised
 *  and you should return NULL from the calling routine.
 */

bool
update_data(float const* flux, PyObject* Data)
{

    // set status code
    bool status = false;

    if(Data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.update_data: Data is NULL");
        return status;
    }

    // initialise attribute pointers
    PyObject *data=NULL, *spectra=NULL, *sflux=NULL;
    PyArrayObject *array=NULL;

    // get the data attribute.
    data  = PyObject_GetAttrString(Data, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.update_data: Data has no attribute"
                        " called data");
        goto failed;
    }

    // data should be a list of Spectra, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError,
                        "doppler.update_data: Data.data is not a sequence");
        goto failed;
    }

    {
        // work out the number of Spectra
        Py_ssize_t nspectra = PySequence_Size(data);

        // loop through the Spectra objects
        for(Py_ssize_t i=0; i<nspectra; i++){
            spectra = PySequence_GetItem(data, i);
            if(!spectra){
                PyErr_SetString(PyExc_ValueError, "doppler.update_data:"
                                " failed to access element in Data.data");
                goto failed;
            }

            // get flux attribute of the Spectra
            sflux = PyObject_GetAttrString(spectra, "flux");

            if(sflux){

                // get fluxes if form that allows easy modification
                array = (PyArrayObject*) \
                    PyArray_FromAny(sflux, PyArray_DescrFromType(NPY_FLOAT),
                                    2, 2, NPY_INOUT_ARRAY | NPY_FORCECAST, 
                                    NULL);
                if(!array){
                    PyErr_SetString(PyExc_ValueError, "doppler.update_data:"
                                    " failed to extract float array from flux");
                    goto failed;
                }
                memcpy (PyArray_DATA(array), flux, PyArray_NBYTES(array));
                flux += PyArray_SIZE(array);

                // clear temporaries
                Py_CLEAR(array);
                Py_CLEAR(sflux);
                Py_CLEAR(spectra);

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.update_data:"
                                " failed to locate a Spectra attribute");
                goto failed;
            }
        }
    }

    // made it!!
    status = true;

 failed:

    Py_XDECREF(array);
    Py_XDECREF(sflux);
    Py_XDECREF(spectra);
    Py_XDECREF(data);
    return status;
}

// Now come the routines that are visible to Python

// Computes data equivalent to a map
static PyObject*
doppler_comdat(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *map = NULL, *data = NULL;
    static const char *kwlist[] = {"map", "data", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OO", (char**)kwlist, 
                                    &map, &data))
        return NULL;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.comdat: map is NULL");
        return NULL;
    }
    if(data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.comdat: data is NULL");
        return NULL;
    }

    // Read sizes
    size_t nipix, ndpix;
    if(!npix_map(map, nipix))
        return NULL;
    if(!npix_data(data, ndpix))
        return NULL;

    // declare the variables to hold the Map data
    float* image = new float[nipix];
    std::vector<Nxyz> nxyz;
    std::vector<int> def;
    std::vector<double> vxy, vz, fwhmxy, fwhmz;
    std::vector<std::vector<double> > wavel;
    std::vector<std::vector<float> > gamma, scale;
    double tzero, period, vfine, vpad;

    // read the Map
    if(!read_map(map, image, nxyz, vxy, vz, wavel, gamma, scale,
                 def, fwhmxy, fwhmz, tzero, period, vfine, vpad)){
        delete [] image;
        return NULL;
    }

    // declare the variables to hold the Data data
    std::vector<size_t> nwave, nspec;
    std::vector<std::vector<double> > time;
    std::vector<std::vector<float> > expose;
    std::vector<std::vector<int> > ndiv;
    std::vector<double> fwhm;
    float *flux   = new float[ndpix];
    float *ferr   = new float[ndpix];
    double *wave  = new double[ndpix];

    // read the data
    if(!read_data(data, flux, ferr, wave, nwave,
                  nspec, time, expose, ndiv, fwhm)){
        delete [] wave;
        delete [] ferr;
        delete [] flux;
        delete [] image;
        return NULL;
    }

    // calculate flux equivalent to image
    op(image, nxyz, vxy, vz, wavel, gamma, scale, tzero, period, vfine, vpad,
       flux, wave, nwave, nspec, time, expose, ndiv, fwhm);

    // write modified data back into flux array
    update_data(flux, data);

    // cleanup
    delete[] wave;
    delete[] ferr;
    delete[] flux;
    delete[] image;

    // return the data
    return Py_BuildValue("O", data);
}

// Computes default image from map (test at the moment)
static PyObject*
doppler_comdef(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *map = NULL;
    static const char *kwlist[] = {"map", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "O", (char**)kwlist, 
                                    &map))
        return NULL;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.comdat: map is NULL");
        return NULL;
    }

    // Read sizes
    size_t nipix;
    if(!npix_map(map, nipix))
        return NULL;

    // declare the variables to hold the Map data
    float* input  = new float[nipix];
    float* output = new float[nipix];
    std::vector<Nxyz> nxyz;
    std::vector<int> def;
    std::vector<double> vxy, vz, fwhmxy, fwhmz;
    std::vector<std::vector<double> > wavel;
    std::vector<std::vector<float> > gamma, scale;
    double tzero, period, vfine, vpad;

    // read the Map
    if(!read_map(map, input, nxyz, vxy, vz, wavel, gamma, scale,
                 def, fwhmxy, fwhmz, tzero, period, vfine, vpad)){
        delete [] output;
        delete [] input;
        return NULL;
    }

    float *iptr = input, *optr = output;
    for(size_t nim=0; nim<nxyz.size(); nim++){
        gaussdef(iptr, nxyz[nim], 10., 10., 10., optr);
        iptr += nxyz[nim].ntot();
        optr += nxyz[nim].ntot();
    }

    // write modified image back into the map
    update_map(output, map);

    // cleanup
    delete[] output;
    delete[] input;

    // return the map
    return Py_BuildValue("O", map);
}

// Computes map to data operation transpose to comdat
static PyObject*
doppler_datcom(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *data = NULL, *map = NULL;
    static const char *kwlist[] = {"data", "map", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OO", (char**)kwlist, 
                                    &data, &map))
        return NULL;

    if(data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.datcom: data is NULL");
        return NULL;
    }
    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.datcom: map is NULL");
        return NULL;
    }

    // Read sizes
    size_t nipix, ndpix;
    if(!npix_map(map, nipix))
        return NULL;
    if(!npix_data(data, ndpix))
        return NULL;

    // declare the variables to hold the Map data
    float* image = new float[nipix];
    std::vector<Nxyz> nxyz;
    std::vector<int> def;
    std::vector<double> vxy, vz, fwhmxy, fwhmz;
    std::vector<std::vector<double> > wavel;
    std::vector<std::vector<float> > gamma, scale;
    double tzero, period, vfine, vpad;

    // read the Map
    if(!read_map(map, image, nxyz, vxy, vz, wavel, gamma, scale, 
                 def, fwhmxy, fwhmz, tzero, period, vfine, vpad))
        return NULL;

    // declare the variables to hold the Data data
    std::vector<size_t> nwave, nspec;
    std::vector<std::vector<double> > time;
    std::vector<std::vector<float> > expose;
    std::vector<std::vector<int> > ndiv;
    std::vector<double> fwhm;
    float *flux   = new float[ndpix];
    float *ferr   = new float[ndpix];
    double *wave  = new double[ndpix];

    // read the data
    if(!read_data(data, flux, ferr, wave, nwave,
                  nspec, time, expose, ndiv, fwhm))
        return NULL;

    // overwriting image
    tr(image, nxyz, vxy, vz, wavel, gamma, scale, tzero, period, vfine, vpad,
       flux, wave, nwave, nspec, time, expose, ndiv, fwhm);

    // write modified image back into the map
    update_map(image, map);

    // cleanup
    delete[] wave;
    delete[] ferr;
    delete[] flux;
    delete[] image;

    // return the map
    return Py_BuildValue("O", map);
}

// Carries out mem iterations on a map given data
static PyObject*
doppler_memit(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *data = NULL, *map = NULL;
    static const char *kwlist[] = {"data", "map", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OO", (char**)kwlist, 
                                    &data, &map))
        return NULL;

    if(data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.datcom: data is NULL");
        return NULL;
    }
    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.datcom: map is NULL");
        return NULL;
    }

    // read the number of image and data pixels.
    size_t nipix, ndpix;
    if(!npix_map(map, nipix))
        return NULL;
    if(!npix_data(data, ndpix))
        return NULL;

    // Set memsys buffer
    const size_t MXBUFF = Mem::memsize(nipix, ndpix);
    std::cerr << "MEM buffer size = " << MXBUFF << std::endl;

    Mem::Gbl::st = new float[MXBUFF];
    Dopp::wave   = new double[ndpix];

    // read the Map straight into the MEM buffer and
    // the globals used to get to opus and tropus
    if(!read_map(map, Mem::Gbl::st+Mem::Gbl::kb[0], 
                 Dopp::nxyz, Dopp::vxy, Dopp::vz, 
                 Dopp::wavel, Dopp::gamma, Dopp::scale,
                 Dopp::def, Dopp::fwhmxy, Dopp::fwhmz, 
                 Dopp::tzero, Dopp::period, Dopp::vfine, 
                 Dopp::vpad)){
        delete[] Dopp::wave;
        delete[] Mem::Gbl::st;
        return NULL;
    }

    // read the data & errors straight into buffer
    if(!read_data(data, Mem::Gbl::st+Mem::Gbl::kb[20], 
                  Mem::Gbl::st+Mem::Gbl::kb[21], Dopp::wave, 
                  Dopp::nwave, Dopp::nspec, Dopp::time, 
                  Dopp::expose, Dopp::ndiv, Dopp::fwhm)){
        delete[] Dopp::wave;
        delete[] Mem::Gbl::st;
        return NULL;
    }

    int def = 2, niter=10;
    float c, test, acc=1., caim=1., cnew, s, rnew, snew, sumf;
    float tlim=1.e-4, rmax=0.2;
    int mode;
    if(def == 1){
        mode = 10;
    }else if(def == 2){
        mode = 30;
    }else{
        PyErr_SetString(PyExc_ValueError, "doppler.memit could not understand default option");
        delete[] Dopp::wave;
        delete[] Mem::Gbl::st;
        return NULL;
    }

    for(int it=0; it<niter; it++){
        std::cerr << "\nIteration " << it+1 << std::endl;
        if(def == 2){
            std::cerr << "Computing gaussian default ..." << std::endl;
            float *iptr = Mem::Gbl::st+Mem::Gbl::kb[0], *optr = Mem::Gbl::st+Mem::Gbl::kb[19];
            for(size_t nim=0; nim<Dopp::nxyz.size(); nim++){
                gaussdef(iptr, Dopp::nxyz[nim], 10., 10., 10., optr);
                iptr += Dopp::nxyz[nim].ntot();
                optr += Dopp::nxyz[nim].ntot();
            }
        }
        Mem::memprm(mode,20,caim,rmax,1.,acc,c,test,cnew,s,rnew,snew,sumf);
        if(test < tlim && c <= caim) break;
    }

    // write modified image back into the map
    update_map(Mem::Gbl::st+Mem::Gbl::kb[0], map);

    // cleanup
    delete[] Dopp::wave;
    delete[] Mem::Gbl::st;

    // return the map
    return Py_BuildValue("O", map);
}


// The methods
static PyMethodDef DopplerMethods[] = {

    {"comdat", (PyCFunction)doppler_comdat, METH_VARARGS | METH_KEYWORDS,
     "comdat(map, data)\n\n"
     "computes the data equivalent to 'map' using 'data' as the template\n\n"
    },

    {"comdef", (PyCFunction)doppler_comdef, METH_VARARGS | METH_KEYWORDS,
     "comdef(map)\n\n"
     "computes the default image to 'map'\n\n"
    },

    {"datcom", (PyCFunction)doppler_datcom, METH_VARARGS | METH_KEYWORDS,
     "datcom(data, map)\n\n"
     "computes the transpose operation to comdat for testing\n\n"
    },

    {"memit", (PyCFunction)doppler_memit, METH_VARARGS | METH_KEYWORDS,
     "memit(map, data, niter)\n\n"
     "carries out niter mem iterations on map\n\n"
    },

    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
init_doppler(void)
{
    (void) Py_InitModule("_doppler", DopplerMethods);
    import_array();
}




