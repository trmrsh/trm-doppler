// The hard work of Doppler imaging is done by the code here. Much of code
// (~1000 lines or so) is unedifying "boilerplate" for interfacing Python and
// C++, and in particular to the C++ mem routines. The minimal number of
// routines needed to do this is provided. Most manipulations of the objects
// are better left to Python. The main routines that do the hard work in mem
// are op and tr that carry out the image to data projection and its
// transpose.
//
// The routines in this file are as follows::
//
//  npix_map    : calculates the number of pixels needed for the images
//  read_map    : makes internal contents of a Map object accessible from C++
//  update_data : overwrites image array(s) of a Map object.
//  npix_data   : calculates the number of pixels needed for fluxes, errors,
//                wave
//  read_data   : makes internal contents of a Data object accessible from C++
//  update_data : overwrites flux array(s) of a Data object.
//  op          : image to data transform routine.

#include <Python.h>
#include <iostream>
#include <vector>
#include "numpy/arrayobject.h"
#include <cmath>
#include <complex.h>
#include <fftw3.h>

// Speed of light in km/s
const double CKMS = 299792.458;

// Ratio FWHM/sigma for a gaussian
const double EFAC = 2.354820045;

// simple structure for containing image dimensions
struct Nxyz{
    Nxyz(size_t nx, size_t ny, size_t nz=1) : nx(nx), ny(ny), nz(nz) {}
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

    // Each image is first projected onto a finely-spaced array which is
    // later blurred. This should be faster than blurring each image pixel.
    // Start by computing the size of the fine array
    size_t nimage = nxyz.size(), nx, ny, nz;
    double vmax = 0.;
    for(size_t nim=0; nim<nimage; nim++){
        nx = nxyz[nim].nx;
        ny = nxyz[nim].ny;
        nz = nxyz[nim].nz;
        vmax = std::max(std::sqrt(std::pow(vxy[nim],2)
                                  *(std::pow(nx,2)+std::pow(ny,2))+
                                  std::pow(vz[nim]*nz,2)), vmax);
    }
    vmax += 2.*vpad;

    // NFINE is the number of pixels needed for the fine array.
    const int nfine = std::max(5,int(vmax/vfine));

    // adjust vmax to be the maximum velocity of the fine array
    vmax = nfine*vfine/2.;

    // Now we need to know how many pixels at most are needed for the blurring
    // function. Loop over the data sets. Also take chance to compute number
    // of data pixels
    size_t nblurr = 0, ndpix = 0;
    double psigma;
    for(size_t nd=0; nd<nspec.size(); nd++){
        // sigma of blurring in terms of fine array pixels
        psigma = fwhm[nd]/vfine/EFAC;

        // total blurr width will be 2*nblurr+1
        nblurr = std::max(nblurr,size_t(round(5.*psigma)));
        ndpix += nwave[nd]*nspec[nd];
    }

    // The blurring is carried out with FFTs requiring zero padding.  Thus the
    // actual number of pixels to grab is the nearest power of 2 larger than
    // nfine+2*nblurr+1. Call this NFINE.
    const size_t NFINE =
        size_t(std::pow(2,int(std::ceil(std::log(nfine+2*nblurr+1)/
                                        std::log(2.)))));

    std::cerr << "vmax = " << vmax << ", nfine = " << nfine <<
        ", NFINE = " << NFINE << std::endl;

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

    // establish plans for the fine pixel FFT and the final inverse
    fftw_plan pforw = fftw_plan_dft_r2c_1d(NFINE, fine, fpfft, 
                                           FFTW_ESTIMATE);
    fftw_plan pback = fftw_plan_dft_c2r_1d(NFINE, fpfft, fine, 
                                           FFTW_ESTIMATE);
    // various local variables
    size_t k, m, n;
    double norm, prf, fp1, fp2, v1, v2, sum;
    double wv, gm, sc;

    // pixel index in fine array
    int nf, ifp1, ifp2;

    // phase, cosine, sine, weight factor
    double phase, cosp, sinp, weight;

    // image indices
    size_t ix, iy, iz;

    // image velocity steps
    double vxyi, vzi = 0.;

    // steps and offset in fine pixel space
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
        for(m=1, n=2*nblurr; m<nblurr; m++, n--){
            prf = std::exp(-std::pow(m/psigma,2)/2.);
            blurr[m] = blurr[n] = prf;
            norm += 2.*prf;
        }
        
        // normalise
        for(m=0;m<2*nblurr+1;m++)
            blurr[m] /= norm;
        
        // create plan
        fftw_plan p = fftw_plan_dft_r2c_1d(NFINE, blurr, bfft, 
                                           FFTW_ESTIMATE);
        // execute it
        fftw_execute(p);
        
        // destroy the plan. FFT now contained in bfft array.
        fftw_destroy_plan(p);
        
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
                        pzoff  = double(NFINE-1)/2. - vzi*double(nz-1)/2./vfine;
                        pzstep = vzi/vfine;
                    }else{
                        pzoff  = double(NFINE-1)/2.;
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
                                ix<nx; ix++, pxoff+=pxstep, iptr++){
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

                // Take the inverse FFT (?? may need normalisation by NFINE ??)
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
                    // projection array ??
                    for(m=1; m<nwave[nd]-1; m++){
                        // velocity of right-hand side of pixel
                        v2 = CKMS*((wptr[m]+wptr[m+1])/2./wv-1.)-gm;

                        if(v1 < vmax || v2 > -vmax){

                            // fp1, fp2 -- start and end limits of data pixel
                            // in fine array pixels 
                            fp1  = v1/vfine + double(nfine-1)/2.;
                            fp2  = v2/vfine + double(nfine-1)/2.;

                            // ifp1, ifp2 -- fine pixel range fully inside the data
                            // in traditional C form (i.e. ifp1<= <ifp2)
                            ifp1 = std::max(nfine,std::min(0,int(std::ceil(fp1+0.5))));
                            ifp2 = std::max(nfine,std::min(0,int(std::ceil(fp2+0.5))));

                            // add full pixels
                            sum = 0.;
                            for(nf=ifp1; nf<ifp2; nf++) sum += fine[nf];

                            // add partial pixels
                            if(ifp1 > 0) sum += (ifp1-0.5-fp1)*fine[ifp1-1];
                            if(ifp2 < nfine) sum += (fp2-ifp2+1.5)*fine[ifp2-1];

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
    delete[] tfine;
    delete[] fine;
    fftw_free(fpfft);
    fftw_free(bfft);
    delete[] blurr;
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
 *  image   :  C-style array (output)
 *             Memory will be allocated internally. It should be deleted in
 *             the calling routine once you are done with it. It should not
 *             point to any memory before entry unless you have another
 *             reference to it since the pointer value will be overwritten. If
 *             there are multiple images they will be written sequentially
 *             into this array.
 *  nxyz    :  vector of image dimensions (output)
 *  vxy     :  vector of VX-VY pixels sizes for each image (output)
 *  vz      :  vector of VZ spacings for each image (output)
 *  wave    :  vector of vectors of wavelengths for each image (output)
 *  gamma   :  vector of vectors of systemic velocities for each image (output)
 *  scale   :  vector of vectors of scale factors for each image (output)
 *  tzero   :  zeropoint of ephemeris (output)
 *  period  :  period of ephemeris (output)
 *  vfine   :  km/s for fine projection array (output)
 *  vpad    :  km/s extra padding for fine array (output)
 *  npix    :  total number of image pixels, useful for allocating memory (output)
 *
 *  Returns true/false according to success. If false, the outputs above
 *  will be useless. If false, a Python exception is raised and you should
 *  return NULL from the calling routine. In this case any memory allocated
 *  internally will have been deleted before exiting and you should not attempt
 *  to free it in the calling routine.
 */

bool
read_map(PyObject *map, float*& images, std::vector<Nxyz>& nxyz,
         std::vector<double>& vxy, std::vector<double>& vz,
         std::vector<std::vector<double> >& wave,
         std::vector<std::vector<float> >& gamma,
         std::vector<std::vector<float> >& scale,
         double& tzero, double& period, double& vfine, double& vpad,
         size_t& npix)
{

    bool status = false;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.read_map: map is NULL");
        return status;
    }

    // determine size needed for images, allocates
    // space and a temporary copy to the pointer
    if(!npix_map(map, npix))
        return status;
    images = new float[npix];
    float *timages=images;

    // initialise attribute pointers
    PyObject *data=NULL, *image=NULL, *idata=NULL, *iwave=NULL;
    PyObject *igamma=NULL, *ivxy=NULL, *iscale=NULL, *ivz=NULL;
    PyObject *itzero=NULL, *iperiod=NULL, *ivfine=NULL, *ivpad=NULL;
    PyArrayObject *darray=NULL, *warray=NULL, *garray=NULL;
    PyArrayObject *sarray=NULL;

    // clear the output vectors
    nxyz.clear();
    vxy.clear();
    vz.clear();
    wave.clear();
    gamma.clear();
    scale.clear();

    // get attributes.
    itzero  = PyObject_GetAttrString(map, "tzero");
    iperiod = PyObject_GetAttrString(map, "period");
    ivfine  = PyObject_GetAttrString(map, "vfine");
    ivpad   = PyObject_GetAttrString(map, "vpad");
    data    = PyObject_GetAttrString(map, "data");
    if(!data || !itzero || !iperiod || !vfine || !vpad){
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
            iscale = PyObject_GetAttrString(image, "scale");
            ivxy   = PyObject_GetAttrString(image, "vxy");
            ivz    = PyObject_GetAttrString(image, "vz");

            if(idata && iwave && igamma && ivxy && iscale && ivz){

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
                double *gptr = (double*) PyArray_DATA(garray);
                std::vector<float> gammas(gptr,gptr+PyArray_SIZE(garray));
                gamma.push_back(gammas);

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
                    double *sptr = (double*) PyArray_DATA(sarray);
                    std::vector<float> scales(sptr,sptr+PyArray_SIZE(sarray));
                    scale.push_back(scales);
                }

                // clear temporaries
                Py_CLEAR(sarray);
                Py_CLEAR(garray);
                Py_CLEAR(warray);
                Py_CLEAR(darray);
                Py_CLEAR(ivz);
                Py_CLEAR(ivxy);
                Py_CLEAR(iscale);
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
    Py_XDECREF(ivxy);
    Py_XDECREF(iscale);
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
 *              data sets which are written sequentially. The pointer itself
 *              will be overwritten so it should not point to any memory on
 *              entry, or you should at least have a copy of the pointer. If
 *              you call the routine multiple times, you should delete or
 *              retain a reference to the memory between calls.
 *  ferr     :  pointer to 1D array where flux errors will be stored (output)
 *              Same remarks about memory for flux apply to ferr.
 *  wave     :  pointer to 1D array where wavelengths will be stored (output)
 *              Same remarks about memory for flux apply to wave.
 *  npix     :  Number of pixels in flux, ferr and wave
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
read_data(PyObject *Data, float*& flux, float*& ferr, double*& wave,
          size_t& npix,
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

    // determine size needed for flux, ferr and wave
    // allocate memory
    if(!npix_data(Data, npix))
        return status;
    flux = new float[npix];
    ferr = new float[npix];
    wave = new double[npix];

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


// test routines

// tests data access routines
static PyObject*
doppler_datatest(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *data = NULL;
    static const char *kwlist[] = {"data", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwords, "O", (char**)kwlist, &data))
        return NULL;

    if(data == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.datatest: data is NULL");
        return NULL;
    }

    // declare the variables to hold the Data
    std::vector<size_t> nwave, nspec;
    std::vector<std::vector<double> > time;
    std::vector<std::vector<float> > expose;
    std::vector<std::vector<int> > ndiv;
    std::vector<double> fwhm;
    float *flux, *ferr;
    double *wave;
    size_t ndpix;

    // read the data
    if(!read_data(data, flux, ferr, wave, ndpix, nwave,
                  nspec, time, expose, ndiv, fwhm))
        return NULL;

    // modify
    for(size_t j=0; j<ndpix; j++)
        flux[j] += 10.;

    // write modified fluxes back into data
    update_data(flux, data);

    // cleanup
    delete[] flux;
    delete[] ferr;
    delete[] wave;

    // return to Python
    return Py_BuildValue("O", data);
}

// tests map access routines
static PyObject*
doppler_maptest(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *map = NULL;
    static const char *kwlist[] = {"map", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "O", (char**)kwlist, &map))
        return NULL;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.maptest: map is NULL");
        return NULL;
    }

    // declare the variables to hold the Map
    float* images;
    std::vector<Nxyz> nxyz;
    std::vector<double> vxy, vz;
    std::vector<std::vector<double> > wave;
    std::vector<std::vector<float> > gamma, scale;
    double tzero, period, vfine, vpad;
    size_t nipix;

    // read the Map
    if(!read_map(map, images, nxyz, vxy, vz, wave, gamma, scale, 
                 tzero, period, vfine, vpad, nipix))
        return NULL;

    // modify it
    for(size_t j=0; j<nipix; j++)
        images[j] += 10.;

    // write modified fluxes back into data
    update_map(images, map);

    // cleanup
    delete[] images;

    // return to Python
    return Py_BuildValue("O", map);
}

// computes data equivalent to a map
static PyObject*
doppler_comdat(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *map = NULL, *data = NULL;
    static const char *kwlist[] = {"map", "data", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "O", (char**)kwlist, 
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

    // declare the variables to hold the Map data
    float* image;
    std::vector<Nxyz> nxyz;
    std::vector<double> vxy, vz;
    std::vector<std::vector<double> > wavel;
    std::vector<std::vector<float> > gamma, scale;
    double tzero, period, vfine, vpad;
    size_t nipix;


    // read the Map
    if(!read_map(map, image, nxyz, vxy, vz, wavel, gamma, scale,
                 tzero, period, vfine, vpad, nipix))
        return NULL;

    // declare the variables to hold the Data data
    std::vector<size_t> nwave, nspec;
    std::vector<std::vector<double> > time;
    std::vector<std::vector<float> > expose;
    std::vector<std::vector<int> > ndiv;
    std::vector<double> fwhm;
    float *flux, *ferr;
    double *wave;
    size_t ndpix;

    // read the data
    if(!read_data(data, flux, ferr, wave, ndpix, nwave,
                  nspec, time, expose, ndiv, fwhm))
        return NULL;

    // calculate flux equivalent to image
    op(image, nxyz, vxy, vz, wavel, gamma, scale, tzero, period, vfine, vpad,
       flux, wave, nwave, nspec, time, expose, ndiv, fwhm);

    // write modified image back into the map
    update_map(image, map);

    // cleanup
    delete[] image;
    delete[] flux;
    delete[] ferr;
    delete[] wave;

    // return the map
    return Py_BuildValue("O", map);
}


// The methods
static PyMethodDef DopplerMethods[] = {

    {"maptest", (PyCFunction)doppler_maptest, METH_VARARGS | METH_KEYWORDS,
     "maptest(map)\n\n"
     "tests simple modification of a Map\n\n"
    },

    {"datatest", (PyCFunction)doppler_datatest, METH_VARARGS | METH_KEYWORDS,
     "datatest(data)\n\n"
     "tests simple modification of a Data\n\n"
    },

    {"comdat", (PyCFunction)doppler_comdat, METH_VARARGS | METH_KEYWORDS,
     "comdat(map, data)\n\n"
     "computes the data equivalent to 'map' using 'data' as the template\n\n"
    },

    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
init_doppler(void)
{
    (void) Py_InitModule("_doppler", DopplerMethods);
    import_array();
}

/*
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

    
*/



