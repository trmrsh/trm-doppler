// The hard work of Doppler imaging is done by the code here.
#include <Python.h>
#include "numpy/arrayobject.h"
//#include <stdlib>
#include <iostream>
#include <vector>
#include "trm/subs.h"
#include "trm/constants.h"

// Converter function. Assumes that 'obj' points to a
// doppler.Map object. By the end of the routine "p"
// points to 


/*
static int mapconv(PyObject* map, void* p) {

    if(map == NULL) return 0;
    
    // Get the data attribute
    PyObject* data  = PyObject_GetAttrString(data, "data")

    // Create a pointer of the right type
    std::vector<rvm>* vrvmp = (std::vector<rvm>*)p;

    // Get number of elements
    const Py_ssize_t NMASS = PySequence_Size(obj);
    if(!NMASS){
        PyErr_SetString(PyExc_ValueError, "orbits.integrate: there need to be at least 2 particles");
        return 0;
    }

    // Set size of output structure
    vrvmp->resize(NMASS);

    int status = 1;

    for(Py_ssize_t i=0; i<NMASS; i++){
        PyObject* rvmp = PySequence_GetItem(obj, i);
        if(rvmp){
            PyObject* pr  = PyObject_GetAttrString(rvmp, "r");
            PyObject* px  = PyObject_GetAttrString(pr, "x");
            PyObject* py  = PyObject_GetAttrString(pr, "y");
            PyObject* pz  = PyObject_GetAttrString(pr, "z");

            PyObject* pv  = PyObject_GetAttrString(rvmp, "v");
            PyObject* pvx = PyObject_GetAttrString(pv, "x");
            PyObject* pvy = PyObject_GetAttrString(pv, "y");
            PyObject* pvz = PyObject_GetAttrString(pv, "z");

            PyObject* pm  = PyObject_GetAttrString(rvmp, "m");
            PyObject* pls = PyObject_GetAttrString(rvmp, "ls");
            PyObject* pvs = PyObject_GetAttrString(rvmp, "vs");
            PyObject* pri = PyObject_GetAttrString(rvmp, "ri");

            if(pr && px && py && pz && pv && pvx && pvy && pvz && pm && pls && pvs && pri){
                // Next functions can convert integers to doubles
                double x  = PyFloat_AsDouble(px);
                double y  = PyFloat_AsDouble(py);
                double z  = PyFloat_AsDouble(pz);
                double vx = PyFloat_AsDouble(pvx);
                double vy = PyFloat_AsDouble(pvy);
                double vz = PyFloat_AsDouble(pvz);
                double m  = PyFloat_AsDouble(pm);
                double ls = PyFloat_AsDouble(pls);
                double vs = PyFloat_AsDouble(pvs);
                double ri = PyFloat_AsDouble(pri);

                if(!PyErr_Occurred()){
                    (*vrvmp)[i] = rvm(x,y,z,vx,vy,vz,m,ls,vs,ri);
                }else{
                    status = 0;
                }
            }else{
                status = 0;
            }

            Py_XDECREF(px);
            Py_XDECREF(py);
            Py_XDECREF(pz);
            Py_XDECREF(pr);

            Py_XDECREF(pvx);
            Py_XDECREF(pvy);
            Py_XDECREF(pvz);
            Py_XDECREF(pv);

            Py_XDECREF(pm);
            Py_XDECREF(pls);
            Py_XDECREF(pvs);
            Py_XDECREF(pri);

            Py_XDECREF(rvmp);
        }else{
            status = 0;
        }
        if(!status) break;
    }

    return status;
}
*/

/* read_map -- reads the data contained in a PyObject representing a Doppler map
 * returning the contents in a series of vectors containing the information for
 * each image of the map. To be called soon after sending a Map object into a 
 * C routine.
 *
 * Arguments::
 *
 *  map     :  the Doppler map (input)
 *  images  :  vector of pointers to image data (output)
 *  nx      :  vector of X-dimensions (output)
 *  ny      :  vector of Y-dimensions (output)
 *  nz      :  vector of Z-dimensions (output)
 *  vxy     :  vector of VX-VY pixels sizes for each image (output)
 *  vz      :  vector of VZ spacings for each image (output)
 *  wave    :  vector of vectors of wavelengths for each image (output)
 *  gamma   :  vector of vectors of systemic velocities for each image (output)
 *  scale   :  vector of vectors of systemic velocities for each image (output)
 *
 *  Returns True/False according to success. If False, the outputs above
 *  will not be correctly assigned. If False, a Python exception is raised
 *  and you should return NULL from the calling routine.
 */

bool
read_map(PyObject *map, std::vector<float*>& images, std::vector<int>& nx, std::vector<int>& ny,
         std::vector<int>& nz, std::vector<double>& ny,, std::vector<double>& ny, 
         std::vector<std::vector<double> >& wave, std::vector<std::vector<double> >& gamma,
         std::vector<std::vector<double> >& scale)
{
    
    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.read_map: map is NULL");
        return false;
    }
    
    // initialise attribute pointers
    PyObject *data=NULL, *image=NULL, *idata=NULL, *iwave=NULL, *igamma=NULL, *ivxy=NULL;
    PyObject *iscale=NULL, *ivz=NULL;
    
    // clear the output vectors
    images.clear();
    nx.clear();
    ny.clear();
    nz.clear();
    vxy.clear();
    vz.clear();
    wave.clear();
    gamma.clear();
    scale.clear();

    // get the data attribute. 
    data  = PyObject_GetAttrString(map, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError, "doppler.read_map: map has no attribute called data");
        goto failed;
    }

    // data should be a list of Images, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError, "doppler.read_map: map.data is not a sequence");
        goto failed;
    }

    // define scope to avoid compilation error
    {
        // work out the number of Images
        Py_ssize_t nimage = PySequence_Size(data);
        std::cerr << "nimage = " << nimage << std::endl;
        
        // loop through Images
        for(Py_ssize_t i=0; i<nimage; i++){
            image  = PySequence_GetItem(data, i);
            if(!image){
                PyErr_SetString(PyExc_ValueError, "doppler.read_map: failed to access element in map.data");
                goto failed;
            }

            // get attributes of the image
            idata  = PyObject_GetAttrString(image, "data");
            iwave  = PyObject_GetAttrString(image, "wave");
            igamma = PyObject_GetAttrString(image, "gamma");
            ivxy   = PyObject_GetAttrString(image, "vxy");
            
            if(idata && iwave && igamma && ivxy && PyArray_Check(idata) && 
               PyArray_Check(iwave) && PyArray_Check(igamma)){
                
                // store pointer to the data (note: avoid copying it)
                images.push_back((float*)PyArray_DATA(idata));

                // store dimensions
                int nddim = PyArray_NDIM(idata);
                npy_intp *ddims = PyArray_DIMS(idata);
                nx.push_back(ddims[0]);
                ny.push_back(ddims[1]);
                if(nddim == 2){
                    nz.push_back(1);
                }else{
                    nz.push_back(ddims[1]);
                }

                // copy over wavelengths
                int nwdim = PyArray_NDIM(iwave);
                if(nwdim != 1){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map: arrays of wavelength must be 1D");
                    goto failed;
                }
                npy_intp *wdims = PyArray_DIMS(iwave);
                double *pwave = (double*)PyArray_DATA(iwave);
                for(npy_intp j=0; j<wdims[0]; j++)
                    wave.push_back(pwave[j]);

                // copy over systemic velocities
                int ngdim = PyArray_NDIM(igamma);
                if(ngdim != 1){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map: arrays of systemic velocities must be 1D");
                    goto failed;
                }
                npy_intp *gdims = PyArray_DIMS(igamma);
                if(gdims[0] != wdims[0]){
                    PyErr_SetString(PyExc_ValueError, "doppler.read_map: systemic velocity array must match size of wavelength array");
                    goto failed;
                }
                double *pgamma = (double*)PyArray_DATA(igamma);
                for(npy_intp j=0; j<wdims[0]; j++)
                    gamma.push_back(pgamma[j]);

                // store the Vx-Vy pixel size
                vxy.push_back(PyFloat_AsDouble(ivxy));

                // store the Vz spacing
                if(nddim == 3){
                    ivz = PyObject_GetAttrString(image, "vz");
                    if(!ivz){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map: failed to access vz in Image");
                        goto failed;
                    }
                    vz.push_back(PyFloat_AsDouble(ivz));
                }

                if(wdims[0] > 1){
                    // copy over scale factors
                    iscale = PyObject_GetAttrString(image, "scale");
                    if(!iscale){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map: failed to access scale in Image");
                        goto failed;
                    }
                    int nsdim = PyArray_NDIM(iscale);
                    if(nsdim != 1){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map: arrays of scale factors must be 1D");
                        goto failed;
                    }
                    npy_intp *sdims = PyArray_DIMS(iscale);
                    if(sdims[0] != wdims[0]){
                        PyErr_SetString(PyExc_ValueError, "doppler.read_map: scale factor array must match size of wavelength array");
                        goto failed;
                    }
                    double *pscale = (double*)PyArray_DATA(iscale);
                    for(npy_intp j=0; j<wdims[0]; j++)
                        scale.push_back(pscale[j]);
                }
                
                Py_XDECREF(image);
                Py_XDECREF(idata);
                Py_XDECREF(iwave);
                Py_XDECREF(igamma);
                Py_XDECREF(ivxy);
                Py_XDECREF(ivz);
                Py_XDECREF(iscale);
                image = idata = iwave = igamma = ixvy = ivz = iscale = NULL;

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.read_map: failed to locate an Image attribute or one or more had the wrong type");
                goto failed;
            }
        }
    }

    PyErr_SetString(PyExc_ValueError, "doppler.read_map: have not implemented a proper return yet");

 failed:
    Py_XDECREF(data);
    Py_XDECREF(image);
    Py_XDECREF(idata);
    Py_XDECREF(iwave);
    Py_XDECREF(igamma);
    Py_XDECREF(ivxy);    
    Py_XDECREF(ivz);    
    Py_XDECREF(iscale);
    return NULL;
}

/*
 * The next routine gets called by the Python interpreter. This is
 * the primary Python/C++ interface routine.
 */

static PyObject*
doppler_tester(PyObject *self, PyObject *args, PyObject *kwords)
{

    // Process and check arguments
    PyObject *map  = NULL;
    static const char *kwlist[] = {"map", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwords, "O", (char**)kwlist, &map))
        return NULL;

    if(map == NULL){
        PyErr_SetString(PyExc_ValueError, "doppler.tester: map is NULL");
        return NULL;
    }
    
    // initialise pointers
    PyObject *data=NULL, *image=NULL, *idata=NULL;
    PyObject *iwave=NULL, *igamma=NULL, *ivxy=NULL;
    PyObject *iscale=NULL, *ivz=NULL;
    
    // store things
    std::vector<float*> images;
    std::vector<int> nx, ny, nz;
    std::vector<std::vector<double>> wave, gamma, scale;
    std::vector<double> vxy, vz;

    // Get the data attribute. 
    data  = PyObject_GetAttrString(map, "data");
    if(!data){
        PyErr_SetString(PyExc_ValueError, "doppler.tester: map has no attribute called data");
        goto failed;
    }

    // data should be a list of Images, so lets test it
    if(!PySequence_Check(data)){
        PyErr_SetString(PyExc_ValueError, "doppler.tester: map.data is not a sequence");
        goto failed;
    }

    // define scope to avoid compilation error
    {
        // work out the number of Images
        Py_ssize_t nimage = PySequence_Size(data);
        std::cerr << "nimage = " << nimage << std::endl;
        
        // loop through Images
        for(Py_ssize_t i=0; i<nimage; i++){
            image  = PySequence_GetItem(data, i);
            if(!image){
                PyErr_SetString(PyExc_ValueError, "doppler.tester: failed to access element in map.data");
                goto failed;
            }
            idata  = PyObject_GetAttrString(image, "data");
            iwave  = PyObject_GetAttrString(image, "wave");
            igamma = PyObject_GetAttrString(image, "gamma");
            ivxy   = PyObject_GetAttrString(image, "vxy");
            
            if(idata && iwave && igamma && ivxy && PyArray_Check(idata) && 
               PyArray_Check(iwave) && PyArray_Check(igamma)){
                
                int nddim = PyArray_NDIM(idata);
                npy_intp *ddims = PyArray_DIMS(idata);
                std::cerr << "Image " << i << " has " << nddim << " dimensions." << std::endl;
                nx.push_back(ddims[0]);
                ny.push_back(ddims[1]);
                if(nddim == 2){
                    nz.push_back(1);
                }else{
                    nz.push_back(ddims[1]);
                }

                for(int j=0;j<nddim;j++)
                    std::cerr << "Dimension " << j << " = " << ddims[j] << std::endl;
                
                double vxy  = PyFloat_AsDouble(ivxy);
                std::cerr << "VXY = " << vxy << std::endl;
                
                Py_DECREF(image);
                Py_DECREF(idata);
                Py_DECREF(iwave);
                Py_DECREF(igamma);
                Py_DECREF(ivxy);
                image = idata = iwave = igamma = ixvy = NULL;

            }else{
                PyErr_SetString(PyExc_ValueError, "doppler.tester: failed to locate an Image attribute or one or more had the wrong type");
                goto failed;
            }
        }

    }

    PyErr_SetString(PyExc_ValueError, "doppler.tester: have not implemented a proper return yet");

 failed:
    Py_XDECREF(data);
    Py_XDECREF(image);
    Py_XDECREF(idata);
    Py_XDECREF(iwave);
    Py_XDECREF(igamma);
    Py_XDECREF(ivxy);    
    Py_XDECREF(ivz);    
    Py_XDECREF(iscale);
    return NULL;
}

// The methods

static PyMethodDef DopplerMethods[] = {

    {"tester", (PyCFunction)doppler_tester, METH_VARARGS | METH_KEYWORDS,
     "tester(map)\n\n"
     "testing routine\n\n"
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
 * map   -- multiple image data, stored in a 1D array for compatibility with mem routines.
 * nxyz  -- dimensions of each image.
 * wave  -- wavelengths for each image.
 * gamma -- systemic velocities for each wavelength
 * scale -- scaling factors for each wavelength
 */

/*
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

    
*/



