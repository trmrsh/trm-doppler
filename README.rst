trm.doppler is a Python implementation of Doppler tomography.

Pre-requisites for installation:

C++ libraries:

  fftw    -- for taking FFTs, standard package.
  cpp-mem -- my translation of memsys into C++.

Python:

  astropy -- general astronomy package which is needed
             for FITS.

Some scripts need other packages. e.g. The volume rendering script
vrend.py needs 's2plot' (but I intend to explore other options for
3D visualiation so don't rely on this).

This is the Python3 version. I am making no attempt to be backwards
compatible with Python2. I strongly recommend installing with 'pip' as:

pip install . --user


