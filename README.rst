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
3D visualiation so don't reply on this).

This is the Python3 version. I am making no attempt to be backwards
compatible with Python2. I recommend installing with 'pip' as in

pip install . --user

run in the directory in which the setup.py file is to be found rather
than 'python setup.py install --user' because 'pip' tracks the files
installed allowing you to uninstall them.

