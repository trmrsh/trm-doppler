Features of the package
=======================

trm.doppler represents a considerable advance upon the earlier F77 code used
in Marsh & Horne (1988). Here are the main features and capabilities of the
new package with "(*new*)" or "(*old*)" at the end of each section to indicate
whether a given feature is new or not.

Multiple datasets
-----------------

trm.doppler loads data from a single FITS file, but this can be constructed
from many independent datasets. To take a slightly artificial example,
consider mapping data covering Hbeta on one arm of a spectrograph and Halpha
on the other. These could have different resolutions, pixel scales,
etc. Normally one would separately map each line, but suppose the
signal-to-noise ratio was poor and you wanted to use a single image to model
both lines, perhaps with a different scale for each. trm.doppler can cope with
this case, and you could also add in datasets from different instruments,
small snatches of data etc. The only restriction now is that the spectra of
any one dataset must have the same number of pixels so that the data can be
passed as a regular 2D array, 1 spectrum per row. Different datasets however
can have differing numbers of pixels. (*new*)

Raw wavelength scales
---------------------

To use the F77 code one had to interpolate the data onto a logarithmic scale
giving a uniform velocity step per pixel. The same scale was required of all
spectra. trm.doppler has done away with this so that for each dataset one now
passes includes an array of wavelengths with the same shape as the fluxes and
flux uncertainties. This means that one can use the spectra in pretty much the
raw format they had at the telescope. (*new*)

3D
--

trm.doppler can use 3D images that vary in the Z-velocity component as well.
(*new*)

FITS
----

Both the data and map files are now in FITS format allowing inspection with
such well-known tools as *fv* and *ds9* as well as using *astropy.io.fits* for
programmatic access perhaps, although see next for a simpler way.
(*new*)

Python
------

The package includes classes for loading the data and map files into
Python. The classes allow easy access to everything contained in them.
This makes the development of scripts, both general and for one-off specialist
use much easier than under the F77 version.
(*new*)

Blended lines
-------------

trm.doppler can cope with blended lines by using multiple images, or one can
assign multiple atomic lines to the same image. For instance if one thought
that all the Balmer lines showed essentially the same pattern, albeit with
different strengths, one can use a single image with scaling factors to
describe them. (*old*)

Finite duration exposures
-------------------------

Finite duration exposures can be modelled by computing multiple projected
spread uniformly through an exposure. (*old*)

