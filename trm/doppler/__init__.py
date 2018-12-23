# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Doppler tomography

This is an implementation of Doppler tomography in Python. It
allows flexible configuration of both images and input data.
"""

from .core  import *
from . import data
from . import map
from . import grid
from ._doppler import *
from .derived import *
from . import scripts

#__all__ = ['Image', 'Map', 'Spectra', 'Data', 'Grid']
