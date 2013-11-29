#!/usr/bin/env python

"""
Doppler tomography package

This is an implementation of Doppler tomography in Python.
"""
from __future__ import absolute_import

from .core import *
from .data import *
from .map  import *
from ._doppler import *

__all__ = ['Image', 'Map', 'Spectra', 'Data']
