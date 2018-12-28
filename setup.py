import os, sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import numpy

# get round irritating compiler warning
class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()

library_dirs = []
include_dirs = []

# need to direct to where includes and  libraries are
if 'TRM_SOFTWARE' in os.environ:
    library_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'lib64'))
    library_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'lib'))
    include_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'include'))
else:
    print >>sys.stderr, \
        "Environment variable TRM_SOFTWARE pointing to location of shareable libraries and includes not defined!"

include_dirs.append(numpy.get_include())

doppler = Extension(
    'trm.doppler._doppler',
    define_macros = [('MAJOR_VERSION', '1'),('MINOR_VERSION', '0')],
    undef_macros = ['USE_NUMARRAY'],
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    runtime_library_dirs = library_dirs,
    libraries = ['mem', 'fftw3', 'gomp', 'm'],
    extra_compile_args=['-fopenmp'],
    sources = [os.path.join('trm', 'doppler', 'doppler.cc')]
)

setup(name='trm.doppler',
      version='1.0',
      packages = ['trm', 'trm.doppler', 'trm.doppler.scripts'],
      ext_modules=[doppler,],
      entry_points={
          'console_scripts' : [
              'comdat=trm.doppler.scripts.comdat:comdat',
              'comdef=trm.doppler.scripts.comdef:comdef',
              'drlimit=trm.doppler.scripts.drlimit:drlimit',
              'entropy=trm.doppler.scripts.entropy:entropy',
              'makedata=trm.doppler.scripts.makedata:makedata',
              'makegrid=trm.doppler.scripts.makegrid:makegrid',
              'makemap=trm.doppler.scripts.makemap:makemap',
              'memit=trm.doppler.scripts.memit:memit',
              'mol2dopp=trm.doppler.scripts.mol2dopp:mol2dopp',
              'mspruit=trm.doppler.scripts.mspruit:mspruit',
              'optscl=trm.doppler.scripts.optscl:optscl',
              'precover=trm.doppler.scripts.precover:precover',
              'psearch=trm.doppler.scripts.precover:psearch',
              'svdfit=trm.doppler.scripts.svdfit:svdfit',
              'svd=trm.doppler.scripts.svdfit:svd',
              'trtest=trm.doppler.scripts.trtest:trtest',
              'vrend=trm.doppler.scripts.vrend:vrend',
          ],
      },

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python Doppler tomography module",
      url='http://www.astro.warwick.ac.uk/',
      long_description="""
doppler is an implementation of Doppler tomography package. It has many advantages over the
former set of F77 routines.
""",
      cmdclass={'build_ext': BuildExt}

      )

