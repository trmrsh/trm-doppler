from distutils.core import setup, Extension
import os, numpy

try:
    from sdist import sdist
    cmdclass = {'sdist': sdist}
except:
    cmdclass = {}

library_dirs = []
include_dirs = []

# need to direct to where includes and  libraries are
if os.environ.has_key('TRM_SOFTWARE'):
    library_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'lib'))
    include_dirs.append(os.path.join(os.environ['TRM_SOFTWARE'], 'include'))
else:
    print >>sys.stderr, \
        "Environment variable TRM_SOFTWARE pointing to location of shareable libraries and includes not defined!"

include_dirs.append(numpy.get_include())

doppler = Extension('trm.doppler._doppler',
                    define_macros   = [('MAJOR_VERSION', '0'),
                                       ('MINOR_VERSION', '1')],
                    undef_macros    = ['USE_NUMARRAY'],
                    include_dirs    = include_dirs,
                    library_dirs    = library_dirs,
                    runtime_library_dirs = library_dirs,
                    libraries       = ['subs'],
                    sources         = [os.path.join('trm', 'doppler', 'doppler.cc')])

setup(name='trm.doppler',
      version='0.1',
      packages = ['trm', 'trm.doppler'],
      ext_modules=[doppler],
#      scripts=['scripts/hms2decimal.py'],

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python Doppler tomography module",
      url='http://www.astro.warwick.ac.uk/',
      long_description="""
doppler is an implementation of Doppler tomography package. It has many advantages over the
former set of F77 routines.
""",
      cmdclass = cmdclass

      )

