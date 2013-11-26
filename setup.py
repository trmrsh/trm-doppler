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
    print >>sys.stderr, "Environment variable TRM_SOFTWARE pointing to location of shareable libraries and includes not defined!"

include_dirs.append(numpy.get_include())

subs = Extension('trm.subs._subs',
                 define_macros   = [('MAJOR_VERSION', '0'),
                                    ('MINOR_VERSION', '1')],
                 undef_macros    = ['USE_NUMARRAY'],
                 include_dirs    = include_dirs,
                 library_dirs    = library_dirs,
                 runtime_library_dirs = library_dirs,
                 libraries       = ['subs'],
                 sources         = [os.path.join('trm', 'subs', 'subs.cc')])

setup(name='trm.subs',
      version='0.3',
      packages = ['trm', 'trm.subs', 'trm.subs.input', 'trm.subs.plot', 'trm.subs.smtp', 'trm.subs.cpp', 'trm.subs.dvect'],
      ext_modules=[subs],
      scripts=['scripts/hms2decimal.py'],

      # metadata
      author='Tom Marsh',
      author_email='t.r.marsh@warwick.ac.uk',
      description="Python basic utility module",
      url='http://www.astro.warwick.ac.uk/',
      long_description="""
subs provides an interface to various basic routines as well as a set of routines of general utility.
""",
      cmdclass = cmdclass

      )

