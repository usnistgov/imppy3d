from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


# To compile the Cython extensions, run the following Python command
# in the terminal, which will execute this setup file. You must, of course,
# be in the same directory as this file.
#
# python cython_build_setup.py build_ext --build-lib ./bin
#
# This will build the Python/Cython source files (with .pyx extensions)
# in "../functions/cython" and translate them into C files within the same
# directory. The resultant binaries (with .pyd extensions in Windows) will 
# be in "../functions/cython/bin". Note, the files will be compiled in the
# scratch directory "./build".


extensions = [
    Extension("im3_processing", 
        ["./im3_processing.pyx"]
    )
]

setup(
    name="XCT Cython Extensions",

    ext_modules = cythonize(
        extensions,
        compiler_directives={'language_level' : "3"}
    ),

    include_dirs=[numpy.get_include()]
)