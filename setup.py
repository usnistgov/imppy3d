from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "imppy3d.im3_processing",
        ["src/imppy3d/cython/im3_processing.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='imppy3d',
    version='0.1.0',
    author='TODO',  # FIXME
    author_email='TODO',  # FIXME
    description='IMPPY3D: A library for processing 3D image stacks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/usnistgov/imppy3d/',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-image',
        'opencv-python',
        'vtk',
        'meshio',
        'imageio',
        'imageio-ffmpeg',
        'tifffile',
        'imagecodecs',
    ],
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level' : "3"}
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.6',
)