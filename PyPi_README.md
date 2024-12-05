# IMPPY3D

Image Processing in Python for 3D image stacks.

## Description
Image Processing in Python for 3D image stacks, or IMPPY3D, is a software
repository comprising mostly Python scripts that simplify post-processing and
3D shape characterization of grayscale image stacks, otherwise known as
volume-based images, 3D images, or voxel models. IMPPY3D was originally created
for post-processing image stacks generated from X-ray computed tomography
measurements. However, IMPPY3D also contains a functions to aid in
post-processing general 2D/3D images. 

Python was chosen for this library because of it is a productive, easy-to-use
language. However, for computationally intense calculations, compiled codes and
libraries are used for improved performance, such as well known libraries like
Numpy and SciKit-Image. Compiled libraries internal to IMPPY3D were created
using Cython. IMPPY3D was developed in an Anaconda environment with Windows 10
and Linux in mind, and suitable Anaconda environment files for these operating
systems are provided to simplify the process of installing the necessary
dependencies although some dependancy resolution may still be required. 

Some of the highlighted capabilies of IMPPY3D include: interactive graphical
user-interfaces (GUIs) available for many image processing functions, various
2D/3D image filters (e.g., blurring, sharpening, denoising, erosion/dilation),
the ability to segment and label continuous 3D objects, precisely rotating an
image stack in 3D and re-slicing along the new Z-axis, multiple algorithms
available for fitting rotated bounding boxes to continuous voxel objects, image
stacks can be converted into 3D voxel models suitable for viewing in ParaView,
and voxel models can be represented as smooth surface-based models like STL
meshes. Additional information and example scripts can be found in the included
ReadMe files.

More information and examples are discussed on the [IMPPY3D GitHub](https://github.com/usnistgov/imppy3d/) page.

## Installation 
The development of IMPPY3D uses the [Mamba](https://github.com/conda-forge/miniforge) 
package manager for handling dependencies, similar to the popular Conda package
manager. We recommend creating a new Python 3.10 environment (in Mamba/Conda)
prior to installing IMPPY3D. The easiest method of installing IMPPY3D is
through `pip`. Wheel files(.whl) for `pip` can be found either on PyPi or in
the IMPPY3D repository(for Windows or Linux machines). Alternatively, IMPPY3D
can be compiled using setuptools. The following subsections go into more
details for each of these installation cases.

### Installing From PyPi (Python 3.10)
The simplest method of installing IMPPY3D is through [PyPi](https://pypi.org/project/imppy3d/). 
Installing IMPPY3D from PyPi can be achieved using `pip` via the following,

`pip install imppy3d==1.1.11`

It is important that you explicitly specify the latest version of IMPPY3D, in
this case, version 1.1.11. Moreover, the pip installation process of IMPPY3D is
currently restricted to Python 3.10 environments.

### Installing Using Pip with Local Binary Files
The binary .whl files are located in the folder, "./dist/", of the
[GitHub repository](https://github.com/usnistgov/imppy3d/). The name
of the .whl files will contain information about the Python version, IMPPY3D
version number, and operating system. Currently, precompiled .whl files are
only available for Windows and Linux operating systems using Python 3.10. To
install IMPPY3D using one of these precompiled .whl files, choose the
appropriate .whl for your operating system and use the `pip` command in your
Python environment,

`pip install /path/to/file.whl`

### Compiling IMPPY3D
IMPPY3D is largely Python-based, but there are also C extensions, via Cython,
that must be compiled in order to use all of the features of IMPPY3D. A
`setup.py` file is provided in the [GitHub repository](https://github.com/usnistgov/imppy3d/) for easier compilation of IMPPY3D. To
compile IMPPY3D, open a terminal with your Python environment active and type,

`python setup.py bdist_wheel sdist` 

This command will use the setuptools library to compile IMPPY3D into a wheel
(.whl). Upon successful completion, the binary .whl file will be located in the
folder, "./dist/". Additionally, the above command will create source
distribution, which is an alternative installation file to the binary
distribution. With the new .whl file created, simply follow the instructions
above stated about installing IMPPY3D using `pip` and a local binary file.

It is important to ensure that your Python environment also has the appropriate
environment variables set for your C compiler. If setuptools cannot find your
compiler, then compilation of IMPPY3D will fail. Furthermore, the same
C-compiler as that which was used to compile the specific Python version you
are using should also be used to create the IMPPY3D C extensions. For Windows
users, see the documentation, 
[https://wiki.python.org/moin/WindowsCompilers](https://wiki.python.org/moin/WindowsCompilers).

## Verify Installation and Usage Examples 
If you would like verify your installation of IMPPY3D, there are unit test
scripts available for download in the "./tests/" folder of the
[GitHub repository](https://github.com/usnistgov/imppy3d/). To execute all of
the tests, run the script, "./tests/main_run_tests.py", which will print a text
summary of the results to the console.

Additionally, in the [GitHub repository](https://github.com/usnistgov/imppy3d/), 
there are a number of example Python scripts provided in the "./examples/"
folder. These example scripts serve as an excellent starting point when
developing new post-processing scripts. As we continue to use IMPPY3D in new
applications, we aim to continue to provide new example scripts in this
folder. 

## A Note About Coordinate Systems in IMPPY3D
It is well known that a stack of images can represent a 3D volume, but the
coordinate systems in such data files is not standardized. IMPPY3D regularly
switches between two coordinate systems: indicial coordinates and Cartesian
coordinates. In IMPPY3D, an image stack is stored as 3D Numpy array, and so,
the indices of this 3D array can directly serve as coordinates. However, it is
often useful to use Cartesian coordinates when applying 3D rotations or
visualizing the volume. When converting to Cartesian coordinates from indicial
coordinates, care must be taken so that the resultant X-, Y-, and Z-axes in the
Cartesian system define a right-handed coordinate system. 

IMPPY3D assumes the following conversion between coordinate systems applies:

* Ascending image indices corresponds to ascending Z-coordinates.

* Ascending row indices corresponds to ascending Y-coordinates.

* Ascending column indices corresponds to ascending X-coordinates.

For a given image, the 2D origin is taken to be the top-left corner of the
image. Therefore, the X-axis points to the right (across the columns), and the
Y-axis points downward (across the rows). However, the Z-axis in an image stack
must be carefully considered. In IMPPY3D, we assume that the first image in the
image stack corresponds to the TOP of the physical volume; every subsequent
image represents a slice of the volume beneath the first image. In doing so, a
right-handed coordinate system is naturally created based on the conversion of the indicial coordinates, as shown in the picture below.

![](https://github.com/usnistgov/imppy3d/blob/main/readme_pics/IMPPY3D_Coordinate_System.png?raw=True)

Tomographic reconstructions from X-ray radiographs often result in the first
image in the image stack representing the bottom of the physical object. In
this case, we recommend reversing the image stack prior to any post-processing
in IMPPY3D. If the image stack is not reversed, then the resultant 3D visuals
(as .vtk files) will be reflections of the physical object. Reversing the image
stack can be automatically performed for you by using the 'flipz' parameter
when importing an image stack using `imppy3d.import_export.load_image_seq()`. 

## Roadmap
* Convert the comment blocks in function definitions to a common standard for 
automatic generation of the documentation using Sphinx.

* Incoporate additional libraries like TomoPy for reconstruction of X-ray CT 
radiographs and removal of ring artifacts.

* Create an optimization routine that stitches multiple X-ray CT fields-of-view
together.

## Contributing
If you'd like to contribute to IMPPY3D, start by searching through the pull
requests to see whether someone else has raised a similar idea or question. If
you don't see your idea listed, and you think it fits into the goals of this
project, open a [pull request](https://github.com/usnistgov/imppy3d/pulls).

## Reporting Issues and Support
If you encounter any bugs or unintended behavior, please create an "Issue" in
the [IMPPY3D GitHub](https://github.com/usnistgov/imppy3d/issues) repository
and label it with the "bug" tag. Provide information about the bug, and if
possible, how to recreate the bug.  

You can also make a request for new features using the Issues mechanism. Please
label these requests with the "enhancement" tag. 

For questions on how best to use IMPPY3D for a specific application, feel free
to contact Dr. Newell Moser (see contact email below).  

## Authors and Acknowledgment

### Lead developer: 
* Dr. Newell Moser, NIST (newell.moser@nist.gov)

### Supporting developers: 
* Dr. Alexander K. Landauer, NIST

* Dr. Orion L. Kafka, NIST

### Acknowledgements:
* Dr. Edward J. Garboczi, NIST Fellow (Retired)

* The JOSS team and reviewers: [@gknapp1](https://github.com/gknapp1), [@sitic](https://github.com/sitic), and [@danielskatz](https://github.com/danielskatz).

## Citing This Library
If IMPPY3D has been significant in your research, and you would like to acknowledge
the project in your academic publication, we suggest citing the following NIST data
repository:

Moser, Newell H., Landauer, Alexander K., Kafka, Orion L. (2023), IMPPY3D: Image
processing in python for 3D image stacks, National Institute of Standards and
Technology, https://doi.org/10.18434/mds2-2806
