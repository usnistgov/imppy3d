This directory contains .yml text files which are used to define an enviroment
within the Anaconda package manager. The user can install all of the
necessary dependencies to run imppy3d from these .yml files. imppy3d has been
tested on Windows 10 x64 and Ubuntu 20.04 x64 using Python 3.9. However,
imppy3d is expected to work for other Linux x64 distros as well as Windows 11
x64, so long as Anaconda and Python 3.9 are used. If a different operating
system is used, then the pre-compiled Cython extensions may not work. In
which case, the Cython extensions would need to be compiled for your
operating system and placed in the "./functions/cython/bin" folder.


---- Anaconda Documentation ----

Links are provided below for downloading Anaconda as well as the official
Anaconda documentation:

https://www.anaconda.com/products/distribution
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


---- Installing the Necessary Anaconda Packages ----

The specific Anaconda environment used for development of these scripts has
been copied into the text files within this folder. These files have .yml 
extensions. To create a new Anaconda environment based on these same package 
dependencies, which is recommended, type the following in an Anaconda prompt:

  conda env create -f env_file.yml

where "env_file.yml" is replaced with the actual name of the text file. For
example, with a Windows 10 (x64) machine and an environment based on Python
3.9, the filename to use is "anaconda_env_py39_W10_x64.yml". Note, the name
of the new environment is given at the top of the "env_file.yml" file, and by
default, it is "env_xct".

Due to dependency challenges with the Conda-Forge branch of OpenCV in a Linux
environment, OpenCV had to be installed manually into the Anaconda
environment using PIP. Utilizing the Linux (x64) YML file will still
automatically install OpenCV, but the user should be aware that additional
installations of new packages using the Conda package manager may cause
issues with the OpenCV library, which is managed sepearately through PIP.

After installing the environment, make sure to activate the environment 
before running any imppy3d examples or scripts. This is achieved by the
following:

  conda activate env_xct


---- Manually Installing Anaconda Packages ----

If you are using imppy3d with a different operating system or Python version,
then you may have to install the dependencies manually. Moreover, the Cython
codes in this library may not work as intentioned unless they are recompiled
on the new system. As newer versions of these dependencies are developed,
there is the possibility that some functions in imppy3d will not work, which
is why the YML environment files are the recommended way to set up the
Anaconda environment. However, the main dependencies used by imppy3d are as
follows:

Numpy: "conda install -c anaconda numpy"
SciPy: "conda install -c anaconda scipy"
OpenCV: "conda install -c conda-forge opencv"
SciKit-Image: "conda install -c anaconda scikit-image"
Pyvista: "conda install -c conda-forge pyvista"
Meshio: "conda install -c conda-forge meshio"
MatPlotLib: "conda install -c conda-forge matplotlib"
Cython: "conda install -c anaconda cython"

However, it is best to install all of the packages at once while setting up a
new environment in order to resolve all of the dependencies. An example line,
though not recommended, to do that would be:

  conda create -n env_xct python=3.9 pip numpy scipy scikit-image opencv pyvista matplotlib meshio cython --channel anaconda --channel conda-forge

In the specific case of Linux, OpenCV had to be installed separately using PIP.
So, the few commands, again not recommended, to do that would be:

  conda create -n env_xct python=3.9 pip numpy scipy scikit-image pyvista matplotlib meshio cython --channel anaconda --channel conda-forge
  conda activate env_xct
  python -m pip install opencv-python==4.5.5.64

However, the above lines do not control the versions that get installed. So,
an even better solution is to try using the generic YML file to let the Conda
manager install these packages along with the specific versions required by
imppy3d:

  conda env create -f anaconda_env_py39_generic.yml

If this still fails to resolve, then try installing all of the packages using
the PIP manager instead of Conda. This can easily be tried using the generic-PIP
YML file:

  conda env create -f anaconda_env_py39_generic_pip.yml
