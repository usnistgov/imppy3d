This directory contains .yml text files which are used to define an enviroment
within the Anaconda package manager. The user can install all of the necessary
dependencies to run IMPPY3D from these .yml files. IMPPY3D has been tested on
Windows 10/11 x64 and Ubuntu 20.04/22.04 x64 using Python 3.9 and Python 3.10.
If a different operating system is used (i.e., macOS), then the pre-compiled
Cython extensions will not work. In which case, the Cython extensions must be
compiled for your operating system and placed in the "./functions/cython/bin"
folder.

The preferred method to install IMPPY3D is using pip. However, during 
development of IMPPY3D, custom Mamba/Conda Python environments were used to
handle dependencies. The directions given here explain how to create one of 
these environments. Note, these instructions only install the required
dependencies. To use IMPPY3D, you will still require the precompiled Cython
extensions (saved as .pyd files) appropriate for your operating system. Ensure
that the .pyd files are present in the folder, "./src/imppy3d/"


---- Anaconda/Miniforge Documentation ----

The authors of IMPPY3D use Miniforge and Mamba to set up the appropriate Python
environment to develop IMPPY3D, but Anaconda and Conda will work as well.
Miniforge is an open-source initiative that closely mimics Anaconda. Similar
Anaconda commands can be used as well in Miniforge, but instead of using
the "conda" command, Miniforge uses the "mamba" command.

For simplicity moving forward, the Python environment that contains all of the
necessary dependencies will be referred to as the Anaconda environment. Links
are provided below for downloading Miniforge; the official Anaconda
documentation is also suitable for Miniforge.

https://github.com/conda-forge/miniforge
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


---- Installing the Necessary (Generic) Anaconda Packages ----

The specific Anaconda environment used for development of these scripts has
been copied into the text files within this folder. These files have .yml 
extensions. To create a new Anaconda environment based on these same package 
dependencies, which is recommended, type the following in an Anaconda prompt:

  conda env create -f env_file.yml

where "env_file.yml" is replaced with the actual name of the text file. 
After installing the environment, make sure to activate the environment 
before running any IMPPY3D examples or scripts. This is achieved by the
following:

  conda activate env_xct

In order to keep the Anaconda dependencies generic for both Windows/Linux
systems, an Anaconda installation environment file has been created called,
"anaconda_env_py310_generic.yml". This Anaconda environment file contains
a list of the dependencies (and their respective versions) required by
IMPPY3D. A new Anaconda environment, which will be called "env_xct", can
be installed with the following command:

  conda env create -f anaconda_env_py310_generic.yml

This will use the Conda package manager to resolve dependency issues.
However, Conda can be time consuming compared to PIP. 

An alternative installation method utilizes the PIP manager instead of Conda.
This can easily be tried using following generic-PIP YML file along with the
following steps:

  conda env create -f anaconda_env_py310_generic_pip.yml

Then, you must activate the new environment,

  conda activate env_xct
  conda update --all

At this point, the additional dependencies can be installed using PIP
via the included text file called "requirements_py310.txt". Use the following
PIP command to install them:

  pip install --user --prefer-binary -r requirements_py310.txt


---- Installing the Necessary (Exact) Anaconda Packages ----

As an alternative to generic environments, exact replicates of the Python
environments that were used to develop and test IMPPY3D can also be installed.
These environment files, saved as .yml files, are provided in the two
subdirectories: one for Linux systems called "Linux_x64_PreMade_Env", and one
for Windows systems called "Windows_x64_PreMade_Env". Be sure to install the 
most up-to-date .yml file that uses Python 3.10, which will be denoted in the
name of the file, such as "miniforge_xct_env_win_x64_py3p10.yml". The same 
installation instructions mentioned above can be used to install one of these
environments. For example,

  conda env create -f miniforge_xct_env_win_x64_py3p10.yml

