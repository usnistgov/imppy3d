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


---- Installing the Necessary (Generic) Anaconda Packages ----

The specific Anaconda environment used for development of these scripts has
been copied into the text files within this folder. These files have .yml 
extensions. To create a new Anaconda environment based on these same package 
dependencies, which is recommended, type the following in an Anaconda prompt:

  conda env create -f env_file.yml

where "env_file.yml" is replaced with the actual name of the text file. 
After installing the environment, make sure to activate the environment 
before running any imppy3d examples or scripts. This is achieved by the
following:

  conda activate env_xct

In order to keep the Anaconda dependencies generic for both Windows/Linux
systems, an Anaconda installation environment file has been created called,
"anaconda_env_py39_generic.yml". This Anaconda environment file contains
a list of the dependencies (and their respective versions) required by
imppy3d. A new Anaconda environment, which will be called "env_xct", can
be installed with the following command:

  conda env create -f anaconda_env_py39_generic.yml

This will use the Conda package manager to resolve dependency issues.
However, Conda can be time consuming compared to PIP. 

An alternative installation method utilizes the PIP manager instead of Conda.
This can easily be tried using following generic-PIP YML file along with the
following steps:

  conda env create -f anaconda_env_py39_generic_pip.yml

Then, you must activate the new environment,

  conda activate env_xct
  conda update --all

At this point, the additional dependencies can be installed using PIP
via the included text file called "requirements.txt". Use the following
PIP command to install them:

  pip install --user --prefer-binary -r requirements.txt


---- Tips and Troubleshooting ----

If you are having trouble installing the generic Anaconda packages, you can
try installing them using the Miniforge3 package manager instead of Anaconda.
Miniforge is an open-source initiative that closely mimics Anaconda, but
performs far more quickly. The same Anaconda commands can be used as well, 
but instead of using "conda", simply use, "mamba". More information can be
found here,

https://github.com/conda-forge/miniforge


---- Installing the Necessary (Exact) Anaconda Packages ----

As an alternative to generic environments, exact replicates of the Python
environments that were used to develop and test imppy3d can also be installed.
These environment files, saved as .yml files, are provided in the two
subdirectories: one for Linux systems called "Linux_x64_PreMade_Env", and one
for Windows systems called "Windows_x64_PreMade_Env". Be sure to install the 
most up-to-date .yml file that uses Python 3.10, which will be denoted in the
name of the file, such as "miniforge_xct_env_win_x64_py3p10.yml". The same 
installation instructions mentioned above can be used to install one of these
environments. For example,

  conda env create -f miniforge_xct_env_win_x64_py3p10.yml

