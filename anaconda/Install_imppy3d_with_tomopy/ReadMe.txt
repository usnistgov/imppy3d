---- Installing the TomoPy First in Anaconda ----

If you would like to use TomoPy for X-ray CT reconstructions or
post-processing to remove ring-artifacts from reconstructed X-ray CT
images, then the following guide will walk you through the installation
process. Note, TomoPy is not available on PyPi, so this library must
be installed using the Conda manager.

The first step is to create an environment using the provided YML file:

  conda env create -f anaconda_env_py39_generic_pip_tomopy.yml

This can take ten minutes to finish. Once complete, you will have
a new Anaconda environment called "env_xct" with Python, PIP, and
TomoPy installed.


---- Installing the Remaining Dependencies Manually ----

The remaining imppy3d dependencies will be installed manually using
PIP. First, you must activate the new environment,

  conda activate env_xct

The dependencies are included in a text file called "requirements.txt".
Use the following PIP command to install them:

  pip install --user -r requirements.txt

Note, without the "--user" flag, this installation will fail. As of
now, it is unclear as to why. 