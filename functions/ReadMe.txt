This is the "functions" directory which contains the definitions of the 
functions which comprise the imppy3d library.

In addition to Python definitions, imppy3d also has custom C-extensions 
based on the Cython library. These are located in the "./cython" folder
in a file called "im3_processing.pyx". This C-extension has been compiled
for both Windows x64 and Linux x64 systems. The compiled executables can be
found in "./cython/bin". These executables are automatically imported by
the necessary functions defined in this folder.

In general, function definitions are categorized as follows:

* Bounding box routines as well as the BBox class are defined in 
"bounding_box.py"

* OpenCV routines, interactive scripts, and driver-functions are defined in
"cv_driver_functions.py", "cv_interactive_processing.py", and 
"cv_processing_wrappers.py".

* Scikit-Image routines, interactive scripts, and driver-functions are defined
in "ski_driver_functions.py", "ski_interactive_processing.py", 
"ski_processing_wrappers.py".

* Simple plotting wrappers for Matplotlib are defined in "plt_wrappers.py".

* Loading and saving image files are defined in "import_export.py".

* Functions aimed at general processing of 3D images (i.e., image stacks) are
defined in "volume_image_processing.py".

* Creating voxel models, surface meshes, and providing wrapper functions for
the VTK library are all defined in "vtk_api.py"