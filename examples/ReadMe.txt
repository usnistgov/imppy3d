This is the "examples" directory, which contains example scripts that
showcase how to use imppy3d. 

These examples use mock-data located in the "./resources" directory. If
no image stacks exist in the "./resources" directory, then you will 
first need to run the script that creates this synthetic data. This is
done by running the script, "./resources/generate_sample_data.py".

Currently, example scripts are provided for the following topics:

* Autocorrelation-based routines for feature size and anisotropy estimates
(See "./autocorrelation/")

* Fitting a rotated bounding box to 3D data
(See "./bounding_box/")

* Characterizing the shape and size of pores (in metal-based AM)
(See "./calc_metrics_pores/")

* Extracting data within a spherical region-of-interest
(See "./extract_sphere/")

* Interactively selecting parameters to filter and segment images
(See "./interactive_filtering")

* Creating voxel and surface models of binarized image stacks
(See "./make_vtk_models")

* Rotating an image stack and then re-slicing it along the new axis
(See "./rotate_reslice_img_stack/")

* Segmenting 3D particles and characterizing their shape and size
(See "./segment_3d_particles/")
