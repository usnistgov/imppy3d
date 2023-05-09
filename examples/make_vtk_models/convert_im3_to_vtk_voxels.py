"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 23, 2023

This script imports a synthetic, binarized image stack. The resultant
image stack can be thought of a voxel model (i.e., 3D pixels). The 
result is visualized to the user and optionally saved to the hard 
drive.
"""

# Import external dependencies
import sys
import numpy as np
from skimage.util import img_as_ubyte
import pyvista as pv

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../functions') 
import import_export as imex
import volume_image_processing as vol
import vtk_api as vapi


# -------- IMPORT IMAGE SEQUENCE --------

# See documentation on load_image_seq() for function details
dir_in_path = "../resources/hexahedron/"
name_in_substr = "hexa_data"
imgs_keep = (0,256)

imgs, imgs_names = imex.load_image_seq(dir_in_path, 
    file_name_in=name_in_substr, indices_in=imgs_keep)

if imgs is None:
    print(f"\nFailed to import images from the directory: \n{dir_in_path}")
    print("\nDouble-check that the example images exist, and if not, run")
    print("the Python script that creates all of the sample data in:")
    print(f"../resources/generate_sample_data.py")
    print("\nQuitting the script...")

    quit()

# Optionally write a log file containing the file names of imported images
#log_file_path = dir_in_path + "log_imported_imgs.txt"
#print(f"\nWriting import log file to: {log_file_path}")
#with open(log_file_path, 'w') as file_obj:
#    file_obj.write("The following image files were imported in this order:"\
#                    "\n\n")
#    for cur_name in imgs_names:
#        file_obj.write(f"{cur_name}\n")

# Extract the image pixel properties from the 3D Numpy array
num_imgs = imgs.shape[0] # Number of images
num_rows = imgs.shape[1] # Number of rows of pixels in each image
num_cols = imgs.shape[2] # Number of columns of pixels in each image


#  -------- IMAGE SEGMENTATION/BINARIZATION --------

# The "hexahedron_image_sequence" resource is already binarized


#  -------- MAKE VTK VOXEL MODEL --------

# vapi.make_vtk_unstruct_grid() depends on a Cython extension that was compiled
# for Windows 10 (64-bit) and Ubuntu 20.04 (64-bit). If you are using a 
# different environment, then you may need to use vapi.make_vtk_unstruct_grid_slow()
vtk_obj = vapi.make_vtk_unstruct_grid(imgs)


#  -------- VISUALIZE VTK MODEL --------

# For large models, may want to set show_edges=False since the black lines can
# be so close to each that the whole model ends up being black. With a black
# background, this can cause the illusion that the model is hidden.
plot1_obj = pv.Plotter()
plot1_obj.add_mesh(vtk_obj, scalars="values", show_edges=True, cmap="bone_r")
plot1_obj.remove_scalar_bar()
plot1_obj.show_axes()
plot1_obj.set_background(color=[0.0/255.0, 0.0/255.0, 0.0/255.0])

print(f"\nClose the plot window or press 'q' to continue...")
plot1_obj.show("Voxel Model")


#  -------- SAVE VTK MODEL --------

# Can save this as a VTK file and open in ParaView
vtk_obj.save("./voxel_model.vtk")