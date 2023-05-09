"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 23, 2023

This script imports a synthetic, binarized image stack. The resultant
image stack can be thought of a voxel model (i.e., 3D pixels). The 
boundary of the object, represented by white voxels, is converted
to a surface using the Marching Cubes algorithm. The result is
visualized to the user and optionally saved to the hard drive.
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


# -------- LOCAL FUNCTION DEFINITIONS --------

def mcubes_func(img_arr_in, iso_level, scale_spacing_in=1.0, is_binary_in=True,
    g_sigdev_in=0.8, pad_boundary_in=True, smth_iter_in=5):

    iso_level_in = iso_level

    voxel_vol = np.count_nonzero(img_arr_in)
    voxel_vol = voxel_vol*(scale_spacing_in**3) 

    verts, faces, normals, vals = vapi.convert_voxels_to_surface(img_arr_in, 
            iso_level=iso_level_in, scale_spacing=scale_spacing_in, 
            is_binary=is_binary_in, g_sigdev=g_sigdev_in, pad_boundary=pad_boundary_in)

    vtk_obj = vapi.make_vtk_surf_mesh(verts, faces, vals, smth_iter=smth_iter_in)
    vol_diff = np.absolute((vtk_obj.volume - voxel_vol)/voxel_vol)

    return vol_diff


def golden_sect_search(img_arr_in, x_L=5, x_U=250, scale_spacing_in=1.0, is_binary_in=True,
    g_sigdev_in=0.8, pad_boundary_in=True, smth_iter_in=5):

    tol = 3
    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

    x_L = x_L
    x_U = x_U

    imgs = img_arr_in
    iso_level_out = 125

    voxel_vol = np.count_nonzero(imgs)
    voxel_vol = voxel_vol*(scale_spacing_in**3) 

    (x_L, x_U) = (min(x_L, x_U), max(x_L, x_U))
    h = x_U - x_L
    if h <= tol:
        return (x_L + x_U)/2.0

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    x1 = x_L + invphi2 * h
    x2 = x_L + invphi * h

    f_x1 = mcubes_func(imgs, x1, scale_spacing_in=scale_spacing_in, 
        is_binary_in=is_binary_in, g_sigdev_in=g_sigdev_in, 
        pad_boundary_in=pad_boundary_in, smth_iter_in=smth_iter_in)

    f_x2 = mcubes_func(imgs, x2, scale_spacing_in=scale_spacing_in,
        is_binary_in=is_binary_in, g_sigdev_in=g_sigdev_in, 
        pad_boundary_in=pad_boundary_in, smth_iter_in=smth_iter_in)

    iter_num = 0
    for k in range(n-1):

        if f_x1 < f_x2:  # Use f_x1 > f_x2 to find the maximum
            x_U = x2
            x2 = x1
            f_x2 = f_x1
            h = invphi * h
            x1 = x_L + invphi2 * h
            f_x1 = mcubes_func(imgs, x1, scale_spacing_in=scale_spacing_in,
                is_binary_in=is_binary_in, g_sigdev_in=g_sigdev_in,
                pad_boundary_in=pad_boundary_in, smth_iter_in=smth_iter_in)

        else:
            x_L = x1
            x1 = x2
            f_x1 = f_x2
            h = invphi * h
            x2 = x_L + invphi * h
            f_x2 = mcubes_func(imgs, x2, scale_spacing_in=scale_spacing_in,
                is_binary_in=is_binary_in, g_sigdev_in=g_sigdev_in,
                pad_boundary_in=pad_boundary_in, smth_iter_in=smth_iter_in)

        iter_num += 1

    if f_x1 < f_x2:
        return np.round((x_L + x2)/2.0)
    else:
        return np.round((x1 + x_U)/2.0)


# -------- IMPORT IMAGE SEQUENCE --------

# See documentation on load_image_seq() for function details
dir_in_path = "../resources/hexahedron/"
name_in_substr = "hexa_data"
imgs_keep = (0,256)
voxel_size = 1.0 # Assume a voxel edge length of unity for this example

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


#  -------- (OPTIONAL) GOLDEN SECTION OPTIMIZATION --------

# Perform a Golden section search to find the optimal isovalue which 
# minimizes the difference in volume between the voxel model and the
# surface mesh. This function makes iterative calls to the marching
# cubes algorithm and can be costly.
iso_val_min = golden_sect_search(imgs, scale_spacing_in=voxel_size, smth_iter_in=0)
iso_val_min = np.round(iso_val_min) # Round the intensity to an integer


#  -------- MAKE VTK SURFACE MODEL --------

# For some models, Laplacian smoothing can be very beneficial. 
# Try smth_iter=50 in that case
verts, faces, normals, vals = vapi.convert_voxels_to_surface(imgs, 
    scale_spacing=voxel_size)

vtk_obj = vapi.make_vtk_surf_mesh(verts, faces, vals, smth_iter=0)


#  -------- VISUALIZE VTK MODEL --------

# For large models, may want to set show_edges=False since the black lines can
# be so close to each that the whole model ends up being black. With a black
# background, this can cause the illusion that the model is hidden.
plot1_obj = pv.Plotter()
plot1_obj.add_mesh(vtk_obj, scalars="values", show_edges=True, cmap=['white'])
plot1_obj.remove_scalar_bar()
plot1_obj.show_axes()
plot1_obj.set_background(color=[0.0/255.0, 0.0/255.0, 0.0/255.0])

print(f"\nClose the plot window to continue...")
plot1_obj.show("Surface Mesh")


#  -------- SAVE VTK MODEL --------

# Can save this as a VTK file and open in ParaView
vtk_obj.save("./surface_model.vtk")

# To save as something else, like an STL, use the meshio save function. To use
# meshio with Python 3.7 or older, you will need to install in Anaconda this
# package: conda install -c anaconda importlib_metadata

#pv.save_meshio("./surface_model.stl", vtk_obj)