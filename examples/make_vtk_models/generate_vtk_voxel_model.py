"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 23, 2023

This script generates synthetic data that represents a binarized 
ellipsoid that has been rotated in 3D space. The data type is a 3D
array of (UINT8) unsigned, integers -- just like a standard 3D image
stack. The resultant image stack can be thought of a voxel model 
(i.e., 3D pixels). The result is visualized to the user and 
optionally saved to the hard drive.
"""

# Import external dependencies
import sys
import numpy as np
from skimage import measure as meas
from skimage import draw
from skimage.util import img_as_ubyte
import pyvista as pv

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../functions') 
import volume_image_processing as vol
import vtk_api as vapi


# -------- DRAW AN ELLIPSOID --------

# ********** USER INPUTS ********** 
# x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
# Define the semi-major and semi-minor (radii) axes via constants a, b, and c.
ellip_a = 15 # Aligned to Z-axis (Image Index)
ellip_b = 31 # Aligned to Y-axis (Row Index)
ellip_c = 63 # Aligned to X-axis (Column Index)
# ********************************* 

print(f"\nCreating an ellipsoid in a cell...")
img_ellip = draw.ellipsoid(ellip_a, ellip_b, ellip_c)
img_ellip = img_as_ubyte(img_ellip)


# -------- APPLY AN OPTIONAL 3D ROTATION --------

# ********** USER INPUTS ********** 
# Chose the direction that you want the longest axis (major semi-axis) of the
# ellipsoid to be aligned with. Make sure to write this desired direction 
# in image coordinates. Note, [img, row, col] <==> [z, y, x]
perform_rotation = True # Set to True to perform the rotation
desired_maj_ax = np.array([1.0, 1.0, 1.0]) # [img, row, col]
# ********************************* 

if perform_rotation:

    print(f"\nRotating the ellipsoid...")

    # Ensure that the input is a unit vector
    ax_mag = np.sqrt( np.power(desired_maj_ax[0], 2) + 
                np.power(desired_maj_ax[1], 2) +
                np.power(desired_maj_ax[2], 2) )
    desired_maj_ax = desired_maj_ax/ax_mag

    # To prevent any clipping, need to pad the boundaries
    abc_arr = np.array([ellip_a, ellip_b, ellip_c])
    abc_max = np.amax(abc_arr)
    abc_max_i = np.argmax(abc_arr)
    img_ellip = vol.pad_image_boundary(img_ellip, n_pad_in=abc_max, quiet_in=True)

    if abc_max_i == 0:
        # Implies it is currently aligned in the Z-direction.
        # Write this in image coordinates: [img, row, col] <==> [z, y, x]
        maj_ellip_ax = np.array([1, 0, 0])
    elif abc_max_i == 1:
        # Implies it is currently aligned in the Y-direction.
        # Write this in image coordinates: [img, row, col] <==> [z, y, x]
        maj_ellip_ax = np.array([0, 1, 0])
    else:
        # Implies it is currently aligned in the X-direction.
        # Write this in image coordinates: [img, row, col] <==> [z, y, x]
        maj_ellip_ax = np.array([0, 0, 1])

    # Calculate the axis of rotation
    rot_axis = np.cross(desired_maj_ax, maj_ellip_ax)
    rot_axis_mag = np.sqrt( np.power(rot_axis[0], 2) + 
                            np.power(rot_axis[1], 2) +
                            np.power(rot_axis[2], 2) )
    rot_axis = rot_axis/rot_axis_mag

    # Use the dot product to find the magnitude of the rotation angle (in radians)
    rot_angle = np.arccos(np.clip(np.dot(maj_ellip_ax, desired_maj_ax), -1.0, 1.0))

    # Perform the 3D rotation
    img_ellip = vol.img_rotate_3D(img_ellip, rot_axis, rot_angle, binarize=True)

# Clip the image sequence back down to the smallest global-axis-aligned box.
# There are more elegant ways to do this, but I already have codes that 
# will calculate the coordinates of boundaries, so might as well use them.
label_arr, num_feats = meas.label(img_ellip, return_num=True, connectivity=2)
feat_props = meas.regionprops(label_arr) 
img_ellip = feat_props[0].image
img_ellip = img_as_ubyte(img_ellip)

# Update sizes
ellip_n_imgs = img_ellip.shape[0] # Number of images
ellip_n_rows = img_ellip.shape[1] # Number of rows 
ellip_n_cols = img_ellip.shape[2] # Number of columns 


#  -------- MAKE VTK VOXEL MODEL OF THE BINARIZED 3D ARRAY --------

# vapi.make_vtk_unstruct_grid() depends on a Cython extension that was compiled
# for Windows 10 (64-bit) and Ubuntu 20.04 (64-bit). If you are using a 
# different environment, then you may need to use vapi.make_vtk_unstruct_grid_slow()
vtk_obj = vapi.make_vtk_unstruct_grid(img_ellip)


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


#  -------- OPTIONALLY SAVE THE VTK MODEL --------

# Can save this as a VTK file and open in ParaView
#vtk_obj.save("./ellipsoid_voxel_model.vtk")