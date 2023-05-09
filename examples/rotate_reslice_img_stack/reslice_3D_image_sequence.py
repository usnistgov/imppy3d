"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 28, 2023

This script imports a 3D image sequence and applies a rotational and
translational transformations. These are also known as affine
transformations. Since a new Z-axis is created through the image
sequence after rotation, this procedure is also sometimes termed
re-slicing. 
"""

# Import external dependencies
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndim
from skimage import measure as meas
from skimage.util import img_as_bool, img_as_ubyte
import pyvista as pv

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../functions') 
import import_export as imex
import cv_processing_wrappers as wrap
import cv_driver_functions as drv
import volume_image_processing as vol
import vtk_api as vapi

# Set constants related to plotting
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)         # Controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # Fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # Fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # Fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # Fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # Legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # Fontsize of the figure title



# -------- IMPORT IMAGE SEQUENCE --------

# See documentation on load_image_seq() for function details
dir_in_path = "../resources/hexahedron/"
name_in_substr = "hexa_data"

# Setting this to some large number implies all images will be imported
imgs_keep = (10000,) 

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


#  -------- MAKE A COPY OF THE ORIGINAL GRAYSCALE IMAGES --------
# If these were actually grayscale images, should make a copy here of them
# here. The binarized images are necessary for the following procedures, but
# the actual rotation should be applied to the grayscale images.
imgs_orig = imgs.copy()


# -------- IMAGE SEGMENTATION/BINARIZATION --------
# The "hexahedron_image_sequence" resource is already binarized. Just going
# to create a new Numpy view here instead of a deep copy
imgs_bin = imgs


# -------- CALCULATE THE SURFACE NORMAL VECTOR OF A SPECIFIED SIDE --------
# Grab a surface patch on the top surface and calculate the normal vector.
# Could have picked a surface patch from any direction though.
# Remember, this function works with image coordinates (image_index, 
# row_index, column_index). To convert this to spatial coordinates,
# use the following conversions: image index --> Z-direction, row index -->
# Y-direction, column index --> X-direction.

# For this hexahedron, this is trivial: we know the unit vector will be
# aligned in the Z-direction. Just demonstrating here.
center_search = ("descend", 64, 64)
radial_dist = (3,3)
norm_dir3, pix_coords, cent_coord = vol.calc_surf_normal3(imgs_bin,
                                    center_search, radial_dist)
print(f"Normal vector (img_index, row_index, col_index): {tuple(norm_dir3)}")


# -------- (OPTIONAL) PLOT THE SURFACE PATCH OF PIXELS AND BEST-FIT PLANE --------
# Although this is optional, it is nice to see what points were selected
# Easier to work in (x,y,z) for plotting purposes
pix_coords_xyz = vol.intrinsic_to_spatial(pix_coords)
cent_xyz = np.flip(cent_coord)
norm_dir3_xyz = np.flip(norm_dir3)

# Points to sample in order to define the plane that will be plotted
XX = np.arange(np.amin(pix_coords_xyz[:,0])-1, np.amax(pix_coords_xyz[:,0])+1)
YY = np.arange(np.amin(pix_coords_xyz[:,1])-1, np.amax(pix_coords_xyz[:,1])+1)
XX, YY = np.meshgrid(XX, YY)

# A plane is defined: a*(x-x0)+b*(y-y0)+c*(z-z0)=0
# where [a,b,c] is the unit normal vector and [x0,y0,z0] is the centroid. 
# Use the x- and y-coordinates of the points that were used to fit the plane
# as inputs to calculate the corresponding z-coordinates of the fitted plane
a_n = norm_dir3_xyz[0]
b_n = norm_dir3_xyz[1]
c_n = norm_dir3_xyz[2]
x_0 = cent_xyz[0]
y_0 = cent_xyz[1]
z_0 = cent_xyz[2]

if np.absolute(c_n) < 1.0E-6:
    c_n = 1.0E-6

ZZ = (-a_n*(XX-x_0) - b_n*(YY-y_0))/c_n + z_0 # Z-coordinates of the plane

# Calculate the min-max ranges in each dimension to set the plot limits
x_range = np.amax(XX) - np.amin(XX)
y_range = np.amax(YY) - np.amin(YY)
z_range = np.amax(ZZ) - np.amin(ZZ)
xyz_range = np.array([x_range, y_range, z_range])
i_max = np.argmax(xyz_range)
lim_half = xyz_range[i_max]/2.0

# Make the 3D plot of the points that were found
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(pix_coords_xyz[:,0], pix_coords_xyz[:,1], pix_coords_xyz[:,2],
    c='r', marker='o')

ax1.plot_surface(XX, YY, ZZ, alpha=0.3)

ax1.quiver(cent_xyz[0], cent_xyz[1], cent_xyz[2], norm_dir3_xyz[0], 
    norm_dir3_xyz[1], norm_dir3_xyz[2], length=1.0, arrow_length_ratio=0.3,
    colors=(0,0,0,1))

ax1.set_xlim(cent_xyz[0] - lim_half, cent_xyz[0] + lim_half)
ax1.set_ylim(cent_xyz[1] - lim_half, cent_xyz[1] + lim_half)
ax1.set_zlim(cent_xyz[2] - lim_half, cent_xyz[2] + lim_half)
ax1.set_title('Surface Patch and Best-Fit Plane')
ax1.set_xlabel('X-Axis (Column Index)')
ax1.set_ylabel('Y-Axis (Row Index)')
ax1.set_zlabel('Z-Axis (Image Index)')
plt.show()


# -------- TRANSLATE THE GRAYSCALE IMAGE SEQUENCE --------
# The object in the image sequence may not be centered with respect to the
# current image stack. So, first calculate the centroid of the object from
# the segmented image sequence and translate the grayscale images accordingly
# This is important because the image stack will be rotated about its center,
# so the feature should be pre-aligned with this center pixel ahead of time.

# Label all of the white-pixel features
label_arr, num_feats = meas.label(imgs_bin, return_num=True, connectivity=1)
feat_props = meas.regionprops(label_arr)

# The primary object will be the feature with the largest number of pixels
max_area = 0
fi = 0
for indx in range(0,num_feats):
    cur_feat = feat_props[indx]
    if cur_feat.area > max_area:
        max_area = cur_feat.area
        fi = indx

# Calculated centroid of the object: (img_index, row_index, column_index)
feat_cent = feat_props[fi].centroid
feat_cent = (np.rint(feat_cent)).astype(np.int32)

# Calculate the center of the image sequence based on its shape
img_mid = (np.floor(num_imgs/2.0)).astype(np.int32)
row_mid = (np.floor(num_rows/2.0)).astype(np.int32) 
col_mid = (np.floor(num_cols/2.0)).astype(np.int32)
img_cent = np.array([img_mid, row_mid, col_mid])

# Calculate the translation vector: [delta_image, delta_row, delta_col]
trans_vec3 = img_cent - feat_cent

# Represent this translation in homogeneous coordinates
trans_mat = np.identity(4, dtype=np.int32)
trans_mat[0,3] = -trans_vec3[0]
trans_mat[1,3] = -trans_vec3[1]
trans_mat[2,3] = -trans_vec3[2]

print("\nCalculating translation transformation matrix...")
print(f"    Center of image sequence (img, row, col): {tuple(img_cent)}")
print(f"    Centroid of the largest feature (img, row, col):"\
    f" {tuple(feat_cent)}")

# Apply this as an affine transformation using linear interpolation.
# However, this can be slow and memory intensive. Another option is
# to just directly move the grayscale pixels using conventional Numpy
# slicing. No interpolation is used in that case. You may want to pad
# boundaries though to avoid clipping.
print("\nPerforming translation affine transformation...")
imgs_orig = ndim.affine_transform(imgs_orig, trans_mat, order=1)


# -------- PERFORM 3D ROTATION OF THE GRAYSCALE IMAGES --------
# We grabbed the surface normal at the top of the image sequence. More 
# specifically, in this special case, we grabbed a unit vector (in image
# coordinates) equal to [1, 0, 0]. Let us decide to align it to the [1, 1, 0]
# direction. Consequently, this will be a clockwise 45 deg rotation about the
# x-axis.
desired_vec3 = np.array([1.0, 1.0, 0.0])

# Ensure it is a unit vector
desired_vec3_mag = np.sqrt( np.power(desired_vec3[0], 2) + 
                        np.power(desired_vec3[1], 2) +
                        np.power(desired_vec3[2], 2) )
desired_vec3 = desired_vec3/desired_vec3_mag

# Use the cross product to find the desired rotation axis
rot_axis = np.cross(desired_vec3, norm_dir3)
rot_axis_mag = np.sqrt( np.power(rot_axis[0], 2) + 
                        np.power(rot_axis[1], 2) +
                        np.power(rot_axis[2], 2) )
rot_axis = rot_axis/rot_axis_mag

# Use the dot product to find the magnitude of the rotation angle (in radians)
rot_angle = np.arccos(np.clip(np.dot(norm_dir3, desired_vec3), -1.0, 1.0))

# Apply the affine transformation
print("\nPerforming rotation affine transformation...")
print(f"    Rotation Axis (img, row, col): {rot_axis}")
print(f"    Rotation Magnitude (CCW): {rot_angle*180.0/np.pi} (degrees)")

# Apply rotation. These are no longer the original images (confusing name now)
imgs_orig = vol.img_rotate_3D(imgs_orig, rot_axis, rot_angle, binarize=True)
print("\nRotation successfully applied!")


# -------- IMAGE SEGMENTATION/BINARIZATION --------
# This would be the place to re-binarize the now rotated grayscale images. In
# this case, the input image sequence was already binarized, so I set the
# binarize flag to True in order to get binarized images as output. In
# general, you should rotate the grayscale images, not the binarized images.


# -------- CREATE BEFORE-AND-AFTER VOXEL MODELS --------

print("\nCreating voxel plots...")
vtk_obj_before = vapi.make_vtk_unstruct_grid(imgs_bin)
vtk_obj_after = vapi.make_vtk_unstruct_grid(imgs_orig)

# Plot the voxel mesh
plot1_obj = pv.Plotter(shape=(1,2))
plot1_obj.subplot(0,0)
plot1_obj.add_mesh(vtk_obj_before, scalars="values", show_edges=True, 
	cmap="bone_r")
plot1_obj.remove_scalar_bar()
plot1_obj.show_axes()
plot1_obj.set_background(color=[0.0/255.0, 0.0/255.0, 0.0/255.0])

plot1_obj.subplot(0,1)
plot1_obj.add_mesh(vtk_obj_after, scalars="values", show_edges=True, 
	cmap="bone_r")
plot1_obj.remove_scalar_bar()
plot1_obj.show_axes()
plot1_obj.set_background(color=[0.0/255.0, 0.0/255.0, 0.0/255.0])

print(f"\nOriginal model on the left, and the rotated and resliced model on the right.")
print(f"Close the plot window to continue...")
plot1_obj.link_views()
plot1_obj.show("LEFT: Before    RIGHT: After")


# -------- SAVE VTK FILE OR EXPORT IMAGE SEQUENCE --------
# This would be the place to save the rotated image sequence either as
# an image sequence or as VTK file. See the other scripts for examples.