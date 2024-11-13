"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 23, 2023

This script creates a fully dense, 3D array which represents an image
stack containing white pixels. Then, it calls a pre-compiled Cython
extension that extracts only pixels located within a spherical 
region-of-interest. The resultant model is converted to a voxel
model and displayed to the user.
"""

# Import external dependencies
import numpy as np
from skimage import measure as meas
import pyvista as pv

import imppy3d.import_export as imex
import imppy3d.volume_image_processing as vol
import imppy3d.vtk_api as vapi


# -------- CREATE A FULLY DENSE IMAGE SEQUENCE --------

num_imgs = 25 # Number of images
num_rows = 25 # Number of rows of pixels in each image
num_cols = 25 # Number of columns of pixels in each image

img_arr = np.ones((num_imgs,num_rows,num_cols), dtype=np.uint8)*255


# -------- EXTRACT CENTRAL SPHERE --------
img_sphere = vol.extract_sphere(img_arr)


# -------- MAKE A VTK SURFACE MODEL TO SURFACE --------
verts, faces, normals, vals = vapi.convert_voxels_to_surface(img_sphere, 
	iso_level=120, g_sigdev=0.8)
vtk_surf_obj = vapi.make_vtk_surf_mesh(verts, faces, vals, smth_iter=25)


# -------- MAKE A VTK VOXEL MODEL --------
vtk_voxel_obj = vapi.make_vtk_unstruct_grid(img_sphere)


# -------- WRITE SOME OUTPUTS TO THE COMMAND WINDOW --------
# Calculate the minimum length of the image shape
label_arr, num_feats = meas.label(img_sphere, return_num=True, connectivity=1)
feat_props = meas.regionprops(label_arr)
sphere_props = feat_props[0]

eqv_diam = sphere_props.equivalent_diameter_area
axis_major = sphere_props.axis_major_length
sph_rad = axis_major/2.0

sph_bbox = sphere_props.bbox # (img_min, row_min, col_min, img_max, row_max, col_max)
bbox_L = sph_bbox[3] - sph_bbox[0]
bbox_W = sph_bbox[4] - sph_bbox[1]
bbox_T = sph_bbox[5] - sph_bbox[2] 

print(f"\nIdeal spherical radius: {sph_rad} units")
print(f"Volume of sphere with radius {sph_rad:.2f}: {(4.0/3.0)*np.pi*(sph_rad)**(3)} units^3")
print(f"Volume of the approximate voxel model: {vtk_voxel_obj.volume} units^3")
print(f"Length, width, and thickness of the bounding box for the voxel model: ({bbox_L}, {bbox_W}, {bbox_T})")
print(f"Volume of the surface model from marching cubes (no smoothing): {vtk_surf_obj.volume} units^3")
print(f"Percent difference between voxel and surface volumes: {100.0*np.absolute(vtk_voxel_obj.volume - vtk_surf_obj.volume)/vtk_voxel_obj.volume} %")

#imex.save_image_seq(img_sphere, "./", "img_stack_binary_sphere_small.tif", index_start_in=1)
#pv.save_meshio("./surf_smooth_sphere_small.stl", vtk_surf_obj, binary=True)


#  -------- VISUALIZE VTK MODEL --------
# For large models, may want to set show_edges=False since the black lines can
# be so close to each that the whole model ends up being black. With a black
# background, this can cause the illusion that the model is hidden.
plot1_obj = pv.Plotter(shape=(1, 2))

plot1_obj.subplot(0, 0)
plot1_obj.add_mesh(vtk_voxel_obj, scalars="values", show_edges=False, cmap="bone_r")
plot1_obj.remove_scalar_bar()
plot1_obj.show_axes()
plot1_obj.set_background(color=[0.0/255.0, 0.0/255.0, 0.0/255.0])

plot1_obj.subplot(0, 1)
plot1_obj.add_mesh(vtk_surf_obj, scalars="values", show_edges=False, cmap=['white'])
plot1_obj.remove_scalar_bar()
plot1_obj.show_axes()
plot1_obj.set_background(color=[0.0/255.0, 0.0/255.0, 0.0/255.0])

plot1_obj.link_views()
plot1_obj.view_isometric()

print("\nWARNING: Press 'q' to safely close this viewer.\n"\
	"DO NOT CLICK THE QUIT/CLOSE BUTTON IN THE CORNER OF THE WINDOW!")
plot1_obj.show(auto_close=False)

# This code will only work if the render window is still open. So, "auto_close=False"
# has been set in the plot1_obj.show() command. Additionally, the user must press the
# 'q' key to safely close the plotting window when it appears. 
orbit_path = plot1_obj.generate_orbital_path(n_points=72, shift=vtk_voxel_obj.length)

# Making an mp4 file is dependent on the imageio-ffmpeg package. To install it, use,
# conda install -c conda-forge imageio-ffmpeg 
# After installing, uncomment below

#plot1_obj.open_movie('./orbit_movie.mp4', framerate=9, quality=8)
#plot1_obj.orbit_on_path(orbit_path, write_frames=True)

plot1_obj.close()
