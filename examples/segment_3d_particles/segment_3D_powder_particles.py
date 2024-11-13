"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
March 6, 2023

This script imports synthetic data representing powder particles.
The synthetic image stack is filtered and then segmented. Clipped
particles are removed from the analysis. The binarized image 
stack is then written to the hard drive. Moreover, a voxel model
containing all of the particles is created, which can be opened
using ParaView. Next, the particles are analyzed and characterized
in terms of shape and size. All of these properties are written
out to a .csv file. Additionally, the analyzed particles are
converted to surface meshes and saved as a .vtk file which can
also be opened in ParaView and compared to the voxel model.
"""

# Import external dependencies
import csv
import numpy as np
from skimage import measure as meas
from skimage import morphology as morph
from skimage import restoration as rest
from skimage import segmentation as seg
from skimage.util import img_as_float, img_as_ubyte

# Import local modules
import imppy3d.import_export as imex
import imppy3d.cv_processing_wrappers as cwrap
import imppy3d.volume_image_processing as vol
import imppy3d.bounding_box as box
import imppy3d.vtk_api as vapi


# -------- IMPORT IMAGE SEQUENCE --------

# See documentation on load_image_seq() for function details
dir_in_path = "../resources/powder_particles/"
name_in_substr = "powder"
imgs_keep = (256,) # Import all images

# WARNING: You may want to reverse the image stack, as is shown here, if
# image 1 corresponds to the bottom of the part. This is done so that a
# right-handed coordinate system can be made using the intrinsic image
# indices: ascending column indices --> X, ascending row indices --> Y,
# and descending image indices --> Z. If you do not want to reverse the
# image stack, set flipz=False.

rev_img_stack = True

imgs, imgs_names = imex.load_image_seq(dir_in_path, 
    file_name_in=name_in_substr, indices_in=imgs_keep, flipz=rev_img_stack)

if imgs is None:
    print(f"\nFailed to import images from the directory: \n{dir_in_path}")
    print("\nDouble-check that the example images exist, and if not, run")
    print("the Python script that creates all of the sample data in:")
    print(f"../resources/generate_sample_data.py")
    print("\nQuitting the script...")

    quit()

if rev_img_stack:
    print(f"\nNote, the image stack has been reversed!")

# Optionally write a log file containing the file names of imported images
#log_file_path = dir_in_path + "log_imported_imgs.txt"
#print(f"\nWriting import log file to: {log_file_path}")
#with open(log_file_path, 'w') as file_obj:
#    file_obj.write("The following image files were imported in this order:"\
#                    "\n\n")
#    for cur_name in imgs_names:
#        file_obj.write(f"{cur_name}\n")

# Extract the image size properties from the 3D Numpy array
num_imgs = imgs.shape[0] # Number of images
num_rows = imgs.shape[1] # Number of rows of pixels in each image
num_cols = imgs.shape[2] # Number of columns of pixels in each image


# -------- (OPTIONAL) CROPPING --------

# Crop images here if desired. See "../interactive_filtering" examples
# for how to crop a batch of images.


# -------- NORMALIZE HISTOGRAM --------

# Linearly normalize the intensity histogram to range from 0 to 255
# for the whole image sequence at once.
imgs = imgs.reshape((num_imgs*num_rows, num_cols))
imgs = cwrap.normalize_histogram(imgs, quiet_in=True)
imgs = imgs.reshape((num_imgs, num_rows, num_cols))


# -------- REDUCE IMAGE NOISE AND THRESHOLD --------

# The nonlocal-means denoising filter does a great job at reducing
# noise, but a bilateral Gaussian blur is quicker and does a fine
# enough job in this case.
imgs_64bit = img_as_float(imgs)

print(f"\nDenoising and segmenting images...")
# During this on a 2D basis, but denoising can be done in 3D
for m, cur_img in enumerate(imgs_64bit):

    w_size = 12
    sig_col = 0.1
    sig_spt = 16

    img_temp = rest.denoise_bilateral(cur_img, win_size=w_size, 
        sigma_color=sig_col, sigma_spatial=sig_spt)

    img_temp = img_as_ubyte(img_temp)

    # Now apply a threshold here. Could have also done this
    # to all of the images at once, but already had a for-
    # loop here for denoising.
    thresh = 100
    img_mask = img_temp > thresh
    img_temp = img_as_ubyte(img_mask)

    imgs[m] = img_temp.copy()

    if (((m+1) % 8) == 0):
        print(f"    Currently processed {m+1}/{num_imgs}...")

print(f"\nSuccessfully processed {num_imgs} images!")


# -------- (OPTIONAL) SAVE IMAGE SEQUENCE --------

# Uncomment below to save the processed images
#save_path_out = "./binary_imgs/"
#file_name_out = "powder_binary.tif"
#imex.save_image_seq(imgs, save_path_out, file_name_out)


# -------- IMPORT REFERENCE IMAGE --------

# This is a unique step associated with characterizing powder 
# particles. The idea is that particles which have been clipped by
# the cylindrical boundary (or field-of-view) of the X-ray CT
# measurements should not be characterized in terms of shape or size.
# So, a reference image is created from one of the grayscale images
# which will be used to create a mask that identifies any particles
# on the boundary which have been clipped.
ref_img_path = "../resources/powder_particles/powder_0127.tif"
img_ref, _props = imex.load_image(ref_img_path, quiet_in=True)


# -------- DELETE CLIPPED PARTICLES --------

print(f"\nDeleting clipped particles...")

# Remove clipped particles along the circular boundary
imgs, n_del_particles = vol.del_edge_particles_3D(imgs, img_ref)
print(f"  Number of circular-edge particles erased: {n_del_particles}")

erased_particles_count = 0

# Next, erase any particles on the top & bottom flat interfaces that 
# have been clipped.
for m in [0, num_imgs-1]: # First and last image in the subvolume
    cur_img = imgs[m]

    # Find any particles on this interface image
    label_arr = meas.label(cur_img, connectivity=2)
    feat_props = meas.regionprops(label_arr)

    # Get a coordinate within each particle to use for successive flood fills
    for cur_feat in feat_props:
        # (row, col) with shape as an (N,2) array
        feat_coords = cur_feat.coords 
        seed_2d = tuple(feat_coords[0])
        seed_3d = (m, seed_2d[0], seed_2d[1])

        # Perform a 3D flood fill at this 3D seed
        imgs = morph.flood_fill(imgs, seed_3d, 0, connectivity=2)
        erased_particles_count += 1

print(f"  Number of top/bottom interface particles erased: {erased_particles_count}")


# -------- REMOVE TINY PARTICLES --------

# It is usually a good idea to remove tiny particles since this may
# be noise or beyond the effective resolution of the X-ray CT machine

min_feat_vol = 3*3*3
print(f"\nRemoving particles with fewer than {min_feat_vol} voxels...")

img_arr_label = meas.label(imgs, connectivity=2)
imgs = morph.remove_small_objects(img_arr_label, min_size=min_feat_vol,
    connectivity=2)

imgs[np.nonzero(imgs)] = 255
imgs = img_as_ubyte(imgs)


# -------- (OPTIONAL) SAVE IMAGE SEQUENCE --------

# Uncomment below to save the processed images
save_path_out = "./binary_imgs/"
file_name_out = "powder_binary_final.tif"
imex.save_image_seq(imgs, save_path_out, file_name_out)


# -------- (OPTIONAL) CREATE A VOXEL MODEL AND SAVE IT --------

# If the image stack is not too large and sufficient RAM is available,
# we can create a voxel model for all of the particles. If memory is
# an issue, then the image stack could be downscaled first. However,
# for this example, this should not be a problem at all for typical
# modern computer. These models can be opened in ParaView

# Pretend that each pixel in the image stack corresponds to 2 um
voxel_size = 2.0 # [um/pixel]

# Come constants we may want for later
voxel_size2 = voxel_size**2
voxel_size3 = voxel_size**3

print(f"\nConstructing the VTK voxel model of all particles...")
vtk_obj = vapi.make_vtk_unstruct_grid(imgs, scale_spacing=voxel_size)

file_name_out = "./particles_all_voxel.vtk"
print(f"\nSaving VTK voxel model to: {file_name_out}")
vtk_obj.save(file_name_out)


# -------- CHARACTERIZE THE SHAPE, ORIENTATION, AND SIZE OF PARTICLES --------

# This is a helper function for calling the bounding box routines,
# and it does some array-accounting for interpretting the results.
def bbox_search(imgs_in, search_flag_in):

    # ---------- FUNCTION INPUTS ----------
    #
    # imgs_in: A Numpy image stack (i.e., 3D array) of data type
    # np.uint8. The images should be binarized such that there are
    # only black pixels (with zero intensity) and white pixels (with
    # 255 intensity). This function will fit a rotated bounding box
    # around all of the white pixels.
    #
    # search_flag_in: An integer that defines what type of algorithm
    # to use in calculating the rotated bounded box. Can be either 0,
    # 1, 2, 3, or 4. See below.
    # 
    # NOTE: In general, the rotated, minimum-volume bounding box of a
    # particle does not do a good job at capturing the general
    # orientation of the particle itself. So, the most accurate
    # algorithms, such as search_flag_in = 0, will not necessarily give
    # the desired results. Our recommendation is to use 
    # search_flag_in = 3 or 4.
    # 
    #  0: Use an exhaustive search. Although not necessarily the
    #  fastest, this method should find the best bounding box within 1
    #  degree of any arbitrary rotation. This method is the default
    #  search algorithm.
    #
    #  1: Perform a singular-value decomposition calculation on the data
    #  set and use the eigenvector corresponding to the minimum
    #  variation (i.e., smallest eigenvalue) as one of the directions
    #  of the bounding box. The other two directions are found by
    #  projecting the points onto a plane normal to this minimum
    #  principal direction, and then solving the 2D bounding box
    #  problem using an exhaustive search. In practice, this method
    #  usually is slightly more accurate than using the maximum
    #  variation (see search=2 below) while being just as
    #  computationally efficient. Performing the convex hull usually
    #  improves the accuracy for this search.
    #
    #  2: Perform a singular-value decomposition calculation on the data
    #  set and use the eigenvector corresponding to the maximum
    #  variation (i.e., largest eigenvalue) as one of the directions of
    #  the bounding box. The other two directions are found by
    #  projecting the points onto a plane normal to this maximum
    #  principal direction, and then solving the 2D bounding box
    #  problem using an exhaustive search. Performing the convex hull
    #  usually improves the accuracy for this search algorithm.
    #
    #  3: Perform a singular-value decomposition calculation on the data
    #  set and use all three eigenvectors as directions of the bounding
    #  box. Although this is the fastest method for large data sets,
    #  this is also the least reliable method in terms of accuracy.
    #  Performing the convex hull usually improves the accuracy for
    #  this search algorithm.
    #
    #  4: Find the longest distance between two points and define this
    #  direction to be major z-axis of the bounding box. The other two
    #  directions are found by projecting the points onto a plane
    #  normal to this maximum principal direction, and then solving the
    #  2D bounding box problem using an exhaustive search.


    # Find the outermost white pixels along the feature boundary
    img_bound = seg.find_boundaries(imgs_in, connectivity=1,
        mode='inner', background=0)

    img_bound = img_as_ubyte(img_bound) # Convert from bool to uint8

    # Create a 2D list of the indices to the boundary pixels.
    # [[row_1, col_1], [row_2, col_2], ..., [row_n, col_n]]
    bound_coords = np.argwhere(img_bound)
    bound_coords_xyz = vol.intrinsic_to_spatial(bound_coords)

    bbox_fit = box.min_bounding_box(bound_coords_xyz, search=search_flag_in)

    b_x_props = np.hstack((np.array(bbox_fit.x_len), bbox_fit.calc_x_vec()))
    b_y_props = np.hstack((np.array(bbox_fit.y_len), bbox_fit.calc_y_vec()))
    b_z_props = np.hstack((np.array(bbox_fit.z_len), bbox_fit.calc_z_vec()))

    b_irc_props = np.row_stack((b_x_props, b_y_props, b_z_props))
    b_irc_props[:,1:4] = vol.spatial_to_intrinsic(b_irc_props[:,1:4])

    # Sort array in ascending order based on the first column
    ind = np.argsort(b_irc_props[:,0])
    b_irc_props = b_irc_props[ind]

    bbox_center = vol.spatial_to_intrinsic(bbox_fit.center)


    # ---------- FUNCTION RETURNS ----------
    #
    # bbox_fit: Returns the bounding box object (BBox class)
    #
    # b_irc_props: A Numpy array that contains the lengths of the bounding 
    # box, and the vector-directions of each of these lengths. They have been 
    # returned in image coordinates as such:
    # [[max_length, img_dir, row_dir, col_dir],
    #  [middle_length, img_dir, row_dir, col_dir],
    #  [min_length, img_dir, row_dir, col_dir]]
    #
    # bbox_center: The center of the bounding box given in image
    # coordinates (as a Numpy array) as such:
    # [img_position, row_position, col_position]

    return (bbox_fit, b_irc_props, bbox_center)


# A big Python list that will store all of the properties
# that we calculate for each particle
part_props = []
vtk_part_list = []

# Label the particles in 3D so we can loop through each particle
label_arr = meas.label(imgs, connectivity=1)
feat_props = meas.regionprops(label_arr)
num_feats = len(feat_props) # Number of particles

# Define the minimum particle size (in voxels) that will be allowable
# to characterize the shape of the particle. This is different than the
# X-ray CT resolution. For example, you can imagine that a perfectly
# spherical particle would be poorly represented with just 27 voxels 
# (i.e., 3-by-3-by-3 voxels); we might think it a cube.
min_feat_size = 6**3 

print(f"\nCharacterizing shape and orientation of particles...")

# Cycle through the particle-features and extract relevant properties
for particle_index, cur_feat in enumerate(feat_props):

    # Unique particle identifier that starts at one instead of zero
    particle_id = particle_index + 1

    # Grab an sub-stack of images of just the current particle feature 
    cur_img = img_as_ubyte(cur_feat.image)

    # Current particle subvolume with potential voids filled in
    cur_img_fill = img_as_ubyte(cur_feat.filled_image)

    # Centroid of the particle rounded to the nearest pixel in image coordinates
    cent = cur_feat.centroid # (img_index, row_index, column_index)
    cent = (np.round(cent[0]).astype(np.int32), 
        np.round(cent[1]).astype(np.int32), 
        np.round(cent[2]).astype(np.int32))

    # Number of pixels in the as-received particle and filled particle
    num_solid_pixels = np.count_nonzero(cur_img)
    num_filled_pixels = np.count_nonzero(cur_img_fill)

    # Convert pixels to a physical volume
    solid_vol = num_solid_pixels*voxel_size3
    filled_vol = num_filled_pixels*voxel_size3

    # If the particle contained a void, we can also report the void volume
    void_vol = filled_vol - solid_vol

    # Equivalent spherical diameter in [um]
    eqv_diam = 2.0*np.cbrt(0.23873241463*filled_vol)

    # Translation vector in (img, row, col) index coordinates, as well as
    # (X, Y, Z) scaled coordinates. These will be used later.
    trans_vec_irc = np.array([cur_feat.bbox[0], cur_feat.bbox[1], cur_feat.bbox[2]])
    trans_vec_xyz = voxel_size*(vol.intrinsic_to_spatial(trans_vec_irc))

    # If the particle is larger than the minimum feature size, then
    # also begin characterizing its topology and orientation.
    if num_filled_pixels >= min_feat_size:

        # Surface area calculations are very poor when using a voxel
        # model. More accurate measurements can be estimated if we first
        # convert the particle to a surface mesh. Can use a Golden
        # Section search optimization here if desired. If so, see the
        # examples in "../make_vtk_models/"
        verts, faces, normals, vals = vapi.convert_voxels_to_surface(cur_img_fill,
                iso_level=125, scale_spacing=voxel_size)

        # Convert the vertices and faces from marching cubes into a standard VTK
        # data structure.
        surf_part = vapi.make_vtk_surf_mesh(verts, faces, vals, smth_iter=5)

        surf_area_part = surf_part.area  # (um^2) surface area
        surf_vol_part = surf_part.volume # (um^3) volume

        # Calculate sphericity (from 0 to 1). Ensure consistent units here!
        sphericity = 1.46459188756*np.cbrt((6.0*surf_vol_part)**2) / surf_area_part

        # Calculate the bounding box of this sub-volume of images. All
        # results will be in relative coordinates to this sub-volume.
        bbox_obj, lwt_arr, b_cent = bbox_search(cur_img, 4)

        # L >= W >= T lengths of the rotated bounding box
        b_l_len = voxel_size*lwt_arr[2,0] # [um]
        b_w_len = voxel_size*lwt_arr[1,0] # [um]
        b_t_len = voxel_size*lwt_arr[0,0] # [um]

        # The unit vectors for L, W, and T in (img, row, col) index coordinates
        b_l_vec = np.array([lwt_arr[2,1], lwt_arr[2,2], lwt_arr[2,3]])
        b_w_vec = np.array([lwt_arr[1,1], lwt_arr[1,2], lwt_arr[1,3]])
        b_t_vec = np.array([lwt_arr[0,1], lwt_arr[0,2], lwt_arr[0,3]])

        # Calculate aspect ratios of the lengths
        b_l_over_t = b_l_len/b_t_len
        b_l_over_w = b_l_len/b_w_len
        b_w_over_t = b_w_len/b_t_len

        # Translate the sub-volume VTK model back into the global coordinates.
        surf_part.translate(tuple(trans_vec_xyz), inplace=True)

        # Next, translate the bounding box center to global coordinates.
        b_cent = b_cent + trans_vec_irc 
        b_cent = b_cent*voxel_size # [um]

        # Assign a new field value to the VTK model of the particle. This
        # will make it possible to search for specific particles by 
        # Particle # when viewing all the particles at once as a VTK model.
        surf_part.point_data['Particle ID'] = particle_id * \
            np.ones(surf_part.n_points, dtype=np.int32)

        # Save each particle surface mesh into a list for later
        vtk_part_list.append(surf_part)

        # Can perform additional segmentation criterion here, or calculate
        # other shape characteristics as needed.

    else: # The particle was not a large enough to characterize shape
        # Values that were not calculated are placed with -1

        sphericity = -1
        b_cent = np.array([-1, -1, -1])
        b_l_len = -1
        b_w_len = -1
        b_t_len = -1
        b_l_over_t = -1
        b_l_over_w = -1
        b_w_over_t = -1
        b_l_vec = np.array([-1, -1, -1])
        b_w_vec = np.array([-1, -1, -1])
        b_t_vec = np.array([-1, -1, -1])

    # Now append all of the values that we wish to be saved
    # into one large Python list
    temp_list = [
        particle_id, #1
        voxel_size, #2
        solid_vol, #3
        void_vol, #4
        eqv_diam, #5
        sphericity, #6
        b_cent[0], #7
        b_cent[1], #8
        b_cent[2], #9
        b_l_len, #10
        b_w_len, #11
        b_t_len, #12
        b_l_over_w, #13
        b_l_over_t, #14
        b_w_over_t, #15
        b_l_vec[0], #16
        b_l_vec[1], #17 
        b_l_vec[2], #18
        b_w_vec[0], #19
        b_w_vec[1], #20
        b_w_vec[2], #21
        b_t_vec[0], #22
        b_t_vec[1], #23
        b_t_vec[2], #24
        ]

    part_props.append(temp_list)

    # Provide updates to the user
    if (particle_index+1)%8 == 0:
        print(f"  Processed {particle_index+1}/{num_feats} particles...")

print(f"  Processed {num_feats}/{num_feats} particles...")

# Next, we define some helper functions to assist in saving
# the particle properties into a CSV file.
def convert_2d_list_to_str(list_in):
    # INPUT
    # 2D python list as input containing values that can be
    # converted to strings.

    list_out = []
    for cur_row in list_in:
        cur_row_str = []

        for cur_val in cur_row:
            cur_row_str.append(str(cur_val))

        list_out.append(cur_row_str)

    # OUTPUT
    # A 2D python list with all values converted to strings
    return list_out


def save_particle_props(list_in, path_in, header_in):
    # INPUT
    #
    # list_in: A 2D python list containing strings and/or numbers.
    #
    # path_in: File path as a string
    #
    # header_in: If list_in is of size (m, n), then header_in
    # should be a 1D Python list of size (n) containing strings. 
    # This list will be written first (i.e., this is the header line)

    list_in_str = convert_2d_list_to_str(list_in)

    print(f"\nWriting particle properties to: {path_in}")

    with open(path_in, 'w', newline='') as file_obj:
        csv_writer = csv.writer(file_obj, delimiter=',')

        csv_writer.writerow(header_in)

        for cur_row in list_in_str:
             csv_writer.writerow(cur_row)


# Define the header line of text for the particle properties CSV file
# Another reminder that the image stack was reversed upon being 
# imported. Therefore, the image-index direction is reversed for 
# values like the center of the bounding box ('BBox Center Img Index')
# with respect to the original image stack.
part_props_header = [
    'Particle ID', #1
    'Voxel Edge Length (um)', #2
    'Solid Volume (um^3)', #3
    'Void Volume (um^3)', #4
    'Equiv. Spher. Diameter (um)', #5
    'Approx. Sphericity', #6
    'BBox Center Rev-Img Pos. [um]', #7
    'BBox Center Row Pos. [um]', #8
    'BBox Center Col Pos. [um]', #9
    'BBox (L)ength [um]', #10
    'BBox (W)idth [um]', #11
    'BBox (T)hickness [um]', #12
    'BBox L/W', #13
    'BBox L/T', #14
    'BBox W/T', #15
    'L_Vec. Rev-Img Index', #16
    'L_Vec. Row Index', #17
    'L_Vec. Col Index', #18
    'W_Vec. Rev-Img Index', #19
    'W_Vec. Row Index', #20
    'W_Vec. Col Index', #21
    'T_Vec. Rev-Img Index', #22
    'T_Vec. Row Index', #23
    'T_Vec. Col Index', #24
    ]

path_part_props = "./particle_shape_properties.csv"
print(f"\nSaving particle properties to {path_part_props}...")
save_particle_props(part_props, path_part_props, part_props_header)

# Construct the global particle model from individual VTK meshes
if vtk_part_list:

    print(f"\nMaking the VTK surface model of analyzed particles...")
    num_vtk = len(vtk_part_list)

    # If there is more than one particle, then we want to merge all of
    # the surface-based VTK models together
    if num_vtk > 1:

        merged_vtk = vtk_part_list[0]
        for vtk_ii in np.arange(1, num_vtk):
            cur_vtk = vtk_part_list[vtk_ii]
            merged_vtk.merge(cur_vtk, merge_points=False, inplace=True)
            
            if (vtk_ii+1)%8 == 0:
                print(f"  Processed {vtk_ii+1}/{num_vtk} VTK files...")

        print(f"  Processed {num_vtk}/{num_vtk} VTK files...")

    else:
        merged_vtk = vtk_part_list[0]

    # Finally, we save the merged model to the hard drive
    vtk_path_out_sph = "./particles_analyzed_surf.vtk"
    print(f"\nSaving VTK surface model to: {vtk_path_out_sph}")
    merged_vtk.save(vtk_path_out_sph)

    del vtk_part_list

print("\nScript finished successfully!")
