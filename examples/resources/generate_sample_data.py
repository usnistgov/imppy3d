"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
March 6, 2023

This script generates synthetic data to be used by the example
scripts. This synthetic data are saved as three image stacks within 
the directories of this folder that this script is saved in.
"""

import numpy as np
from skimage import draw as draw
from skimage import io as io
from skimage import measure as meas
from skimage import filters as filt
from skimage.util import img_as_ubyte, img_as_float, random_noise

# Import local packages.
import imppy3d.import_export as imex
import imppy3d.volume_image_processing as vol


########## Mock Additively Manufactured Metal Cube With Pores ##########

print(f"\nBuilding the additively manufactured metal cube image sequence...")

num_am_pores = 64

v_axis_min = 3
v_axis_max = 5
v_axis_mid = (v_axis_max + v_axis_min)/2.0
v_sigma = (v_axis_max - v_axis_mid)/3.0

# Statistics for the size of voids in some particles (voids will be ellipsoids)
v_ax_a = np.random.default_rng().normal(loc=v_axis_mid, scale=v_sigma, size=num_am_pores)
v_ax_b = np.random.default_rng().normal(loc=v_axis_mid, scale=v_sigma, size=num_am_pores)
v_ax_c = np.random.default_rng().normal(loc=v_axis_mid, scale=v_sigma, size=num_am_pores)
v_ax_a = np.absolute(v_ax_a)
v_ax_b = np.absolute(v_ax_b)
v_ax_c = np.absolute(v_ax_c)

img_arr = np.zeros((256, 256, 256), dtype=np.uint8)
img_arr_shape = img_arr.shape

# Fill in a cubic region of white pixels representing the metal
metal_ii_lower_bound = 32
metal_ii_upper_bound = 256 - 32
img_arr[metal_ii_lower_bound:metal_ii_upper_bound,
        metal_ii_lower_bound:metal_ii_upper_bound,
        metal_ii_lower_bound:metal_ii_upper_bound] = 255

for m in range(num_am_pores):

    cur_a = v_ax_a[m]
    cur_b = v_ax_b[m]
    cur_c = v_ax_c[m]

    img_void = draw.ellipsoid(cur_a, cur_b, cur_c)
    img_void = img_as_ubyte(img_void)
    void_shape = img_void.shape
    void_cent_ii = np.floor(np.array(void_shape)/2)

    coord_max_i = (np.round(img_arr_shape[0] - void_shape[0] - 32)).astype(np.int32)
    coord_max_j = (np.round(img_arr_shape[1] - void_shape[1] - 32)).astype(np.int32)
    coord_max_k = (np.round(img_arr_shape[2] - void_shape[2] - 32)).astype(np.int32)

    coord_i = np.random.randint(32, high=coord_max_i)
    coord_j = np.random.randint(32, high=coord_max_j)
    coord_k = np.random.randint(32, high=coord_max_k)

    img_arr_mask = np.zeros(img_arr_shape, dtype=np.uint8)

    img_arr_mask[coord_i:coord_i+void_shape[0], \
                coord_j:coord_j+void_shape[1], \
                coord_k:coord_k+void_shape[2]] = img_void

    img_arr[img_arr_mask > 0] = 0

    if ((m+1)%10) == 0:
        print(f"  Synthetically generated {m+1}/{num_am_pores} voids...")

print(f"  Synthetically generated {num_am_pores}/{num_am_pores} voids...")

save_path_out = "./porous_metal/"
file_name_out = "metal_am.tif"

imex.save_image_seq(img_arr, save_path_out, file_name_out, compression=True)


########## Mock Particles ##########

print(f"\nBuilding the particles image sequence...")

# Make random ellipsoids positioned randomly, some with random pores
# Extract a subset image stack from this
# Apply a circle mask to imitate X-ray CT
# Post-process with some blurring and noise to make more realistic

num_particles = 64

p_axis_min = 8
p_axis_max = 24

v_axis_min = 3
v_axis_max = 5

p_axis_mid = (p_axis_max + p_axis_min)/2.0
p_sigma = (p_axis_max - p_axis_mid)/3.0

v_axis_mid = (v_axis_max + v_axis_min)/2.0
v_sigma = (v_axis_max - v_axis_mid)/3.0

# Particle size statistics (particles will be ellipsoids)
p_ax_a = np.random.default_rng().normal(loc=p_axis_mid, scale=p_sigma, size=num_particles)
p_ax_b = np.random.default_rng().normal(loc=p_axis_mid, scale=p_sigma, size=num_particles)
p_ax_c = np.random.default_rng().normal(loc=p_axis_mid, scale=p_sigma, size=num_particles)
p_ax_a = np.absolute(p_ax_a)
p_ax_b = np.absolute(p_ax_b)
p_ax_c = np.absolute(p_ax_c)

# Statistics for the size of voids in some particles (voids will be ellipsoids)
v_ax_a = np.random.default_rng().normal(loc=v_axis_mid, scale=v_sigma, size=num_particles)
v_ax_b = np.random.default_rng().normal(loc=v_axis_mid, scale=v_sigma, size=num_particles)
v_ax_c = np.random.default_rng().normal(loc=v_axis_mid, scale=v_sigma, size=num_particles)
v_ax_a = np.absolute(v_ax_a)
v_ax_b = np.absolute(v_ax_b)
v_ax_c = np.absolute(v_ax_c)

# Statistics for whether a particle should contain a void
v_bool_arr = np.random.randint(0, high=5, size=num_particles)

img_arr = np.zeros((256, 256, 256), dtype=np.uint8)
img_arr_shape = img_arr.shape
for m in range(num_particles):

    cur_a = p_ax_a[m]
    cur_b = p_ax_b[m]
    cur_c = p_ax_c[m]

    img_part = draw.ellipsoid(cur_a, cur_b, cur_c)
    img_part = img_as_ubyte(img_part)
    part_shape = img_part.shape
    part_cent_ii = np.floor(np.array(part_shape)/2)

    if v_bool_arr[m] <= 1: # Include a void

        cur_a = v_ax_a[m]
        cur_b = v_ax_b[m]
        cur_c = v_ax_c[m]

        img_void = draw.ellipsoid(cur_a, cur_b, cur_c)
        img_void = img_as_ubyte(img_void)
        void_shape = img_void.shape
        void_cent_ii = np.floor(np.array(void_shape)/2)

        trans_vec = np.round(part_cent_ii - void_cent_ii)
        trans_vec = trans_vec.astype(np.int32)

        void_mask = np.zeros(part_shape, dtype=np.uint8)

        void_mask[trans_vec[0]:trans_vec[0]+void_shape[0], \
                  trans_vec[1]:trans_vec[1]+void_shape[1], \
                  trans_vec[2]:trans_vec[2]+void_shape[2]] = img_void

        img_part[void_mask > 0] = 0

    coord_max_i = np.round(img_arr_shape[0] - part_shape[0])
    coord_max_j = np.round(img_arr_shape[1] - part_shape[1])
    coord_max_k = np.round(img_arr_shape[2] - part_shape[2])

    coord_max_i = coord_max_i.astype(np.int32)
    coord_max_j = coord_max_j.astype(np.int32)
    coord_max_k = coord_max_k.astype(np.int32)

    coord_i = np.random.randint(0, high=coord_max_i)
    coord_j = np.random.randint(0, high=coord_max_j)
    coord_k = np.random.randint(0, high=coord_max_k)

    img_arr_mask = np.zeros(img_arr_shape, dtype=np.uint8)

    img_arr_mask[coord_i:coord_i+part_shape[0], \
                coord_j:coord_j+part_shape[1], \
                coord_k:coord_k+part_shape[2]] = img_part

    img_arr[img_arr_mask > 0] = 255

    if ((m+1)%10) == 0:
        print(f"  Synthetically generated {m+1}/{num_particles} particles...")

print(f"  Synthetically generated {num_particles}/{num_particles} particles...")
print(f"\nApplying post-processing noise and blur...")

img_arr = (np.round(0.667*img_arr)).astype(np.uint8)
img_arr = (img_arr + 40).astype(np.uint8)

img_arr2 = img_as_float(img_arr)
img_arr2 = filt.gaussian(img_arr2, sigma=1)

img_arr2 = random_noise(img_arr2, mode='gaussian', mean=0, var=0.016)

img_arr2 = filt.median(img_arr2, behavior='ndimage')
img_arr2 = img_as_ubyte(img_arr2)

img_arr2_masked = np.zeros(img_arr2.shape, dtype=np.uint8)

circ_mid_i = np.floor(img_arr2.shape[0]/2)
circ_mid_j = np.floor(img_arr2.shape[1]/2)
circ_rad = (np.amax(img_arr2.shape))/2 - 4

rr, cc = draw.disk((circ_mid_i, circ_mid_j), circ_rad)
img_arr2_masked[:, rr, cc] = img_arr2[:, rr, cc]

save_path_out = "./powder_particles/"
file_name_out = "powder.tif"

imex.save_image_seq(img_arr2_masked, save_path_out, file_name_out, compression=True)


########## Hexahedron ##########

print(f"\nBuilding the hexahedron image sequence...")
# Hexahedral box with different width, height, and length
img_seq = np.zeros((128,128,128), dtype=np.uint8)

# Aim for (height, width, length) == (48, 32, 16)
extent = (16,32)
start = (56,48)
for indx in range(40,88):
   cur_img = img_seq[indx]

   rr, cc = draw.rectangle(start, extent=extent, shape=cur_img.shape)
   cur_img[rr,cc] = 255

   img_seq[indx] = cur_img.copy()

save_path_out = "./hexahedron/"
file_name_out = "hexa_data.tif"

imex.save_image_seq(img_seq, save_path_out, file_name_out, compression=True)


# ########## Uniform Grid of Ellipsoids ########## 

# -------- DRAW AN ELLIPSOID --------

# ********** USER INPUTS ********** 
# x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
# Define the semi-major and semi-minor axes via constants a, b, and c.
ellip_a = 5  # Aligned to Z-axis (Image Index)
ellip_b = 9 # Aligned to Y-axis (Row Index)
ellip_c = 17 # Aligned to X-axis (Column Index)
# ********************************* 

print(f"\nBuilding ellipsoid array...")
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

    #print(f"Rotating the ellipsoid...")

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

# This automatically clips the domain down
img_ellip = feat_props[0].image
img_ellip = img_as_ubyte(img_ellip)

# Update sizes
ellip_n_imgs = img_ellip.shape[0] # Number of images
ellip_n_rows = img_ellip.shape[1] # Number of rows 
ellip_n_cols = img_ellip.shape[2] # Number of columns 


# -------- TILE THE ELLIPSOID RVE INTO A LARGER IMAGE SEQUENCE --------

# ********** USER INPUTS ********** 
img_repeats = 8
row_repeats = 8
col_repeats = 8
# ********************************* 

#print(f"Creating a periodic array of ellipsoid cells...")

img_arr = np.zeros((img_repeats*ellip_n_imgs, \
    row_repeats*ellip_n_rows, col_repeats*ellip_n_cols), dtype=np.uint8)

for ii in np.arange(img_repeats):
    for rr in np.arange(row_repeats):
        for cc in np.arange(col_repeats):

            ii_0 = ii*ellip_n_imgs
            ii_1 = ii_0 + ellip_n_imgs

            rr_0 = rr*ellip_n_rows
            rr_1 = rr_0 + ellip_n_rows

            cc_0 = cc*ellip_n_cols
            cc_1 = cc_0 + ellip_n_cols

            img_arr[ii_0:ii_1, rr_0:rr_1, cc_0:cc_1] = img_ellip


# -------- SAVE IMAGE SEQUENCE --------

# ********** USER INPUTS ********** 
save_path_out = "./ellipsoid_pattern/"
file_name_out = "ellip_array.tif"
# ********************************* 

# Uncomment the line below to save images
imex.save_image_seq(img_arr, save_path_out, file_name_out, compression=True)