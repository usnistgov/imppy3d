"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 27, 2023

Script designed to segment multiple images into black and white pixels, 
called binarization. This is done in a non-interactive (i.e., batch) way. 
"""

# Import external dependencies
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Import local modules
import imppy3d.import_export as imex
import imppy3d.cv_processing_wrappers as cwrap
import imppy3d.cv_driver_functions as drv



# -------- IMPORT IMAGE SEQUENCE --------

# See documentation on load_image_seq() for function details
dir_in_path = "../resources/powder_particles/"
name_in_substr = "powder"
imgs_keep = (0,21) # Change this to (256,) to process all images

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

# Extract the image size properties from the 3D Numpy array
num_imgs = imgs.shape[0] # Number of images
num_rows = imgs.shape[1] # Number of rows of pixels in each image
num_cols = imgs.shape[2] # Number of columns of pixels in each image


# -------- (OPTIONAL) CROPPING --------
# Image cropping needs to occur first since this will change the size of
# the 3D image array, thus requiring a temporary copy to be made.
roi_crop = (256, 256) # Tuple containing two integers (rows, cols)
imgs_temp = imgs.copy()
imgs = np.zeros((num_imgs, roi_crop[0], roi_crop[1]), dtype=np.uint8)

# Changing how many images to loop through can effectively crop the image
# sequence in the Z-direction too. Just be sure to update the size of imgs
for img_index, cur_img in enumerate(imgs_temp):
    
    # Crop the image about the center pixel to the desired width and height
    img2 = cwrap.crop_img(cur_img, roi_crop, quiet_in=True)

    # Store the newly cropped image
    imgs[img_index,:,:] = img2.copy()

# Update the image size properties from the 3D Numpy array
num_imgs = imgs.shape[0] # Number of images
num_rows = imgs.shape[1] # Number of rows of pixels in each image
num_cols = imgs.shape[2] # Number of columns of pixels in each image


# -------- NORMALIZE HISTOGRAM --------

# Linearly normalize the intensity histogram to range from 0 to 255
# for the whole image sequence at once.
imgs_temp_2d = imgs_temp.reshape((num_imgs*num_rows, num_cols))
imgs_temp_2d = cwrap.normalize_histogram(imgs_temp_2d, quiet_in=True)
imgs_temp = imgs_temp_2d.reshape((num_imgs, num_rows, num_cols))

imgs = imgs_temp.copy()

del imgs_temp, imgs_temp_2d


# -------- LOOP THROUGH AND PROCESS EACH IMAGE INDIVIDUALLY --------

print(f"\nBeginning to process images:")
# Also could get each image via slice operations, img0 = imgs[0,:,:]
for img_index, cur_img in enumerate(imgs):


    # -------- EROSION/DILATION/OPEN/CLOSE MORPHOLOGICAL OPERATIONS --------
    # See documentation on apply_driver_morph() for function details
    flag_open_close = 1
    flag_rect_ellps = 1
    k_size = 3
    num_erode = 0
    num_dilate = 0
    morph_params = [flag_open_close, flag_rect_ellps, k_size, num_erode, 
                    num_dilate]
    img2 = drv.apply_driver_morph(img2, morph_params, quiet_in=True)


    # -------- NON-LOCAL MEANS DENOISING FILTER --------
    # See documentation on apply_driver_denoise() for function details
    cur_h = 15
    cur_tsize = 5
    cur_wsize = 27
    denoise_params = [cur_h, cur_tsize, cur_wsize]
    img2 = drv.apply_driver_denoise(img2, denoise_params, quiet_in=True)


    # ---------- BLUR FILTER ----------
    # Do not need an additional blur after non-local means denoising, but 
    # including it here for demonstrative purposes. See documentation on
    # apply_driver_blur() for function details.
    d_size = 4
    sig_intsty = 0 
    fltr_params = ["bilateral", d_size, sig_intsty]
    img2 = drv.apply_driver_blur(img2, fltr_params, quiet_in=True)


    # -------- SHARPEN FILTER --------
    # Do not need a sharpen filter after the blur filter, but including it
    # here for demonstrative purposes. See documentation on 
    # apply_driver_sharpen() for function details.
    cur_amount = 110 # In percent
    k_size = 5
    fltr_type = 0
    std_dev = -1
    fltr_params = ["unsharp", cur_amount, k_size, fltr_type, std_dev]
    img2 = drv.apply_driver_sharpen(img2, fltr_params, quiet_in=True)


    #  -------- GLOBAL/ADAPTIVE HISTOGRAM EQUALIZATION --------
    # Tends to add background noise, and is not really needed here. Including
    # it for demonstrative purposes. See documentation on 
    # apply_driver_equalize() for function details
    clip_limit = 1
    grid_size = 0
    eq_params = ["adaptive", clip_limit, grid_size]
    img2 = drv.apply_driver_equalize(img2, eq_params, quiet_in=True)


    #  -------- IMAGE SEGMENTATION/BINARIZATION --------
    # See documentation on apply_driver_thresh() for function details
    cur_thsh = 105
    thsh_params = ["global", cur_thsh]
    img2 = drv.apply_driver_thresh(img2, thsh_params, quiet_in=True)


    # -------- EROSION/DILATION/OPEN/CLOSE MORPHOLOGICAL OPERATIONS --------
    # See documentation on apply_driver_morph() for function details
    flag_open_close = 0
    flag_rect_ellps = 0
    k_size = 2
    num_erode = 0
    num_dilate = 0
    morph_params = [flag_open_close, flag_rect_ellps, k_size, num_erode, 
                    num_dilate]
    img2 = drv.apply_driver_morph(img2, morph_params, quiet_in=True)


    #  -------- REMOVE PIXEL ISLANDS/ARTIFACTS --------
    # See documentation on apply_driver_blob_fill() for function details
    area_thresh_min = 0
    area_thresh_max = 1
    circty_thresh_min = 0.0
    circty_thresh_max = 1.1
    ar_min = 0.0
    ar_max = 10.0
    COLOR = (0, 0, 0)
    blob_params = [area_thresh_min, area_thresh_max, circty_thresh_min,
                   circty_thresh_max, ar_min, ar_max, COLOR]
    img2 = drv.apply_driver_blob_fill(img2, blob_params, quiet_in=True)


    # -------- STORE PROCESSED IMAGE IN THE ORIGINAL NUMPY ARRAY --------

    imgs[img_index,:,:] = img2.copy()


    # ---------- PROGRESS UPDATE TO THE TERMINAL ----------
    if (((img_index+1) % 10) == 0):
        print(f"    Currently processed {img_index+1}/{num_imgs}...")

print(f"\nSuccessfully processed {num_imgs} images!")


# -------- SAVE IMAGE SEQUENCE --------

# Uncomment below to save the processed images
#save_path_out = "./"
#file_name_out = "powder_binary.tif"
#imex.save_image_seq(imgs, save_path_out, file_name_out)