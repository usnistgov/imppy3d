"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 27, 2023

Script designed to segment an image into black and white pixels, called
binarization. This is done in a non-interactive (i.e., batch) way. The
implementation makes calls to OpenCV algorithms, however, the imppy3d
library also contains wrapper scripts to apply similar image post-
processing techniques via Scikit-Image. For more information, look into
the python script called "ski_driver_functions.py" in the "functions"
folder.
"""

# Import external dependencies
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../functions') 
import import_export as imex
import plt_wrappers as pwrap
import cv_processing_wrappers as cwrap
import cv_driver_functions as drv


# -------- IMPORT IMAGE FILE --------

file_path = "../resources/powder_particles/powder_0100.tif"
img1, img1_prop = imex.load_image(file_path)

if img1 is None:
    print(f"\nFailed to import images from the directory: \n{file_path}")
    print("\nDouble-check that the example images exist, and if not, run")
    print("the Python script that creates all of the sample data in:")
    print(f"../resources/generate_sample_data.py")
    print("\nQuitting the script...")

    quit()

# Optionally, extract the (Numpy) properties of image.
img1_size = img1_prop[0]  # Total number of pixels
img1_shape = img1_prop[1] # Tuple containing the number of rows and columns
img1_dtype = img1_prop[2] # Returns the image data type (i.e., uint8)


# -------- (OPTIONAL) CROP IMAGE AND NORMALIZE HISTOGRAM --------

# Crop the image about the center pixel to the desired width and height
roi_crop = (256, 256) # Tuple containing two integers (rows, cols)
img1 = cwrap.crop_img(img1, roi_crop)

# Linearly normalize the intensity histogram to range from 0 to 255
img2 = cwrap.normalize_histogram(img1)


# -------- EROSION/DILATION/OPEN/CLOSE MORPHOLOGICAL OPERATIONS --------
# See documentation on apply_driver_morph() for function details
flag_open_close = 1
flag_rect_ellps = 1
k_size = 3
num_erode = 0
num_dilate = 0
morph_params = [flag_open_close, flag_rect_ellps, k_size, num_erode, 
                num_dilate]
img2 = drv.apply_driver_morph(img2, morph_params)


# -------- NON-LOCAL MEANS DENOISING FILTER --------
# See documentation on apply_driver_denoise() for function details
cur_h = 15
cur_tsize = 5
cur_wsize = 27
denoise_params = [cur_h, cur_tsize, cur_wsize]
img2 = drv.apply_driver_denoise(img2, denoise_params)


# ---------- BLUR FILTER ----------
# Do not need an additional blur after non-local means denoising, but 
# including it here for demonstrative purposes. See documentation on
# apply_driver_blur() for function details.
d_size = 4
sig_intsty = 0 
fltr_params = ["bilateral", d_size, sig_intsty]
img2 = drv.apply_driver_blur(img2, fltr_params)


# -------- SHARPEN FILTER --------
# Do not need a sharpen filter after the blur filter, but including it
# here for demonstrative purposes. See documentation on 
# apply_driver_sharpen() for function details.
cur_amount = 110 # In percent
k_size = 5
fltr_type = 0
std_dev = -1
fltr_params = ["unsharp", cur_amount, k_size, fltr_type, std_dev]
img2 = drv.apply_driver_sharpen(img2, fltr_params)


#  -------- GLOBAL/ADAPTIVE HISTOGRAM EQUALIZATION --------
# Tends to add background noise, and is not really needed here. Including
# it for demonstrative purposes. See documentation on 
# apply_driver_equalize() for function details
clip_limit = 1
grid_size = 0
eq_params = ["adaptive", clip_limit, grid_size]
img2 = drv.apply_driver_equalize(img2, eq_params)


#  -------- IMAGE SEGMENTATION/BINARIZATION --------
# See documentation on apply_driver_thresh() for function details
cur_thsh = 105
thsh_params = ["global", cur_thsh]
img2 = drv.apply_driver_thresh(img2, thsh_params)


# -------- EROSION/DILATION/OPEN/CLOSE MORPHOLOGICAL OPERATIONS --------
# See documentation on apply_driver_morph() for function details
flag_open_close = 0
flag_rect_ellps = 0
k_size = 2
num_erode = 0
num_dilate = 0
morph_params = [flag_open_close, flag_rect_ellps, k_size, num_erode, 
                num_dilate]
img2 = drv.apply_driver_morph(img2, morph_params)


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
img2 = drv.apply_driver_blob_fill(img2, blob_params)


# -------- CALCULATE RELATIVE DENSITY --------

rel_density = cwrap.calc_rel_density(img2)


# -------- BEFORE AND AFTER PICTURE --------

# Show the final image compared against the original image
fig1, ax1 = plt.subplots(2, 2, sharex='row', sharey='row')
fig1.set_size_inches(12, 10) # Big enough, and still fit in modern monitors

# Include vmin & vmax, else imshow() automatically normalizes from 0 to 1
ax1[0,0].set_aspect('equal')
ax1[0,0].imshow(img1, cmap='gray', vmin=0, vmax=255)
ax1[0,0].set_title("Original Image")
ax1[0,0].set_xlabel("X Pixel Number")
ax1[0,0].set_ylabel("Y Pixel Number")

ax1[0,1].set_aspect('equal')
ax1[0,1].imshow(img2, cmap='gray', vmin=0, vmax=255)
ax1[0,1].set_title("Final Image")
ax1[0,1].set_xlabel("X Pixel Number")

ax1[1,0].hist(img1.ravel(),256,[0,256])
ax1[1,0].set_title("Original Histogram")
ax1[1,0].set_xlabel("Grayscale Intensity")
ax1[1,0].set_ylabel("Counts") 

ax1[1,1].hist(img2.ravel(),256,[0,256])
ax1[1,1].set_title("Final Histogram")
ax1[1,1].set_xlabel("Grayscale Intensity")

# Makes the subplots fit better in the figure's canvas. 
plt.tight_layout() 
plt.show()


# -------- (OPTIONALLY) SAVE IMAGE TO THE HARD DRIVE --------

# Uncomment lines below to save the image

#file_out_path = "./foam_vn02_binary.tif"
#save_flag = imex.save_image(img2, file_out_path)
#if not save_flag:
#    print(f"\nCould not save {file_out_path}")


# -------- TERMINATION OF SCRIPT --------

plt.close('all') # Close all matplotlib figures
cv.destroyAllWindows() # Close all OpenCV windows