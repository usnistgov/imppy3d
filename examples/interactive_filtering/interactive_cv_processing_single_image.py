"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
February 27, 2023

Script designed to segment an image into black and white pixels, called
binarization, through interactive means. The interactive functions
make calls to OpenCV algorithms, however, the imppy3d library also
contains some wrapper scripts to interactively post-process an image
using Skikit-Image. For more information, look into the python
script called "ski_driver_functions.py" in the "functions" folder.

WARNING: Due to updates to the OpenCV library, the interactive windows
do not display correctly. This makes it difficult to read the prompts
to the slider variables. Consider using the Skikit-Image interactive
windows instead.
"""

# Import external dependencies
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import imppy3d.import_export as imex
import imppy3d.cv_processing_wrappers as cwrap
import imppy3d.cv_driver_functions as drv

# Set constants related to plotting (for MatPlotLib)
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
roi_crop = (256, 256) # Tuple containing two integers: (rows, cols)
img1 = cwrap.crop_img(img1, roi_crop)

# Linearly normalize the intensity histogram to range from 0 to 255
img2 = cwrap.normalize_histogram(img1)


# -------- EROSION/DILATION/OPEN/CLOSE MORPHOLOGICAL OPERATIONS --------
# Applied here to the grayscale images, but this same function also works for
# binarized (black-and-white) images
img2 = drv.interact_driver_morph(img2)


# -------- NON-LOCAL MEANS DENOISING FILTER --------
# Note, these filters are using OpenCV's library. Although not interactive, the
# SciKit-Image Python library has an extensive selection of filters, too.
img2 = drv.interact_driver_denoise(img2)


# -------- BLUR FILTER --------
# Probably do not need a blur filter on top of non-local means denoising.
# Use either one or the other. Just including here for demonstrative purposes.
# Supported input strings are "average", "gaussian", "median", and 
# "bilateral". Note, "median" and "bilateral" are edge-preserving. Note, 
# these filters are using OpenCV's library. Although not interactive,
# the SciKit-Image Python library has an extensive selection of filters, too.
blur_fltr = "bilateral"
img2 = drv.interact_driver_blur(img2, blur_fltr)


# -------- SHARPEN FILTER --------
# Do not usually do a sharpen immediately after a blur, but again, included
# here for demonstrative purposes. Supported input strings "unsharp", 
# "laplacian", and "canny". Either the unsharp mask or the Laplacian method 
# is recommended. Note, these filters are using OpenCV's library. Although not
# interactive, the SciKit-Image Python library has an extensive selection of
# filters, too.
edge_fltr = "unsharp"
img2 = drv.interact_driver_sharpen(img2, edge_fltr)


#  -------- GLOBAL/ADAPTIVE HISTOGRAM EQUALIZATION --------
# Can create undesirable background noise, like sharpen filters do. Use either 
# "global" or "adaptive" methods.
eq_type = "adaptive"
img2 = drv.interact_driver_equalize(img2, eq_type)


#  -------- IMAGE SEGMENTATION/BINARIZATION --------
# Supported input strings include "global", "adaptive_mean", and
# "adaptive_gaussian". Note, Otsu's method is available through the "global"
# method.
thresh_type = "global"
img2 = drv.interact_driver_thresh(img2, thresh_type)

# Note, use the following function to invert the black/white pixels
# img2 = cwrap.invert_binary_image(img2)


# -------- EROSION/DILATION/OPEN/CLOSE MORPHOLOGICAL OPERATIONS --------

img2 = drv.interact_driver_morph(img2)


#  -------- REMOVE PIXEL ISLANDS/ARTIFACTS --------

img2 = drv.interact_driver_blob_fill(img2)


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

#file_out_path = "./filtered_image.tif"
#save_flag = imex.save_image(img2, file_out_path)
#if not save_flag:
#    print(f"\nCould not save {file_out_path}")


# -------- TERMINATION OF SCRIPT --------

plt.close('all') # Close all matplotlib figures
cv.destroyAllWindows() # Close all OpenCV windows