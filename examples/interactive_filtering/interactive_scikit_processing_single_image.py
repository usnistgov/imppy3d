"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
November 8, 2023

Script designed to segment an image into black and white pixels, called
binarization, through interactive means. The interactive functions
make calls to SciKit-Image algorithms, however, the imppy3d library also
contains some wrapper scripts to interactively post-process an image
using OpenCV.
"""

# Import external dependencies
import numpy as np
from skimage.util import img_as_ubyte, img_as_bool
from skimage.util import invert as ski_invert
from matplotlib import pyplot as plt

# Import local modules
import imppy3d.cv_processing_wrappers as cwrap
import imppy3d.import_export as imex
import imppy3d.plt_wrappers as pwrap
import imppy3d.ski_driver_functions as sdrv


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


# -------- USER INPUTS --------

# Provide the filepath to the image that should be imported and segmented.
# Imported images will be converted to grayscale as UINT8 (i.e., max
# grayscale intensities of 255).
file_in_path = "../resources/sem_data_ti64/sem_alpha_beta_ti64.tif"

# The resultant segmentation will be saved in the following filepath.
file_out_path = "../resources/sem_data_ti64/sem_alpha_beta_ti64_segmented.tif"

# Segmentation should result in the grain boundaries being WHITE. If the
# resultant segmentation illustrates black grain boundaries, then the image
# grayscale values should be inverted after it is imported.
invert_grayscales = False



# -------- IMPORT IMAGE FILE --------
img1, img1_prop = imex.load_image(file_in_path)

if img1 is None:
    print(f"\nFailed to import images from the directory: \n{file_in_path}")
    print("\nQuitting the script...")

    quit()

# Extract the (Numpy) properties of image.
img1 = img_as_ubyte(img1)
img1_size = img1_prop[0]  # Total number of pixels
img1_shape = img1_prop[1] # Tuple containing the number of rows and columns
img1_dtype = img1_prop[2] # Returns the image data type (i.e., uint8)

if invert_grayscales:
    img1 = img_as_ubyte(ski_invert(img1))

img2 = img1.copy()


# -------- NON-LOCAL MEANS DENOISING FILTER --------
print(f"\nInitiating interactive non-local means denoising...")
img2 = sdrv.interact_driver_denoise(img2, "nl_means")


# --------- INTERACTIVE CANNY EDGE ENHANCEMENT --------
#print(f"\nInitiating interactive canny edge enhancement...")
#img2, canny_mask = sdrv.interact_driver_edge_detect(img2, "canny")


# -------- SHARPEN FILTER --------
print(f"\nInitiating interactive sharpening filter...")
img2 = sdrv.interact_driver_sharpen(img2, "unsharp_mask")


# -------- GLOBAL THRESHOLDING --------
print(f"\nInitiating interactive global thresholding...")
img2 = sdrv.interact_driver_thresholding(img2, "global_threshold")


# -------- ADAPTIVE THRESHOLDING --------
#print(f"\nInitiating interactive adaptive thresholding...")
#img2 = sdrv.interact_driver_thresholding(img2, "adaptive_threshold")


# -------- HYSTERESIS THRESHOLDING --------
#print(f"\nInitiating interactive hysteresis thresholding...")
#img2 = sdrv.interact_driver_thresholding(img2, "hysteresis_threshold_text")


# -------- MORPHOLOGICAL OPERATIONS --------
print(f"\nInitiating interactive morphological binary operation...")
img2 = sdrv.interact_driver_morph(img2)


# -------- REMOVE PIXEL ISLANDS AND SMALL HOLES --------
print(f"\nInitiating interactive deletion of small features and filling of small holes...")
img2 = sdrv.interact_driver_del_features(img2)


# -------- SKELETONIZE --------
print(f"\nInitiating interactive skeletonization...")
img2 = sdrv.interact_driver_skeletonize(img2)


# -------- CALCULATE RELATIVE DENSITY --------

rel_density = cwrap.calc_rel_density(img2)


# -------- BEFORE AND AFTER PICTURE --------

fig1, ax1 = pwrap.create_2_bw_figs(img1, img2)
fig1.suptitle('Left: Original Image   |   Right: Segmented Image')

plt.tight_layout() 
plt.show()


# -------- (OPTIONALLY) SAVE IMAGE TO THE HARD DRIVE --------

save_flag = imex.save_image(img2, file_out_path)
if not save_flag:
    print(f"\nCould not save {file_out_path}")


# -------- TERMINATION OF SCRIPT --------

plt.close('all') # Close all matplotlib figures
print("\nInteractive script successfully finished!")