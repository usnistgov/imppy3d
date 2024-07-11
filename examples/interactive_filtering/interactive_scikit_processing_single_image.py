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
import sys
import numpy as np
import cv2 as cv
import skimage.feature as feature
from skimage import morphology as morph
from skimage import filters as sfilt
from skimage.util import img_as_ubyte, img_as_float, img_as_bool
from skimage.util import invert as ski_invert
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button, RadioButtons

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../functions') 
import import_export as imex
import plt_wrappers as pwrap
import ski_driver_functions as sdrv
import cv_processing_wrappers as cwrap

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


# --------- INTERACTIVE CANNY EDGE --------

def interact_canny_edge(img_in):

    img_0 = img_in.copy()

    # Initial values for the filter
    sigma_0 = 2.0
    low_thresh_0 = 0.1*255
    high_thresh_0 = 0.2*255

    img_mask_0 = feature.canny(img_0, sigma=sigma_0, 
        low_threshold=low_thresh_0, high_threshold=high_thresh_0)

    img_temp = img_0.copy()
    img_temp[img_mask_0] = img_0[img_mask_0]/2

    # Global variables (within this function) that will be returned. 
    # Initialize these to the starting values for now.
    sigma_out = sigma_0
    low_thresh_out = low_thresh_0
    high_thresh_out = high_thresh_0
    img_out = img_temp
    mask_out = img_as_ubyte(img_mask_0)

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    # Create new axes objects for each button/slider/text widget
    # 4-tuple of floats rect = [left, bottom, width, height]. A new axes 
    # is added with dimensions rect in normalized (0, 1) units using 
    # add_axes on the current figure.
    sigma_ax = fig.add_axes([0.1, 0.11, 0.15, 0.06])
    sigma_text_box = TextBox(ax=sigma_ax, label='Sigma  ', 
        initial=str(sigma_0), textalignment='center')

    low_thresh_ax = fig.add_axes([0.45, 0.11, 0.15, 0.06])
    low_thresh_text_box = TextBox(ax=low_thresh_ax, label='Low Threshold  ', 
        initial=str(low_thresh_0), textalignment='center')

    high_thresh_ax = fig.add_axes([0.8, 0.11, 0.15, 0.06])
    high_thresh_text_box = TextBox(ax=high_thresh_ax, label='High Threshold  ', 
        initial=str(high_thresh_0), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    # Update the figure anytime the 'update' button is clicked
    def canny_edge_update(event):
        # Use the new Python keyword 'nonlocal' to gain access and 
        # update these variables from within this scope.
        nonlocal sigma_out
        nonlocal low_thresh_out
        nonlocal high_thresh_out
        nonlocal img_out
        nonlocal mask_out

        # The GUI widgets are defined in a higher-level scope, so
        # they can be accessed directly within this interior function 
        sigma_out = float(sigma_text_box.text)
        low_thresh_out = float(low_thresh_text_box.text)
        high_thresh_out = float(high_thresh_text_box.text)

        img_mask = feature.canny(img_0, sigma=sigma_out, 
            low_threshold=low_thresh_out, high_threshold=high_thresh_out)

        mask_out = img_as_ubyte(img_mask)
        img_out = img_0.copy()
        img_out[img_mask] = img_0[img_mask]/2

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(canny_edge_update)

    plt.show()

    # Save final filter parameters
    fltr_params = ["canny", sigma_out, low_thresh_out, high_thresh_out]

    return [img_out, mask_out, fltr_params]

# Uncomment below to use Canny edge detection. 
#print(f"\nInitiating interactive canny edge detection...")
#[img2, canny_mask, canny_params] = interact_canny_edge(img2)
#print(f"\nSuccessfully applied the 'canny' edge detection filter:\n"\
#    f"    Pre-filter Gaussian blur sigma: {canny_params[1]}\n"\
#    f"    Hysteresis lower threshold: {canny_params[2]}\n"\
#    f"    Hysteresis upper threshold: {canny_params[3]}" )


# -------- SHARPEN FILTER --------
print(f"\nInitiating interactive sharpening filter...")
img2 = sdrv.interact_driver_sharpen(img2, "unsharp_mask")


def interact_adaptive_thresholding(img_in):
    img_0 = img_in.copy()

    # Initial values
    block_sz = 3
    thresh_offset = 0

    img_thresh = sfilt.threshold_local(img_0, block_size=block_sz, 
        method='gaussian', offset=thresh_offset)

    img_out = img_as_ubyte(img_0 > img_thresh)

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.26)

    block_sz_ax = fig.add_axes([0.20, 0.12, 0.15, 0.06])
    block_sz_txt_box = TextBox(ax=block_sz_ax, label='Block Size (odd)  ', 
        initial=str(block_sz), textalignment='center')

    filt_off_ax = fig.add_axes([0.70, 0.12, 0.15, 0.06])
    filt_off_txt_box = TextBox(ax=filt_off_ax, label='Offset (+/-)  ', 
        initial=str(thresh_offset), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    def adapt_thresh_update(event):
        nonlocal block_sz
        nonlocal thresh_offset
        nonlocal img_out

        block_sz = int(block_sz_txt_box.text)
        thresh_offset = float(filt_off_txt_box.text)

        if block_sz % 2 == 0:
            block_sz = block_sz + 1

        img_thresh = sfilt.threshold_local(img_0, block_size=block_sz, 
            method='gaussian', offset=thresh_offset)

        img_out = img_as_ubyte(img_0 > img_thresh)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(adapt_thresh_update)

    plt.show()

    # Save final filter parameters
    fltr_params = [block_sz, thresh_offset]

    return [img_out, fltr_params]

# Uncomment below to use adaptive thresholding
print(f"\nInitiating interactive adaptive thresholding...")
img2, thresh_params = interact_adaptive_thresholding(img2)

print(f"\nSuccessfully applied adaptive thresholding:")
print(f"    Block size: {thresh_params[0]}")
print(f"    Intensity offset: {thresh_params[1]}")


# -------- HYSTERESIS THRESHOLDING --------
#print(f"\nInitiating interactive hysteresis thresholding...")
#img2 = sdrv.interact_driver_thresholding(img2, "hysteresis_threshold_text")


# -------- MORPHOLOGICAL OPERATIONS --------
print(f"\nInitiating interactive morphological binary operation...")
img2 = sdrv.interact_driver_morph(img2)


# -------- REMOVE PIXEL ISLANDS AND SMALL HOLES --------

def interact_del_features(img_in):

    img_0 = img_in.copy()

    # Initial values
    min_feat_sz = 3
    max_hole_sz = 9 

    img_bool = img_as_bool(img_0)
    img_temp = morph.remove_small_holes(img_bool, max_hole_sz, connectivity=1)
    img_out = morph.remove_small_objects(img_temp, min_feat_sz, connectivity=1)
    img_out = img_as_ubyte(img_out)

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.35)

    min_feat_ax = fig.add_axes([0.45, 0.20, 0.15, 0.06])
    min_feat_txt_box = TextBox(ax=min_feat_ax, label='Min Feature Size to Delete (Pixels)  ', 
        initial=str(min_feat_sz), textalignment='center')

    max_area_ax = fig.add_axes([0.45, 0.11, 0.15, 0.06])
    max_area_txt_box = TextBox(ax=max_area_ax, label='Max Area of Holes to Fill (Pixels)  ', 
        initial=str(max_hole_sz), textalignment='center')

    update_ax = plt.axes([0.25, 0.03, 0.5, 0.05])
    update_button = Button(update_ax, 'Update')

    def del_features_update(event):
        nonlocal min_feat_sz
        nonlocal max_hole_sz
        nonlocal img_out

        # Initial values
        min_feat_sz = int(min_feat_txt_box.text)
        max_hole_sz = int(max_area_txt_box.text)

        img_bool = img_as_bool(img_0)
        img_temp = morph.remove_small_holes(img_bool, max_hole_sz, connectivity=1)
        img_out = morph.remove_small_objects(img_temp, min_feat_sz, connectivity=1)
        img_out = img_as_ubyte(img_out)

        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(del_features_update)

    plt.show()

    # Save final filter parameters
    fltr_params = [max_hole_sz, min_feat_sz]

    return [img_out, fltr_params]

print(f"\nInitiating interactive deletion of small features and filling small holes...")
img2, del_params = interact_del_features(img2)

print(f"\nSuccessfully filled in holes:")
print(f"    Maximum hole (area) size: {del_params[0]} pixels")

print(f"\nSuccessfully deleted features:")
print(f"    Minimum feature (area) size: {del_params[1]} pixels")


# -------- SKELETONIZE --------

def interact_skeletonize(img_in):

    img_0 = img_in.copy()

    # Initial values
    apply_skel = True

    img_bool = img_as_bool(img_0)
    img_temp = morph.skeletonize(img_bool)
    img_out = img_as_ubyte(img_temp)

    # Create a figure
    fig_size = (9,9)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax.set_aspect('equal')

    # Show the image and save the "matplotlib.image.AxesImage"
    # object for updating the figure later.
    img_obj = ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    ax.set_xlabel("X Pixel Number")
    ax.set_ylabel("Y Pixel Number")

    plt.subplots_adjust(bottom=0.25)

    apply_skel_ax = fig.add_axes([0.12, 0.03, 0.25, 0.15])
    apply_skel_radio = RadioButtons(ax=apply_skel_ax, labels=[" True: skeletonize",
        " False: skeletonize"], active=0)

    update_ax = fig.add_axes([0.62, 0.07, 0.25, 0.05])
    update_button = Button(update_ax, 'Update')

    def skeletonize_update(event):
        nonlocal apply_skel
        nonlocal img_out

        if apply_skel_radio.value_selected == " True: skeletonize":
            apply_skel = True
        else:
            apply_skel = False

        if apply_skel:
            img_bool = img_as_bool(img_0)
            img_temp = morph.skeletonize(img_bool)
            img_out = img_as_ubyte(img_temp)

        else:
            img_out = img_0
        
        # Update the image
        img_obj.set(data=img_out)

    # Call the update function when the 'update' button is clicked
    update_button.on_clicked(skeletonize_update)

    plt.show()

    # Save final filter parameters
    fltr_params = [apply_skel]

    return [img_out, fltr_params]

print(f"\nInitiating interactive skeletonization...")
img2, skel_params = interact_skeletonize(img2)

print(f"\nSuccessfully computed the skeleton of the binary image:")
print(f"    Applied skeletonize: {skel_params[0]}")


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