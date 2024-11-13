# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import interactive_3d_vol_viewer as vol_view
from skimage.feature import match_template

import imppy3d.import_export as imex
import imppy3d.cv_processing_wrappers as wrap
import imppy3d.cv_interactive_processing as intprc

eps_float = 1e-12

"""
Autocorrelation-based feature size identifcation.

Source code for 3d image handling from Newell Moser

@author: Alexander K Landauer, MML, NIST
"""

# -------- IMPORT IMAGE SEQUENCE --------

# See documentation on load_image_seq() for function details
dir_in_path = "../resources/ellipsoid_pattern/"
name_in_substr = "ellip_array"
imgs_keep = (0,250)

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

# -------- CROPPING --------
# Image cropping needs to occur first since this will change the size of
# the 3D image array, thus requiring a temporary copy to be made.
roi_crop = (128, 128) # Tuple containing two integers (rows, cols)
imgs_temp = imgs.copy()
imgs = np.zeros((num_imgs, roi_crop[0], roi_crop[1]), dtype=np.uint8)

# Changing how many images to loop through can effectively crop the image
# sequence in the Z-direction too. Just be sure to update the size used
# to define imgs = np.zeros() above
for img_index, cur_img in enumerate(imgs_temp):

    # Crop the image about the center pixel to the desired width and height
    img2 = wrap.crop_img(cur_img, roi_crop, quiet_in=True)

    # Store the newly cropped image
    imgs[img_index,:,:] = img2.copy()

del imgs_temp

# Update the image size properties from the 3D Numpy array
num_imgs = imgs.shape[0] # Number of images
num_rows = imgs.shape[1] # Number of rows of pixels in each image
num_cols = imgs.shape[2] # Number of columns of pixels in each image


###  start with de-noising (filtering)
img_denoise = np.zeros((roi_crop[0], roi_crop[1]), dtype=np.uint8)
[img_denoise, denoise_params] = intprc.interact_denoise(imgs[round(num_imgs/2),:,:])

for img_index in range(num_imgs):
    imgs[img_index,:,:] = cv.fastNlMeansDenoising(imgs[img_index,:,:], h=denoise_params[0],
                templateWindowSize=denoise_params[1], searchWindowSize=denoise_params[2])
    if np.mod(img_index,10.0) <= 2*eps_float:
        print(f"\nCurrent denoise image: {img_index}")

###  define basic metrics
N = np.size(imgs) # number of pixels
I_bar = np.mean(imgs)
sigma_I = np.std(imgs)

# run autocorrelation of the full volume on itself
print('\nRunning cross correlation...')
acf_vol = imgs.transpose([1,2,0])
pad = bool(1)
nxcorrcoeff_vol = match_template(acf_vol,acf_vol,pad)
print('\nCross correlation complete')

# set up coordinate system
sizeACF = [0,0,0]
for dim in range(3):
    sizeACF[dim] = nxcorrcoeff_vol.shape[dim]

px_x_v = np.linspace(1,sizeACF[0],(sizeACF[0]))
px_y_v = np.linspace(1,sizeACF[1],(sizeACF[1]))
px_z_v = np.linspace(1,sizeACF[2],(sizeACF[2]))

[X,Y,Z] = np.meshgrid(px_y_v,px_x_v,px_z_v)

X = X - np.fix(sizeACF[0]/2)
Y = Y - np.fix(sizeACF[1]/2)
Z = Z - np.fix(sizeACF[2]/2)

# for each point in the carteasian grid compute the radius
r = np.sqrt(np.power(X,2) + np.power(Y,2)+ np.power(Z,2))
# theta = np.arctan2(np.sqrt(np.power(X,2)+np.power(Y,2)),Z)
# phi = np.zeros(sizeACF)
# for i in range(sizeACF[0]):
#     for j in range(sizeACF[1]):
#         for k in range(sizeACF[2]):
#             if X[i,j,k] >= 0:
#                 phi[i,j,k] = np.arctan2(Y[i,j,k],X[i,j,k])
#             else:
#                 phi[i,j,k] = np.arctan2(Y[i,j,k],X[i,j,k])+np.pi

# for each integer radius ("fix"-type rounding to bin fractional radii)
# compute the mean and st dev autocorrelation coeffiecent 
r_fix = np.fix(r)
r_max = int(np.max(r_fix))
corr_coeff_mean = np.zeros([r_max-1,1])
corr_coeff_std = np.zeros([r_max-1,1])
for r_val in range(r_max-1):
    corr_coeff_mean[r_val] = np.mean(nxcorrcoeff_vol[r_fix == r_val])
    corr_coeff_std[r_val] = np.std(nxcorrcoeff_vol[r_fix == r_val])

### plotting

# open an interactive volume viewer (use 'j' and 'k')
#vol_view.multi_slice_viewer(nxcorrcoeff_vol)

fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(range(r_max-1), corr_coeff_mean, label='corr coeff mean')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line2, = ax.plot(range(r_max-1), corr_coeff_mean - corr_coeff_std, dashes=[4,2],color='red', label='neg 1 std')
line3, = ax.plot(range(r_max-1), corr_coeff_mean + corr_coeff_std, dashes=[4,2],color='red', label='plus 1 std')

ax.legend()
plt.show()
