# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Import imppy3d
import imppy3d.import_export as imex
import imppy3d.cv_processing_wrappers as wrap
import imppy3d.cv_interactive_processing as intprc


eps_float = 1e-12

"""
Implementation of the autocorrelation function based anisotropy analysis of
doi.org/10.1016/j.mtla.2021.101112

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


A_r = np.real_if_close((np.fft.ifftn(np.fft.fftn(imgs)*np.conjugate(np.fft.fftn(imgs)))/N -
                        np.power(I_bar,2))/np.power(sigma_I,2))

# vol.plot_3d_slices(A_r)
a_1 = np.zeros((50,1),dtype=float)
a_2 = np.zeros((50,1),dtype=float)
theta_1 = np.zeros((50,1),dtype=float)
A_0 = np.linspace(0.2,0.5,50) # autocorr threshold
cnt = 0
for cnt in range(50):
    A_0_cur = A_0[cnt]
    M_acf_11,M_acf_22,M_acf_33,M_acf_12,M_acf_13,M_acf_23 = 0,0,0,0,0,0
    for z in range(imgs.shape[0]):
        for y in range(imgs.shape[1]):
            for x in range(imgs.shape[2]):
                if A_r[z,y,x] > A_0_cur:
                    M_acf_11 = M_acf_11 + A_r[z,y,x]*(y**2 + z**2)
                    M_acf_22 = M_acf_22 + A_r[z,y,x]*(x**2 + z**2)
                    M_acf_33 = M_acf_33 + A_r[z,y,x]*(x**2 + y**2)
                    M_acf_12 = M_acf_12 + A_r[z,y,x]*(x*y)
                    M_acf_13 = M_acf_13 + A_r[z,y,x]*(x*z)
                    M_acf_23 = M_acf_23 + A_r[z,y,x]*(y*z)

    M_acf = np.zeros((3,3), dtype=float)
    M_acf[0,0] = M_acf_11
    M_acf[1,1] = M_acf_22
    M_acf[2,2] = M_acf_33
    M_acf[0,1] = M_acf_12
    M_acf[0,2] = M_acf_13
    M_acf[1,2] = M_acf_23
    M_acf[1,0] = M_acf[0,1]
    M_acf[2,0] = M_acf[0,2]
    M_acf[2,1] = M_acf[1,2]

    lambda_acf,v_acf = np.linalg.eig(M_acf)
    lambda_acf_sortidx = np.argsort(lambda_acf)

    a_1[cnt] = 1 - lambda_acf[lambda_acf_sortidx[0]]/lambda_acf[lambda_acf_sortidx[2]]
    a_2[cnt] = 1 - np.max((lambda_acf[lambda_acf_sortidx[0]]/lambda_acf[lambda_acf_sortidx[1]],
                      lambda_acf[lambda_acf_sortidx[1]]/lambda_acf[lambda_acf_sortidx[2]]))

    theta_1[cnt] = np.arccos(abs(np.dot([1,0,0],v_acf[:,lambda_acf_sortidx[0]])))




fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(A_0, a_1, label='a_1')
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line2, = ax.plot(A_0, a_2, dashes=[6, 2], label='a_2')
line3, = ax.plot(A_0, theta_1, dashes=[10, 1], label='theta_1')

ax.legend()
plt.show()
