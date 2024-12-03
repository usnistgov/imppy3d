"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
December 2, 2024

Tests for calcualating the minimum rotated bounding box of a point cloud.
"""

import os
import numpy as np
import imppy3d.ski_driver_functions as sdrv


def make_dummy_img_stack(vec_dir='z'):
    imgs = np.zeros((129,129,129), dtype=np.uint8)

    rad_maj = 31
    rad_min = 15
    mid_i = int(np.floor(129/2.0))
    
    if vec_dir=='z': # Along image coordinates
        imgs[mid_i-rad_maj:mid_i+rad_maj+1, mid_i-rad_min:mid_i+rad_min+1,
            mid_i-rad_min:mid_i+rad_min+1] = 255

    elif vec_dir=='y': # Along row coordinates
        imgs[mid_i-rad_min:mid_i+rad_min+1, mid_i-rad_maj:mid_i+rad_maj+1, 
            mid_i-rad_min:mid_i+rad_min+1] = 255

    elif vec_dir=='x': # Along column coordinates
        imgs[mid_i-rad_min:mid_i+rad_min+1, mid_i-rad_min:mid_i+rad_min+1,
            mid_i-rad_maj:mid_i+rad_maj+1] = 255

    return imgs


def make_dummy_img():
    img_width = 129

    img = np.zeros((img_width,img_width), dtype=np.uint8)
    mid_i = int(np.floor(img_width/2.0))

    rad_maj = 31
    rad_min = 31
    img[mid_i-rad_maj:mid_i+rad_maj+1, mid_i-rad_min:mid_i+rad_min+1] = 127

    rad_maj = 15
    rad_min = 15
    img[mid_i-rad_maj:mid_i+rad_maj+1, mid_i-rad_min:mid_i+rad_min+1] = 255

    return img


def run_scikit_filters():
    # These are wrappers to the Scikit-Image library. Hence, no need to test
    # the actual accuracy of the filters since that is a verification already
    # handled by the Scikit-Image library.
    test_name = "\nTEST: EXECUTING SCIKIT-IMAGE WRAPPERS..."
    print(test_name)

    img_gray = make_dummy_img()
    img_bin = img_gray.copy()
    img_bin[img_gray > 0] = 255

    try:
        sharp_radius = 2    # int
        sharp_amount = 0.2  # float
        params = ["unsharp_mask", int(sharp_radius), sharp_amount] 
        img_temp = sdrv.apply_driver_sharpen(img_gray, params, quiet_in=True)
    except:
        print(f"\nERROR: Failed to apply unsharp filter.")
        return [1, test_name + " ERROR"]

    try:
        global_thresh = 200
        params = ["global_threshold", global_thresh]
        img_temp = sdrv.apply_driver_thresholding(img_gray, params, quiet_in=True)
    except:
        print(f"\nERROR: Failed to apply global thresholding filter.")
        return [1, test_name + " ERROR"]

    try:
        h_out = 0.01
        patch_size_out = 3
        patch_dist_out = 5
        params = ["nl_means", h_out, patch_size_out, patch_dist_out]
        img_temp = sdrv.apply_driver_denoise(img_gray, params, quiet_in=True)
    except:
        print(f"\nERROR: Failed to apply nonlocal-means denoising filter.")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors

    try:
        operation_type_out = 2
        footprint_type_out = 0
        n_radius_out = 1
        params = [operation_type_out, footprint_type_out, n_radius_out]
        img_temp = sdrv.apply_driver_morph(img_bin, params, quiet_in=True)
    except:
        print(f"\nERROR: Failed to apply binary morphological filter.")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors
    

if __name__ == '__main__':

    flag_400, msg_400 = run_scikit_filters()

    print(f"\n\n\n---------- SUMMARY ----------")
    print(msg_400)
    print("\n")