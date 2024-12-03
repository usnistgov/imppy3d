"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
December 2, 2024

Test 3D rotations of image stacks.
"""

import os, pathlib, glob
import numpy as np
from skimage.util import img_as_ubyte, img_as_float32
import imppy3d.volume_image_processing as vol

#import imppy3d.vtk_api as vapi
#import imppy3d.import_export as imex
#import pyvista as pv


def make_dummy_img_stack(vec_dir='z'):
    imgs = np.zeros((129,129,129), dtype=np.uint8)

    rad_maj = 31
    rad_min = 1
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


def rotate_about_x():
    test_name = "\nTEST: ROTATING BINARY IMAGE STACK ABOUT X-AXIS..."
    print(test_name)

    imgs_src = make_dummy_img_stack('z')
    imgs_ref = make_dummy_img_stack('y')

    x_dir = np.array([0,0,1])
    rot_axis = x_dir
    rot_angle = 90*np.pi/180.0

    try:
        imgs_calc = vol.img_rotate_3D(imgs_src, rot_axis, rot_angle, 
            binarize=True)
    except:
        print(f"\nERROR: Failed to rotate the image stack.")
        return [1, test_name + " ERROR"]

    TOL = 1.0E-2
    imgs_diff = np.absolute(img_as_float32(imgs_calc) - img_as_float32(imgs_ref))
    max_diff = np.amax(imgs_diff)

    if max_diff >= TOL:
        print(f"\nERROR: Rotated image stack does not match reference image stack")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


def rotate_about_y():
    test_name = "\nTEST: ROTATING BINARY IMAGE STACK ABOUT Y-AXIS..."
    print(test_name)

    imgs_src = make_dummy_img_stack('z')
    imgs_ref = make_dummy_img_stack('x')

    y_dir = np.array([0,1,0])
    rot_axis = y_dir
    rot_angle = 90*np.pi/180.0

    try:
        imgs_calc = vol.img_rotate_3D(imgs_src, rot_axis, rot_angle, 
            binarize=True)
    except:
        print(f"\nERROR: Failed to rotate the image stack.")
        return [1, test_name + " ERROR"]

    TOL = 1.0E-2
    imgs_diff = np.absolute(img_as_float32(imgs_calc) - img_as_float32(imgs_ref))
    max_diff = np.amax(imgs_diff)

    if max_diff >= TOL:
        print(f"\nERROR: Rotated image stack does not match reference image stack")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


def rotate_about_z():
    test_name = "\nTEST: ROTATING BINARY IMAGE STACK ABOUT Z-AXIS..."
    print(test_name)

    imgs_src = make_dummy_img_stack('x')
    imgs_ref = make_dummy_img_stack('y')

    z_dir = np.array([1,0,0])
    rot_axis = z_dir
    rot_angle = 90*np.pi/180.0

    try:
        imgs_calc = vol.img_rotate_3D(imgs_src, rot_axis, rot_angle, 
            binarize=True)
    except:
        print(f"\nERROR: Failed to rotate the image stack.")
        return [1, test_name + " ERROR"]

    TOL = 1.0E-2
    imgs_diff = np.absolute(img_as_float32(imgs_calc) - img_as_float32(imgs_ref))
    max_diff = np.amax(imgs_diff)

    if max_diff >= TOL:
        print(f"\nERROR: Rotated image stack does not match reference image stack")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


def rotate_about_xyz():

    test_name = "\nTEST: ARBITRARY ROTATION OF BINARY IMAGE STACK..."
    print(test_name)

    imgs_src = make_dummy_img_stack('z')

    desired_vec3 = np.array([1.0,1.0,1.0])
    z_vec3 =  np.array([1.0,0.0,0.0])

    desired_vec3_mag = np.sqrt( np.power(desired_vec3[0], 2) + 
                            np.power(desired_vec3[1], 2) +
                            np.power(desired_vec3[2], 2) )
    desired_vec3 = desired_vec3/desired_vec3_mag

    rot_axis = np.cross(desired_vec3, z_vec3)
    rot_axis_mag = np.sqrt( np.power(rot_axis[0], 2) + 
                            np.power(rot_axis[1], 2) +
                            np.power(rot_axis[2], 2) )
    rot_axis = rot_axis/rot_axis_mag

    rot_angle = np.arccos(np.clip(np.dot(z_vec3, desired_vec3), -1.0, 1.0))

    # Apply the affine transformation
    print("\nPerforming rotation affine transformation...")
    print(f"    Rotation Axis (img, row, col): {rot_axis}")
    print(f"    Rotation Magnitude (CCW): {rot_angle*180.0/np.pi} (degrees)")

    try:
        imgs_calc = vol.img_rotate_3D(imgs_src, rot_axis, rot_angle, 
            binarize=True)
    except:
        print(f"\nERROR: Failed to rotate the image stack.")
        return [1, test_name + " ERROR"]

    xyz_pnts_tup = np.nonzero(imgs_calc)
    img_coords = np.asarray(xyz_pnts_tup[0])
    row_coords = np.asarray(xyz_pnts_tup[1])
    col_coords = np.asarray(xyz_pnts_tup[2])
    xyz_pnts_arr = np.column_stack((img_coords, row_coords, col_coords))

    # Fit a line to the newly rotated points
    try:
        mean_pnt, eig_vec1 = vol.line_fit_3d(xyz_pnts_arr)
    except:
        print(f"\nERROR: Could not fit a 3D line to the rotated "\
            "image stack.")
        return [1, test_name + " ERROR"]

    # Check within a tolerance of 3 degree
    angle_TOL = 3.0*np.pi/180.0 # Radians

    # The fitted line should be in the [1, 1, 1] direction
    paral_bools = np.isclose(desired_vec3, eig_vec1, rtol=0.0, atol=angle_TOL)
    if not all(paral_bools):
        print(f"\nERROR: Rotated image stack is misaligned.")
        return [1, test_name + " ERROR"]

    return [0, test_name + " SUCCESS"] # (False) No errors


if __name__ == '__main__':

    flag_200, msg_200 = rotate_about_x()
    flag_201, msg_201 = rotate_about_y()
    flag_202, msg_202 = rotate_about_z()
    flag_203, msg_203 = rotate_about_xyz()

    print(f"\n\n\n---------- SUMMARY ----------")
    print(msg_200)
    print(msg_201)
    print(msg_202)
    print(msg_203)
    print("\n")