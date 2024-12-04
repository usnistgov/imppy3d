"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
December 2, 2024

Test creation of (3D) VTK models, which can be used by ParaView.
"""

import os, pathlib, glob
import numpy as np
from skimage import draw
from skimage.util import img_as_ubyte, img_as_float32
import imppy3d.vtk_api as vapi


def del_all_vtk_files(dir_path='./'):
    file_ext = (".vtk", ".vtu")
    img_names = []

    # All files in directory
    files = glob.glob(dir_path + '*') 
    if files:
        for cur_name in files:
            # Makes forward slashes into backward slashes for W10, and
            # also makes the directory string all lowercase
            cur_name = os.path.normcase(cur_name)
            # Only keep image files
            if (cur_name.lower()).endswith(file_ext):
                img_names.append(cur_name)

    # If empty, nothing to delete
    if not img_names:
        return 

    img_names = list(dict.fromkeys(img_names)) # Removes duplicates
    img_names.sort() # Ascending order based on the string file names

    # Deletes all of the image files found in this directory
    for cur_name in img_names:
        file_to_rem = pathlib.Path(cur_name)
        file_to_rem.unlink()


def make_dummy_ellipsoid_img_stack():
    # Draw an ellipsoid
    # ********** USER INPUTS ********** 
    # x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    # Define the semi-major and semi-minor (radii) axes.
    ellip_a = 15 # Aligned to Z-axis (Image Index)
    ellip_b = 31 # Aligned to Y-axis (Row Index)
    ellip_c = 63 # Aligned to X-axis (Column Index)
    # ********************************* 

    img_ellip = draw.ellipsoid(ellip_a, ellip_b, ellip_c)
    img_ellip = img_as_ubyte(img_ellip)

    return img_ellip


def make_dummy_box_img_stack(vec_dir='z'):
    imgs = np.zeros((129,129,129), dtype=np.uint8)

    rad_maj = 31
    rad_min = 31
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


def create_vtk_polydata_surf():
    # Creates a surface from a voxel model of an ellipsoid
    test_name = "\nTEST: CREATE VTK POLYDATA SURFACE USING MARCHING CUBES..."
    print(test_name)

    imgs_b = make_dummy_box_img_stack()
    n_voxels = 63*63*63

    try:
        verts, faces, normals, vals = vapi.convert_voxels_to_surface(imgs_b)
    except:
        print(f"\nERROR: Failed to apply marching cubes to the voxel model.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    try:
        vtk_obj = vapi.make_vtk_surf_mesh(verts, faces, vals, smth_iter=0)
    except:
        print(f"\nERROR: Failed to create a VTK polydata object.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    TOL = 0.05*n_voxels
    vol_diff = np.absolute(vtk_obj.volume - n_voxels)

    if vol_diff >= TOL:
        print(f"\nERROR: Created surface model does not contain the correct"\
            " volume.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    return [0, test_name + " SUCCESS"] # No errors


def create_vtk_unstructured_grid():
    # Creates a voxel model of an ellipsoid
    test_name = "\nTEST: CREATE VTK UNSTRUCTURED VOXEL MODEL (C EXTENSION)..."
    print(test_name)

    imgs_b = make_dummy_box_img_stack() # 63 x 63 x 63 cuboid
    n_voxels = 63*63*63

    try:
        vtk_obj = vapi.make_vtk_unstruct_grid(imgs_b, quiet_in=True)
    except:
        print(f"\nERROR: Failed to create VTK voxel model.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    # Check that the correct number of voxels were created
    if vtk_obj.n_cells != n_voxels:
        print(f"\nERROR: Created voxel model does not contain the correct"\
            " number of cells.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    return [0, test_name + " SUCCESS"] # No errors


def create_vtk_uniform_grid():
    # Creates a VTK image data model of an ellipsoid
    test_name = "\nTEST: CREATE VTK IMAGE DATA MODEL..."
    print(test_name)

    imgs_b = make_dummy_box_img_stack()
    n_voxels = 63*63*63

    try:
        vtk_obj = vapi.make_vtk_uniform_grid(imgs_b)
    except:
        print(f"\nERROR: Failed to create VTK image data model.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    vtk_arr = vtk_obj.cell_data['values']
    vtk_arr2 = vtk_arr[vtk_arr > 128]

    if vtk_arr2.size != n_voxels:
        print(f"\nERROR: Created uniform grid does not contain the correct"\
            " number of cells.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    return [0, test_name + " SUCCESS"] # No errors


if __name__ == '__main__':

    flag_100, msg_100 = create_vtk_unstructured_grid()
    flag_101, msg_101 = create_vtk_polydata_surf()
    flag_102, msg_102 = create_vtk_uniform_grid()

    print(f"\n\n\n---------- SUMMARY ----------")
    print(msg_100)
    print(msg_101)
    print(msg_102)
    print("\n")