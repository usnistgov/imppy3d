"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
December 2, 2024

Use this script to unit test the core features of IMPPY3D, as well as
verify your installation of IMPPY3D.
"""

import os
import test_import_export as t_io
import test_vtk_models as t_vtk
import test_rotations as t_rot
import test_bbox as t_box
import test_scikit_filters as t_sfilt


# Need a temporary directory to read/write to for unit testing. 
# WARNING: All image files in this directory will be deleted after testing is
# complete. Please ensure that this is an empty directory before testing.
temp_dir_path = "./local_swap/"

print(f"\n\nSTARTING ALL UNIT TESTS...")


# ---------------- IMPORT/EXPORT IMAGE STACKS ---------------- 
os.makedirs(temp_dir_path, exist_ok=True)
flag_000, msg_000 = t_io.io_img_stack_8bit(temp_dir_path)
flag_001, msg_001 = t_io.io_img_stack_16bit(temp_dir_path)
flag_002, msg_002 = t_io.io_multipage_tiff_stack(temp_dir_path)


# ---------------- CREATION OF VTK MODELS ---------------- 
flag_100, msg_100 = t_vtk.create_vtk_unstructured_grid()
flag_101, msg_101 = t_vtk.create_vtk_polydata_surf()
flag_102, msg_102 = t_vtk.create_vtk_uniform_grid()


# ---------------- ROTATING/RE-SLICING IMAGE STACKS ---------------- 
flag_200, msg_200 = t_rot.rotate_about_x()
flag_201, msg_201 = t_rot.rotate_about_y()
flag_202, msg_202 = t_rot.rotate_about_z()
flag_203, msg_203 = t_rot.rotate_about_xyz()


# ---------------- BOUNDING BOX ROUTINES ---------------- 
flag_300, msg_300 = t_box.bbox_svd()
flag_301, msg_301 = t_box.bbox_exhaustive()
flag_302, msg_302 = t_box.bbox_LWT()


# ---------------- SCIKIT WRAPPERS ----------------
flag_400, msg_400 = t_sfilt.run_scikit_filters()


# ---------------- SUMMARIZE TEST RESULTS ---------------- 

print(f"\n\n\n---------- SUMMARY OF ALL UNIT TESTS ----------")
print(msg_000)
print(msg_001)
print(msg_002)

print(msg_100)
print(msg_101)
print(msg_102)

print(msg_200)
print(msg_201)
print(msg_202)
print(msg_203)

print(msg_300)
print(msg_301)
print(msg_302)

print(msg_400)
print("\n")