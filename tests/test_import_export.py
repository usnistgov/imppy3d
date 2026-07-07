"""
Written by Newell Moser
NRC Postdoctoral Fellow
National Institute of Standards and Technology
December 2, 2024

Test import/export of image stacks.
"""

import os, pathlib, glob
import numpy as np
from skimage.util import img_as_ubyte, img_as_float32
import imppy3d.import_export as imex


def del_all_img_files(dir_path='./'):
    file_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".pgm")
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


def io_img_stack_8bit(dir_path='./'):
    test_name = "\nTEST: IMPORTING/EXPORTING UNSIGNED 8-BIT IMAGE STACKS..."
    print(test_name)

    os.makedirs(dir_path, exist_ok=True)

    # Create an empty image stack containing 4 images
    imgs_out_8bit = np.zeros((4,128,128), dtype=np.uint8)
    imgs_out_8bit[0,0,0] = 250
    imgs_out_substr = "test_imgs_8bit.tif"

    try:
        imex.save_image_seq(imgs_out_8bit, dir_path, imgs_out_substr)
    except:
        print(f"\nERROR: Failed to save 8-bit image stack.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    try:
        imgs_in_8bit, imgs_names = imex.load_image_seq(dir_path, 
            file_name_in=imgs_out_substr[0:-4], indices_in=(4,),
            img_bitdepth_in='uint8')
    except:
        print(f"\nERROR: Failed to load 8-bit image stack.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    if imgs_out_8bit.dtype != imgs_in_8bit.dtype:
        print(f"\nERROR: The data type of the imported image stack is not the")
        print(f"same as the exported image stack.")
        return [1, test_name + " ERROR"]

    if imgs_out_8bit.shape != imgs_in_8bit.shape:
        print(f"\nERROR: The shape of the imported image stack is not the")
        print(f"same as the exported image stack.")
        return [1, test_name + " ERROR"]

    TOL = 1.0E-2
    imgs_diff = np.absolute(img_as_float32(imgs_out_8bit) - img_as_float32(imgs_in_8bit))
    max_diff = np.amax(imgs_diff)

    if max_diff >= TOL:
        print(f"\nERROR: Imported image stack differs from the exported image"\
            " stack.")
        return [1, test_name + " ERROR"]

    del_all_img_files(dir_path)
    return [0, test_name + " SUCCESS"] # (False) No errors


def io_img_stack_16bit(dir_path='./'):
    test_name = "\nTEST: IMPORTING/EXPORTING UNSIGNED 16-BIT IMAGE STACKS..."
    print(test_name)

    os.makedirs(dir_path, exist_ok=True)

    # Create an empty image stack containing 4 images
    imgs_out_16bit = np.zeros((4,128,128), dtype=np.uint16)
    imgs_out_16bit[0,0,0] = 65000
    imgs_out_substr = "test_imgs_16bit.tif"

    try:
        imex.save_image_seq(imgs_out_16bit, dir_path, imgs_out_substr)
    except:
        print(f"\nERROR: Failed to save 16-bit image stack.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    try:
        imgs_in_16bit, imgs_names = imex.load_image_seq(dir_path, 
            file_name_in=imgs_out_substr[0:-4], indices_in=(4,), 
            img_bitdepth_in='uint16')
    except:
        print(f"\nERROR: Failed to load 16-bit image stack.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    if imgs_out_16bit.dtype != imgs_in_16bit.dtype:
        print(f"\nERROR: The data type of the imported image stack is not the")
        print(f"same as the exported image stack.")
        return [1, test_name + " ERROR"]

    if imgs_out_16bit.shape != imgs_in_16bit.shape:
        print(f"\nERROR: The shape of the imported image stack is not the")
        print(f"same as the exported image stack.")
        return [1, test_name + " ERROR"]

    TOL = 1.0E-2
    imgs_diff = np.absolute(img_as_float32(imgs_out_16bit) - img_as_float32(imgs_in_16bit))
    max_diff = np.amax(imgs_diff)

    if max_diff >= TOL:
        print(f"\nERROR: Imported image stack differs from the exported image"\
            " stack.")
        return [1, test_name + " ERROR"]

    del_all_img_files(dir_path)
    return [0, test_name + " SUCCESS"] # (False) No errors


def io_multipage_tiff_stack(dir_path='./'):
    test_name = "\nTEST: IMPORTING/EXPORTING MULTI-PAGE TIFF STACKS..."
    print(test_name)

    os.makedirs(dir_path, exist_ok=True)

    # Create an empty image stack containing 4 images
    imgs_out_8bit = np.zeros((8,128,128), dtype=np.uint8)
    imgs_out_8bit[0,0,0] = 250
    imgs_path = os.path.normcase(dir_path + "test_imgs_8bit.tiff")

    try:
        imex.save_multipage_image(imgs_out_8bit, imgs_path)
    except:
        print(f"\nERROR: Failed to save 8-bit multi-page tiff stack.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    try:
        imgs_in_8bit, imgs_names = imex.load_multipage_image(imgs_path, 
            indices_in=(8,), img_bitdepth_in='uint8')
    except:
        print(f"\nERROR: Failed to load 8-bit multi-page tiff stack.")
        return [1, test_name + " ERROR"] # (True) Errors occurred

    if imgs_out_8bit.dtype != imgs_in_8bit.dtype:
        print(f"\nERROR: The data type of the imported image stack is not the")
        print(f"same as the exported image stack.")
        return [1, test_name + " ERROR"]

    if imgs_out_8bit.shape != imgs_in_8bit.shape:
        print(f"\nERROR: The shape of the imported image stack is not the")
        print(f"same as the exported image stack.")
        return [1, test_name + " ERROR"]

    TOL = 1.0E-2
    imgs_diff = np.absolute(img_as_float32(imgs_out_8bit) - img_as_float32(imgs_in_8bit))
    max_diff = np.amax(imgs_diff)

    if max_diff >= TOL:
        print(f"\nERROR: Imported image stack differs from the exported image"\
            " stack.")
        return [1, test_name + " ERROR"]

    del_all_img_files(dir_path)
    return [0, test_name + " SUCCESS"] # (False) No errors


def natural_sort_img_seq(dir_path='./'):
    test_name = "\nTEST: NATURAL (NUMERIC) ORDERING OF IMAGE SEQUENCES..."
    print(test_name)

    os.makedirs(dir_path, exist_ok=True)

    # Write image files with UN-zero-padded numeric names (natsort_1.tif ..
    # natsort_12.tif). save_image_seq() zero-pads its output, so it cannot
    # reproduce this case -- we write the names directly. Each frame encodes its
    # own 1-based index in pixel [0, 0] so the load order can be verified.
    # A plain string sort would order these as 1, 10, 11, 12, 2, 3, ... .
    n_imgs = 12
    for ii in range(1, n_imgs + 1):
        img_ii = np.zeros((16, 16), dtype=np.uint8)
        img_ii[0, 0] = ii
        img_path = os.path.normcase(dir_path + f"natsort_{ii}.tif")
        try:
            imex.save_image(img_ii, img_path, quiet_in=True)
        except:
            print(f"\nERROR: Failed to save natural-sort test image {ii}.")
            return [1, test_name + " ERROR"]

    try:
        imgs_in, imgs_names = imex.load_image_seq(dir_path,
            file_name_in="natsort_", img_bitdepth_in='uint8')
    except:
        print(f"\nERROR: Failed to load the natural-sort test image stack.")
        return [1, test_name + " ERROR"]

    if imgs_in is None or imgs_in.shape[0] != n_imgs:
        print(f"\nERROR: Expected {n_imgs} images, got "
              f"{None if imgs_in is None else imgs_in.shape[0]}.")
        del_all_img_files(dir_path)
        return [1, test_name + " ERROR"]

    # Frame k (0-based) must be source image k+1 -> pixel [0, 0] == k + 1.
    loaded_order = [int(imgs_in[k, 0, 0]) for k in range(n_imgs)]
    if loaded_order != list(range(1, n_imgs + 1)):
        print(f"\nERROR: Image stack loaded out of order. Expected "
              f"{list(range(1, n_imgs + 1))}, got {loaded_order}.")
        del_all_img_files(dir_path)
        return [1, test_name + " ERROR"]

    del_all_img_files(dir_path)
    return [0, test_name + " SUCCESS"] # (False) No errors


if __name__ == '__main__':

    temp_dir_path = "./local_swap/"
    flag_000, msg_000 = io_img_stack_8bit(temp_dir_path)
    flag_001, msg_001 = io_img_stack_16bit(temp_dir_path)
    flag_002, msg_002 = io_multipage_tiff_stack(temp_dir_path)
    flag_003, msg_003 = natural_sort_img_seq(temp_dir_path)

    print(f"\n\n\n---------- SUMMARY ----------")
    print(msg_000)
    print(msg_001)
    print(msg_002)
    print(msg_003)
    print("\n")