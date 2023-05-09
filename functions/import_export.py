import numpy as np
import cv2 as cv
from skimage.util import img_as_ubyte, img_as_uint
from skimage import io
import glob
import os.path


def load_image(path_in, img_bitdepth='uint8', quiet_in=False):
    """ 
    Loads an image file using OpenCV routines. Color images will be
    converted to grayscale, and uint16 images will linearly scaled to
    uint8. Only uint16 and uint8 image data types are supported. 
    
    ---- INPUT ARGUMENTS ----
    path_in: String that contains the file path (and name and extension)
        of the image being loaded.

    img_bitdepth [string]: Bit depth for the reader to use. Either
        of unsigned 8-bit (uint8) or unsigned 16-bit (uint16) are
        nominally supported. 8-bit is the default (and what the rest of
        the codes currently expect).
    
    quiet_in: An optional boolean that is by default set to False. Set 
        to True to prevent outputting any messages.

    ---- RETURNED ---- 
    [img, img_prop]: List of length two. If the image failed to load 
        correctly, this list will contain just None objects. In more 
        detail, these returned items are described below.

        img: OpenCV's Mat class that describes the image data via a
            dense n-dimensional array of numbers. Conventional numpy 
            operations can be safely applied to img.

        img_prop: List of length three that describes the image
            properties. More details given below.

            img_prop[0]: Total number of pixels in the image.
            img_prop[1]: A tuple of length two that provides the number 
                of rows and columns of pixels. 
            img_prop[2]: A string that confirms the image data type. 
                Since load_image(...) converts all images to 'uint8', 
                this will  always be equal to 'uint8'.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings may be
    printed to standard output.
    """

    # ---- Start Local Copies ----
    # Forcing Python to create a new string variable in memory
    # File path to the image file
    file_path = (path_in + '.')[:-1]

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    file_path = os.path.normcase(file_path)
    
    quiet = quiet_in
    # ---- End Local Copies ----

    # Load a image in possibly uint16 or uint8 format (in grayscale).
    #img = cv.imread(file_path, cv.IMREAD_ANYDEPTH)
    # Note, OpenCV opens color images in a BGR convention, not RGB. To swap
    # the channels, which is necessary to plot color images in matlplotlib,
    # you can use the following function: cv.cvtColor(image,
    # cv.COLOR_RGB2BGR). However, this script converts all color images into
    # grayscale, so this does not actually apply here.

    # OpenCV could not handle image rotations correctly for 16-bit images.
    # Using SciKit-Image instead.
    img = io.imread(file_path, as_gray=True)

    # Catch the case that the file path was incorrect, or the image is
    # corrupted
    if img is None:
        return [None, None]
    
    # Total number of pixels
    image_size = img.size
    
    # If the image is color, img.shape returns a tuple containing the number
    # of  rows, columns, and channels. If the image is grayscale, this returns
    # a tuple containing the number of rows and columns (no color channels)
    image_shape = img.shape
    
    # Returns the image data type (e.g., uint8)
    image_data_type = img.dtype

    # If uint16, convert to uint8 image data type
    if img_bitdepth == "uint8":
        # Use SciKit-Image to be more robust; is should handle n-d arrays
        img = img_as_ubyte(img)
        
    elif img_bitdepth == "uint16":
        # Use SciKit-Image to be more robust
        img = img_as_uint(img)
        
    else:
        if not quiet:
            print("\nWARNING: Unsupported bit depth detected. Defaulting to 8-bit")
        img = img_as_ubyte(img)
    
    # Update image parameters in case any changes were made
    image_size = img.size
    image_shape = img.shape
    image_data_type = img.dtype
    img_prop = [image_size, image_shape, image_data_type]

    if not quiet:
        print(f"\nSuccessfully imported image: {file_path}")

    # Return the image objects and image properties
    return [img, img_prop]


def load_multipage_image(path_in, indices_in=[], bigtiff=False,\
                         img_bitdepth_in="uint8", flipz=False,
                         quiet_in=False):
    """
    Read a multipage image (e.g. monolithic tiff stack) from file using
    the skimage library.
    
    ---- INPUT ARGUMENTS ----
    path_in [string]: Path to tiff file to load
    
    indices_in: An optional tuple of either length 1 or length 2. If 
        length 1, then it should contain a positive integer 
        corresponding to the number of images to be kept centered about
        the middle image of the image sequence. If length 2, then the
        first and second elements will be used directly to slice the
        list corresponding to the image sequence. For example, (0, 100)
        would keep the first 100 images. Fair warning, if you provide
        invalid indices, you will get an error.

    bigtiff [bool]: If False, the OpenCV multi-page TIFF function will
        be used to import the image. This is fine for standard TIFF 
        files. However, if the TIFF file is saved with a less 
        conventional format/header, such as for BigTIFF or ImageJ 
        Hyperstack formats, then this should be set to True in order to
        use a different importer. Note, these alternative TIFF formats
        should be used anytime the TIFF file is larger than 4 GB. When
        set to True, the tifffile library will be used via a plugin 
        within the Sci-Kit Image library.
    
    img_bitdepth_in [string]: Bit depth for the reader to use. Either
        of unsigned 8-bit (uint8) or unsigned 16-bit (uint16) are nominally
        supported. 8-bit is the default (and what the rest of the codes
        currently expect).

    flipz [bool]: When True, the images are reversed in the image stack,
        which can be thought of as flipping the image stack along the Z-
        direction. This occurs after any images have been removed from
        'indices_in' above. When False, the original order of the image
        stack is maintained. If the first image corresponds to the 
        bottom of a part, then in general, flipz should be True so as
        to ensure a right-handed coordinate system. Positive X will be
        along ascending column indices, positive Y will be along 
        ascending row indices, and positive Z will be along DESCENDING
        image indices.
        
    quiet_in [bool]: Set to true to suppress any output dialog
    
    --- RETURNED ---
    [img, img_prop]

    img: OpenCV Mats n-d array containing images (can be operated on
        like a numpy array). Its shape is [num_images, num_rows, 
        num_cols].

    img_prop: List of length three that describes the image
            properties.
        img_prop[0]: Total number of pixels in the image.
        img_prop[1]: A tuple of length three that provides the number 
            of rows, columns, and pages of pixels. 
        img_prop[2]: A string that confirms the image data type. 
            Since load_image(...) converts all images to 'uint8', 
            this will  always be equal to 'uint8'.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings may be
    printed to standard output.
    
    Warning: this loads the full image stack and then slices out the
        desired parts defined in indices_in. There maybe be a way to
        do this better ...
    
    """
    
    # ---- Start Local Copies ----
    # Forcing Python to create a new string variable in memory
    # File path to the image file
    file_path = (path_in + '.')[:-1]

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    file_path = os.path.normcase(file_path)
    
    quiet = quiet_in

    img_bitdepth = img_bitdepth_in.lower()

    indices_keep = list(indices_in) # Ex: (0, 100) keeps first 100 images

    # ---- End Local Copies ----


    if not quiet:
        print(f"\nImporting multi-page TIFF stack:\n   {file_path}")

    imgs = io.imread(fname=file_path, plugin='tifffile')

    # Catch the case that the file path was incorrect, or the image is
    # corrupted
    if (len(imgs) == 0) and (not quiet):
        print(f"\nERROR: Image not loaded from filepath:\n  {file_path}\n")
        # Use "raise Exception()" here too?
        return [None, None]

    imgs = np.array(imgs) # Convert to Numpy array if not already
    
    # Contains the number of row, columns, and pages in a 3D image
    num_imgs = imgs.shape[0]
    num_rows = imgs.shape[1]
    num_cols = imgs.shape[2]
    image_shape = [num_imgs, num_rows, num_cols]

    # Returns the image data type (e.g., uint8)
    image_data_type = imgs[0].dtype
    
    # If uint16, convert to uint8 image data type
    if img_bitdepth == "uint8":
        # Use SciKit-Image to be more robust; is should handle n-d arrays
        imgs = img_as_ubyte(imgs)
        
    elif img_bitdepth == "uint16":
        # Use SciKit-Image to be more robust
        imgs = img_as_uint(imgs)
        
    else:
        if not quiet:
            print("\nWARNING: Unsupported bit depth detected. Defaulting to 8-bit")
        imgs = img_as_ubyte(imgs)
    
    if len(indices_in) == 0:
        # return full image
        indx_bounds = [0, num_imgs]
        if not quiet:
            print("\nReturning full image")

    elif len(indices_in) == 1:
        n_keep = indices_keep[0]
        cur_img_len = num_imgs # Total number of images

        if n_keep == 0: # Keeping zero images doesn't make sense
            n_keep = 1  # Will just keep the middle image then
        elif n_keep > cur_img_len: # Also doesn't make sense
            n_keep = cur_img_len   # So, keep them all

        # Index of middle element, rounds down if even
        if cur_img_len % 2 == 0: # If even
            mid_indx = int(cur_img_len/2) - 1
        else: # If odd
            mid_indx = int(cur_img_len/2)

        # Force the desired number of images to keep to be odd for now
        keep_even = False
        if n_keep % 2 == 0:
            keep_even = True
            n_keep -= 1

        # Number of images to keep above and below the middle index
        indx_rad = int((n_keep - 1)/2)

        # Index bounds for the images to be kept
        indx_bounds = [mid_indx - indx_rad, mid_indx + indx_rad + 1]

        # If originally even number of images to keep, correct it here
        if keep_even:
            indx_bounds[1] = indx_bounds[1] + 1

    # If there's two numbers, user gave the upper and lower numbers
    elif len(indices_in) == 2:
        indx_bounds = [indices_keep[0], indices_keep[1]]

    else:
        indx_bounds = [0, num_imgs]
        if not quiet: 
            print("\nWARNING: 'indices_in' is malformed. Returning full image")
    
    imgs = imgs[indx_bounds[0]:indx_bounds[1],:,:]

    # Reverse the order of the image stack
    if flipz:
        if not quiet:
            print(f"\nReversing the order of the image stack (i.e.," \
                + " flipping the Z-direction)...")
        imgs = np.flip(imgs, axis=0)

    # Update image parameters in case any changes were made
    image_size = imgs.size

    num_imgs = imgs.shape[0]
    num_rows = imgs.shape[1]
    num_cols = imgs.shape[2]
    image_shape = [num_imgs, num_rows, num_cols]

    image_data_type = imgs.dtype
    img_prop = [image_size, image_shape, image_data_type]

    if not quiet:
        print(f"\nSuccessfully imported image: {file_path}")

    # Return the image objects and image properties
    return [imgs, img_prop]


def read_pgm(filename_in, transpose=False, ASCII=True):
    """
    Read ASCII PGM file with magic number P2 using Ed's conventions.
    First line should contain "P2". Second line contains two integers:
    in the standard format, the first number should be the number of
    columns and second number should be the number of rows. Finally, the
    flattened array of pixel values are given -- one per line. Ed has
    them flattened in row-major order (C convention). The input, 
    "filename_in", is a string path to the ASCII image file that will be
     imported. The returned image is a Numpy 2D matrix with data type
     UINT8 (i.e., grayscale 0 - 255). Inputs and outputs are similar in
     style to the above function, load_image(...). Note, to ready 
    """

    # Force a local copy of this string
    file_name = (filename_in + '.')[:-1]
    file_name = os.path.normcase(file_name)

    if ASCII:
        with open(file_name) as f:
            lines = f.readlines()

        # Ignores commented lines
        for l in list(lines):
            if l[0] == '#':
                lines.remove(l)

        # Makes sure it is ASCII format (P2)
        assert lines[0].strip() == 'P2' 

        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c) for c in line.split()])
        
        if transpose:
            img1_shape = (data[0], data[1]) # (num of rows, num of cols)

        else:
            # Standard PGM file format
            img1_shape = (data[1], data[0]) # (num of rows, num of cols)

        img1_max = data[2]
        img1 = np.array(data[3:])
        img1 = np.reshape(img1, img1_shape, order='C')

        # Convert the data to a grayscale image from 0 - 255
        if np.amin(img1) < 0:
            img1 = img1 - np.amin(img1)

        if (img1_max > 255) or (img1_max != 255):
            ratio = img1_max/255.0
            img1 = (img1/ratio).astype(np.uint8)

        img1 = img_as_ubyte(img1)

    else: 
        # Import as binary format. This must be in the standard format.
        
        img1 = io.imread(file_name)
        img1 = img_as_ubyte(img1)

        img1_shape = img1.shape
    
    return (img1, img1_shape)

def load_image_seq(path_in, file_name_in='', img_bitdepth_in='uint8', 
    indices_in=(), flipz=False):
    """
    Loads an image sequence into memory in a single batch operation.
    This is done by repeatedly using load_image(...). Hence, the list
    of returned images will be grayscale and of type uint8. It is 
    assumed all of the images are in the same directory. Images are 
    sorted in ascending order (alphabetical).

    ---- INPUT ARGUMENTS ---- 
    path_in: String that contains the directory path.

    file_name_in: An optional string that is taken to be a substring of
        the image files to be imported. For  example, assume the
        following images are in a folder: img001.tif,  img002.tif,
        img003.tif, and other_img.png. Then, file_name_in can be set
        equal to "img" to ensure that only the .tif files are imported.
        By default, it is an empty string, in which case, every image
        in the directory will be imported.

    img_bitdepth_in [string]: Bit depth for the reader to use. Either
        of unsigned 8-bit (uint8) or unsigned 16-bit (uint16) are
        nominally supported. 8-bit is the default (and what the rest of
        the codes currently expect).

    indices_in: An optional tuple of either length 1 or length 2. If 
        length 1, then it should contain a positive integer 
        corresponding to the number of images to be kept centered about
        the middle image of the image sequence. If length 2, then the
        first and second elements will be used directly to slice the
        list corresponding to the image sequence. For example, (0, 100)
        would keep the first 100 images. Fair warning, if you provide
        invalid indices, you will get an error.

    flipz [bool]: When True, the images are reversed in the image stack,
        which can be thought of as flipping the image stack along the Z-
        direction. This occurs after any images have been removed from
        'indices_in' above. When False, the original order of the image
        stack is maintained. If the first image corresponds to the 
        bottom of a part, then in general, flipz should be True so as
        to ensure a right-handed coordinate system. Positive X will be
        along ascending column indices, positive Y will be along 
        ascending row indices, and positive Z will be along DESCENDING
        image indices.

    ---- RETURNED ---- 
    [[imgs], [img_names]]: List of length two. If the images failed to
        load correctly, this list will contain just None objects. See 
        below for more details.

        [imgs]: A 3D Numpy array that contains OpenCV's Mat class  
            objects, which are 2D Numpy arrays in of themselves. 
            Conventional Numpy operations can be safely applied to each
            entry of imgs. Images are converted to grayscale and uint8
            automatically. The shape of imgs will be (num_images,
            num_rows, num_cols).

        [img_names]: A list of strings that describe the file paths used
            to import every image. 

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings may be
    printed to standard output.
    """

    # ---- Start Local Copies ----
    # Forcing Python to create a new string variable in memory
    # Path to the directory containing images
    dir_path = (path_in + '.')[:-1]
    if not dir_path.endswith( ('\\', '/') ):
        dir_path = dir_path + '/'

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    dir_path = os.path.normcase(dir_path) 
    
    # Substring of the filenames to be imported
    file_name = (file_name_in + '.')[:-1] 
    file_name = os.path.normcase(file_name)

    # If length 1, then total number of images to keep about the center
    # If length 2, then lower and upper bounds of the images to be kept
    indices_keep = list(indices_in) # Ex: (0, 100) keeps first 100 images
    # ---- End Start Local Copies ----

    # Reduced list (actually tuple) of supported image file extensions
    file_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".bmp",
                ".dib", ".pbm", ".ppm", ".pgm", ".pnm")

    # Look for files that contain file_name in their names
    img_names = [] # A list of strings for the image file paths
    if file_name:
        files = glob.glob(dir_path + '*') # All files in directory
        if files:
            for cur_name in files:
                # Makes forward slashes into backward slashes for W10, and
                # also makes the directory string all lowercase
                cur_name = os.path.normcase(cur_name)
                if file_name in cur_name:
                    # Also ensure this is actually an image file
                    if (cur_name.lower()).endswith(file_ext):
                        img_names.append(cur_name)

    # Otherwise, import all common image file types (case insensitive)  
    else:
        files = glob.glob(dir_path + '*') # All files in directory
        if files:
            # Makes forward slashes into backward slashes for W10, and
            # also makes the directory string all lowercase
            cur_name = os.path.normcase(cur_name)
            for cur_name in files:
                # Only keep image files
                if (cur_name.lower()).endswith(file_ext):
                    img_names.append(cur_name)

    # Ensure that the list of names is not empty
    if not img_names:
        print(f"\nNo image files were found in {dir_path}")
        return [None, None]

    # The Glob module does not guarantee that the results will be sorted.
    img_names = list(dict.fromkeys(img_names)) # Removes duplicates
    img_names.sort() # Asecending order based on the string file names

    if indices_keep: # If not empty
        # If not already, force the data to be positive and of type integer
        for count, element in enumerate(indices_keep):
            indices_keep[count] = int(abs(element))

        if len(indices_keep) == 1: # Select a number of images from the center
            n_keep = indices_keep[0]
            cur_img_len = len(img_names) # Total number of images

            if n_keep == 0: # Keeping zero images doesn't make sense
                n_keep = 1  # Will just keep the middle image then
            elif n_keep > cur_img_len: # Also doesn't make sense
                n_keep = cur_img_len   # So, keep them all

            # Index of middle element, rounds down if even
            if cur_img_len % 2 == 0: # If even
                mid_indx = int(cur_img_len/2) - 1
            else: # If odd
                mid_indx = int(cur_img_len/2)

            # Force the desired number of images to keep to be odd for now
            keep_even = False
            if n_keep % 2 == 0:
                keep_even = True
                n_keep -= 1

            # Number of images to keep above and below the middle index
            indx_rad = int((n_keep - 1)/2)

            # Index bounds for the images to be kept
            indx_bounds = [mid_indx - indx_rad, mid_indx + indx_rad + 1]

            # If originally even number of images to keep, correct it here
            if keep_even:
                indx_bounds[1] = indx_bounds[1] + 1

            # Perform the slice operation
            img_names = img_names[indx_bounds[0]:indx_bounds[1]]

        elif len(indices_keep) == 2: # Expecting lower and upper indices
            ind_low = indices_keep[0]
            ind_upp = indices_keep[1]

            img_names = img_names[ind_low:ind_upp]

        else:
            print(f"\nInvalid indices to slice the array of images. Provided"\
                f" input was: {indices_keep}\nExpecting a tuple of length "\
                f"1 or 2. For example:\n    1) (200,) --> Keep 200 images "\
                f"about the center.\n    2) (0, 100) --> Keep the first 100 "\
                f"images")

    # Number of images in the finalized list of file paths
    num_imgs = len(img_names)

    # [img_props]: A list containing the image properties for each 
    # imported image. Each entry in img_props is a list in of 
    # itself. More details given below.
    # 
    # img_prop[m][0]: Total number of pixels in the m-th image.
    # img_prop[m][1]: A tuple of length two that provides the  
    #     number of rows and columns of pixels for the m-th image. 
    # img_prop[m][2]: A string that confirms the image data type 
    #     for m-th image. Since load_image(...) converts all 
    #     images to 'uint8', this will always be equal to 'uint8'.
    img_props = []
    imgs = [] # Will be converted to Numpy array later
    counter = 1
    print(f"\nBeginning to import {num_imgs} images...")
    for cur_name in img_names:
        # Now actually call OpenCV routines to import the images
        # Use the defined function from above to import images in gray scale
        # and as uint8. Also, use quiet-mode (no outputs to the terminal)
        cur_img, cur_img_prop = load_image(cur_name, 
            img_bitdepth=img_bitdepth_in, quiet_in=True)

        if cur_img is None:
            print(f"\nFailed to import image: {cur_name}")
            return [None, None]

        # Store the image objects and their properties in lists
        imgs.append(cur_img)
        img_props.append(cur_img_prop)

        # Write out updates for the user
        if (counter%20) == 0:
            print(f"    Currently imported {counter}/{num_imgs}...")
        
        counter += 1

    print(f"\nSuccessfully imported {num_imgs} images!")

    # Check to make sure all the images are the same shape
    first_img_shape = img_props[0][1] # Tuple of the image shape (rows, cols)
    cur_index = 0
    for cur_img_prop in img_props:
        cur_shape = cur_img_prop[1]

        if cur_shape != first_img_shape:
            print(f"\nWARNING! Not all imported images are of the same shape")
            print(f"\nFirst image shape: {first_img_shape}")
            print(f"\nImage shape of {img_names[cur_index]}: {cur_shape}")

        cur_index += 1

    # Convert from a Python list of Numpy arrays to a fully 3D Numpy array
    imgs = np.array(imgs)

    # Reverse the order of the image stack
    if flipz:
        print(f"\nReversing the order of the image stack (i.e.," \
            + " flipping the Z-direction)...")
        imgs = np.flip(imgs, axis=0)

    # Return the image objects, image properties, and string file names
    return [imgs, img_names]


def load_image_seq_ASCII(path_in, file_name_in='', indices_in=(), 
    transpose=False, ASCII=True):
    """
    Loads an image sequence into memory in a single batch operation.
    This is done by repeatedly using load_image(...). Hence, the list
    of returned images will be grayscale and of type uint8. It is 
    assumed all of the images are in the same directory. Images are 
    sorted in ascending order (alphabetical). This function is 
    specifically for Ed's ASCII PGM image file type.

    ---- INPUT ARGUMENTS ---- 
    path_in: String that contains the directory path.

    file_name_in: An optional string that is taken to be a substring of
        the image files to be imported. For  example, assume the
        following images are in a folder: img001.pgm,  img002.pgm,
        img003.pgm, and other_img.png. Then, file_name_in can be set
        equal to "img" to ensure that only the .pgm files are imported.
        By default, it is an empty string, in which case, every image
        in the directory will be imported.

    indices_in: An optional tuple of either length 1 or length 2. If 
        length 1, then it should contain a positive integer 
        corresponding to the number of images to be kept centered about
        the middle image of the image sequence. If length 2, then the
        first and second elements will be used directly to slice the
        list corresponding to the image sequence. For example, (0, 100)
        would keep the first 100 images. Fair warning, if you provide
        invalid indices, you will get an error.

    transpose: A boolean that determines how to interpret the shape of 
        each image. A standard PGM file format will include a line with
        two integers corresponding to the shape of the image being
        imported. If transpose=False, then the first integer is taken to
        be the number of columns and the second integer the number of 
        rows; this is the standard format. If transpose=True, then the
        first integer is taken to be the number of rows and the second
        integer the number of columns.

    ASCII: A boolean to determine if the images should be imported as
        ASCII files or as binary images. Set to True for ASCII files,
        and set to False for binary files.

    ---- RETURNED ---- 
    [[imgs], [img_names]]: List of length two. If the images failed to
        load correctly, this list will contain just None objects. See 
        below for more details.

        [imgs]: A 3D Numpy array that contains OpenCV's Mat class  
            objects, which are 2D Numpy arrays in of themselves. 
            Conventional Numpy operations can be safely applied to each
            entry of imgs. Images are converted to grayscale and uint8
            automatically. The shape of imgs will be (num_images,
            num_rows, num_cols).

        [img_names]: A list of strings that describe the file paths used
            to import every image. 

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Nothing is written to the 
    hard drive. This function is a read-only function. Strings may be
    printed to standard output.
    """

    # ---- Start Local Copies ----
    # Forcing Python to create a new string variable in memory
    # Path to the directory containing images
    dir_path = (path_in + '.')[:-1]
    if not dir_path.endswith( ('\\', '/') ):
        dir_path = dir_path + '/'

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    dir_path = os.path.normcase(dir_path)

    # Substring of the filenames to be imported
    file_name = (file_name_in + '.')[:-1] 
    file_name = os.path.normcase(file_name)

    # If length 1, then total number of images to keep about the center
    # If length 2, then lower and upper bounds of the images to be kept
    indices_keep = list(indices_in) # Ex: (0, 100) keeps first 100 images
    # ---- End Start Local Copies ----

    # Reduced list (actually tuple) of supported image file extensions
    file_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".bmp",
                ".dib", ".pbm", ".ppm", ".pgm", ".pnm")

    # Look for files that contain file_name in their names
    img_names = [] # A list of strings for the image file paths
    if file_name:
        files = glob.glob(dir_path + '*') # All files in directory
        if files:
            for cur_name in files:
                # Makes forward slashes into backward slashes for W10, and
                # also makes the directory string all lowercase
                cur_name = os.path.normcase(cur_name)
                if file_name in cur_name:
                    # Also ensure this is actually an image file
                    if (cur_name.lower()).endswith(file_ext):
                        img_names.append(cur_name)

    # Otherwise, import all common image file types (case insensitive)  
    else:
        files = glob.glob(dir_path + '*') # All files in directory
        if files:
            # Makes forward slashes into backward slashes for W10, and
            # also makes the directory string all lowercase
            cur_name = os.path.normcase(cur_name)
            for cur_name in files:
                # Only keep image files
                if (cur_name.lower()).endswith(file_ext):
                    img_names.append(cur_name)

    # Ensure that the list of names is not empty
    if not img_names:
        print(f"\nNo image files were found in {dir_path}")
        return [None, None]

    # The Glob module does not guarantee that the results will be sorted.
    img_names = list(dict.fromkeys(img_names)) # Removes duplicates
    img_names.sort() # Asecending order based on the string file names

    if indices_keep: # If not empty
        # If not already, force the data to be positive and of type integer
        for count, element in enumerate(indices_keep):
            indices_keep[count] = int(abs(element))

        if len(indices_keep) == 1: # Select a number of images from the center
            n_keep = indices_keep[0]
            cur_img_len = len(img_names) # Total number of images

            if n_keep == 0: # Keeping zero images doesn't make sense
                n_keep = 1  # Will just keep the middle image then
            elif n_keep > cur_img_len: # Also doesn't make sense
                n_keep = cur_img_len   # So, keep them all

            # Index of middle element, rounds down if even
            if cur_img_len % 2 == 0: # If even
                mid_indx = int(cur_img_len/2) - 1
            else: # If odd
                mid_indx = int(cur_img_len/2)

            # Force the desired number of images to keep to be odd for now
            keep_even = False
            if n_keep % 2 == 0:
                keep_even = True
                n_keep -= 1

            # Number of images to keep above and below the middle index
            indx_rad = int((n_keep - 1)/2)

            # Index bounds for the images to be kept
            indx_bounds = [mid_indx - indx_rad, mid_indx + indx_rad + 1]

            # If originally even number of images to keep, correct it here
            if keep_even:
                indx_bounds[1] = indx_bounds[1] + 1

            # Perform the slice operation
            img_names = img_names[indx_bounds[0]:indx_bounds[1]]

        elif len(indices_keep) == 2: # Expecting lower and upper indices
            ind_low = indices_keep[0]
            ind_upp = indices_keep[1]

            img_names = img_names[ind_low:ind_upp]

        else:
            print(f"\nInvalid indices to slice the array of images. Provided"\
                f" input was: {indices_keep}\nExpecting a tuple of length "\
                f"1 or 2. For example:\n    1) (200,) --> Keep 200 images "\
                f"about the center.\n    2) (0, 100) --> Keep the first 100 "\
                f"images")

    # Number of images in the finalized list of file paths
    num_imgs = len(img_names)

    # [img_props]: A list containing the image properties for each 
    # imported image. Each entry in img_props is a list in of 
    # itself. More details given below.
    # 
    # img_prop[m][0]: Total number of pixels in the m-th image.
    # img_prop[m][1]: A tuple of length two that provides the  
    #     number of rows and columns of pixels for the m-th image. 
    # img_prop[m][2]: A string that confirms the image data type 
    #     for m-th image. Since load_image(...) converts all 
    #     images to 'uint8', this will always be equal to 'uint8'.
    img_props = []
    imgs = [] # Will be converted to Numpy array later
    counter = 1
    print(f"\nBeginning to import {num_imgs} images...")
    for cur_name in img_names:
        # Now actually call OpenCV routines to import the images
        # Use the defined function from above to import images in gray scale
        # and as uint8. Also, use quiet-mode (no outputs to the terminal)

        [cur_img, temp1] = read_pgm(cur_name, transpose=transpose, 
            ASCII=ASCII)
        cur_img_prop = [cur_img.size, cur_img.shape, cur_img.dtype]

        #cur_img, cur_img_prop = load_image(cur_name, quiet_in=True)
        #if cur_img is None:
        #    print(f"\nFailed to import image: {cur_name}")
        #    return [None, None]

        # Store the image objects and their properties in lists
        imgs.append(cur_img)
        img_props.append(cur_img_prop)

        # Write out updates for the user
        if (counter%20) == 0:
            print(f"    Currently imported {counter}/{num_imgs}...")
        
        counter += 1

    print(f"\nSuccessfully imported {num_imgs} images!")

    # Check to make sure all the images are the same shape
    first_img_shape = img_props[0][1] # Tuple of the image shape (rows, cols)
    cur_index = 0
    for cur_img_prop in img_props:
        cur_shape = cur_img_prop[1]

        if cur_shape != first_img_shape:
            print(f"\nWARNING! Not all imported images are of the same shape")
            print(f"\nFirst image shape: {first_img_shape}")
            print(f"\nImage shape of {img_names[cur_index]}: {cur_shape}")

        cur_index += 1

    # Convert from a Python list of Numpy arrays to a fully 3D Numpy array
    imgs = np.array(imgs)

    # Return the image objects, image properties, and string file names
    return [imgs, img_names]


def save_image(img_in, path_in, compression=False, quiet_in=False):
    """ 
    Saves an OpenCV image to the hard drive.
    
    ---- INPUT ARGUMENTS ----
    [img_in]: OpenCV's numpy array for a grayscale image. It is assumed 
        that the image is already grayscale and of type uint8. The array
        should thus be 2D, where each value represents the intensity for
        each corresponding pixel. 

    path_in: String that contains the file path (and name and extension)
        of the image being saved.

    compression: A boolean that determines if the tiff images should
        be compressed. If True, LZW compression is done. If False,
        no compression is performed. This compression flag is only
        used for saving tif (or tiff) images.

    quiet_in: An optional boolean that is by default set to False. Set 
        to True to prevent outputting any messages.

    ---- RETURNED ---- 
    retval: A boolean value. Returns True if img_in was successfully
        saved using the file path given by path_in. If img_in failed
        to save, retval will be False.

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. An image file is written
    to the hard drive. Strings may be printed to standard output.
    """

    # Make local copies
    img = img_in # No need for a deep copy. This image won't be altered

    # Forcing Python to create a new string variable in memory
    # Path to the directory to save the images
    file_path = (path_in + '.')[:-1]

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    file_path = os.path.normcase(file_path)
    compress_bool = compression
    quiet = quiet_in

    # Reduced list (actually tuple) of supported image file extensions
    file_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".bmp",
                ".dib", ".pbm", ".ppm", ".pgm", ".pnm")

    retval = False

    try:
        if compress_bool:
            # LZW compression if True
            retval = cv.imwrite(file_path, img, 
                params=(cv.IMWRITE_TIFF_COMPRESSION, 5))

        else:
            # No compression if False
            retval = cv.imwrite(file_path, img, 
                params=(cv.IMWRITE_TIFF_COMPRESSION, 1))

    except:
        if not quiet:
            print(f"\nERROR: Encountered an error trying to save {file_path}")

        if not (file_path.lower()).endswith(file_ext):

            if not quiet:
                print(f"Detected invalid file extension. Attempting to "\
                "save as a '.tif' file...")

            file_path = file_path + ".tif"

            try:
                if compress_bool:
                    # LZW compression if True
                    retval = cv.imwrite(file_path, img, 
                        params=(cv.IMWRITE_TIFF_COMPRESSION, 5))

                else:
                    # No compression if False
                    retval = cv.imwrite(file_path, img, 
                        params=(cv.IMWRITE_TIFF_COMPRESSION, 1))

            except:
                if not quiet:
                    print(f"FAILED to save {file_path}")

    if retval:
        if not quiet_in:
            print(f"\nSuccessfully saved {file_path}")

    return retval


def save_image_seq(imgs_in, dir_in_path, file_name_in, 
    index_start_in=0, compression=False):
    """ 
    Saves a list of OpenCV images to the hard drive.
    
    ---- INPUT ARGUMENTS ----
    [imgs_in]: A 3D Numpy array containing images, each represented by
        a Numpy array, which corresponds to OpenCV's single-channel 
        uint8 (grayscale) image data type. The shape of imgs_in should
        be (num_images, num_rows, num_cols).

    dir_in_path: String that contains the directory path of the image
        being saved. The directory path will be normalized and made 
        lowercase, so forward or backward slashes are acceptable.

    file_name_in: String containing the name to be used for saving all
        of the processed images. The image names will automatically be
        appended with a 4-digit number to denote the correct sequence.
        Be sure to include the desired filetype, else by default, the
        images will be saved as '.tif' files. For example, if you had
        three images to save and chose name_out_substr = "out.tif", 
        then the saved images would be 'out_0000.tif', 'out_0001.tif',
        and 'out_0002.tif'.

    index_start_in: An integer that is used as initial value for the
        appended number to the saved image name. For example, if you had
        three images to save and chose name_out_substr = "out.tif" and
        index_start_in = 10,  then the saved images would be
        'out_0010.tif', 'out_0011.tif', and 'out_0012.tif'.

    compression: A boolean that determines if the tiff images should
        be compressed. If True, LZW compression is done. If False,
        no compression is performed. This compression flag is only
        used for saving tif (or tiff) images.

    ---- RETURNED ---- 
    Nothing is returned    

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Image files are written
    to the hard drive. Strings may be printed to standard output.
    """
    
    # Make local copies
    imgs = imgs_in # No need for a deep copy. This image won't be altered

    # Forcing Python to create a new string variable in memory
    file_path = (dir_in_path + '.')[:-1]
    if not file_path.endswith( ('\\', '/') ):
        file_path = file_path + '/'

    file_name = (file_name_in + '.')[:-1]
    file_name = os.path.normcase(file_name)

    # Makes forward slashes into backward slashes for W10, and also makes the
    # directory string all lowercase
    file_path = os.path.normcase(file_path)
    index_start = int(index_start_in)
    compress_bool = compression

    # Reduced list (actually tuple) of supported image file extensions
    file_ext3 = (".tif", ".png", ".jpg", ".jp2", ".bmp", ".dib", ".pbm",
                 ".ppm", ".pgm", ".pnm")
    file_ext4 = (".tiff", ".jpeg")

    # Check for the correct file extension and save it
    if (file_name.lower()).endswith(file_ext3):
        file_ext = file_name[-4:] # Does inclue '.'
        name_substr = file_name[:-4] # Does not inluce '.'

    elif (file_name.lower()).endswith(file_ext4):
        file_ext = file_name[-5:] # Does inclue '.'
        name_substr = file_name[:-5] # Does not inluce '.'

    else:
        print(f"\nWarning: No supported file extension found in "\
            f"'{file_name}'\n    Defaulting to '.tif' file extension.")
        name_substr = file_name
        file_ext = ".tif"

    num_imgs = imgs.shape[0]
    img_ii = index_start
    print(f"\nBeginning to save {num_imgs} images to:\n    {file_path}")
    for img_count, cur_img in enumerate(imgs):

        if num_imgs < 1E4:
            img_str_indx = "_" + str(img_ii).zfill(4)

        elif 1E4 <= num_imgs < 1E5:
            img_str_indx = "_" + str(img_ii).zfill(5)

        elif 1E5 <= num_imgs < 1E6:
            img_str_indx = "_" + str(img_ii).zfill(6)

        elif 1E6 <= num_imgs < 1E7:
            img_str_indx = "_" + str(img_ii).zfill(7)

        else:
            print("\nERROR: Over 1E7 images not currently supported in "\
                "save_image_seq(...). \nPlease use smaller batches.")
            return

        # Construct the full file path for the current image
        # (directory_path) + (file_name_substring) + (file_index) + 
        # (file_extension)
        cur_file_name = file_path + name_substr + img_str_indx + file_ext

        save_flag = save_image(cur_img, cur_file_name, 
            compression=compress_bool, quiet_in=True)

        if not save_flag:
            # I wonder if using a "raise Exception()" would be better
            # than a simple print statement here? That way one could call
            # it more robustly in a try...except block, right?
            print(f"\nERROR: Failed to save {cur_file_name}")

        elif ((img_count+1)%20) == 0:
            print(f"    Currently saved {img_count+1}/{num_imgs}...")

        img_ii += 1

    if save_flag:
        print(f"\nSuccessfully saved {num_imgs} images!")


def save_multipage_image(imgs_in, path_in, bigtiff=0,\
    compression=False, quiet_in=False):
    """ 
    Saves a list (or Numpy array) of images to the hard drive.
    The image stack will be saved specifically as a multipage
    TIFF file. To save the image stack as multiple, individual
    images, see save_image_seq(...).

    ---- INPUT ARGUMENTS ----
    [imgs_in]: A 3D Numpy array containing images, each represented by
        a Numpy array, which corresponds to OpenCV's single-channel 
        uint8 (grayscale) image data type. The shape of imgs_in should
        be (num_images, num_rows, num_cols).

    path_in: String that contains the file path of the image being 
        saved. This string should end with either '.tif' or '.tiff'. 
        Note, the file path will be normalized and made lowercase 
        (on Windows 10), so forward or backward slashes are acceptable.

    bigtiff: Integer that is either 0, 1, or 2. Changes the header and
        TIFF codec formatting. Select 0 for a standard TIFF file. If
        the file format will exceed 4 GB, then select either 1 or 2.
        Selecting 1 corresponds to using BigTIFF format, and selecting
        2 corresponds to using the native ImageJ Hyperstack format.

    compressions: Boolean that governs whether the image will be 
        compressed via ZLib. True to compress the multi-page TIFF
        image, and False to leave it as uncompressed.

    quiet_in: An optional boolean that is by default set to False. Set 
        to True to prevent outputting any messages.

    ---- RETURNED ---- 
    Nothing is returned    

    ---- SIDE EFFECTS ---- 
    Function input arguments are not altered. Image files are written
    to the hard drive. Strings may be printed to standard output.
    """

    # Make local copies
    imgs = imgs_in # No need for a deep copy. This image won't be altered

    # Forcing Python to create a new string variable in memory
    file_path = (path_in + '.')[:-1]
    file_path = os.path.normcase(file_path)

    quiet = quiet_in

    # Reduced list (actually tuple) of supported image file extensions
    file_ext3 = (".tif",)
    file_ext4 = (".tiff",)

    # Check for the correct file extension and save it
    if (file_path.lower()).endswith(file_ext3):
        file_ext = file_path[-4:] # Not actually used

    elif (file_path.lower()).endswith(file_ext4):
        file_ext = file_path[-5:] # Not actually used

    else:
        print(f"\nWarning: No supported file extension found in "\
            f"'{file_path}'\n    Defaulting to '.tif' file extension.")
        file_path = file_path + ".tif"

    num_imgs = imgs.shape[0]

    if not quiet:
        print(f"\nBeginning to save {num_imgs} images to:\n    {file_path}")

    # OpenCV's imwrite is supposed to automatically create a multi-page tiff
    # file when the filepath ends with ".tif" and a 3D Numpy array is given
    # as input. However, it does not work for me.
    #retval = cv.imwrite(file_path, img_list)

    # Sci-Kit Image does not officially support multi-page TIFF files, but
    # a number of other libraries do, such as PIL and TIFFFILE. Instead of
    # pulling in yet another dependency, I tried a workaround where I call
    # different plugin within Sci-Kit Image to perform the operation. As
    # it turns out, TIFFFILE automatically recognizes a 3D Numpy array and
    # will save it as a multi-page TIF file.


    # note: I think this is a somewhat cleaner way to set it up.
    # P.S. I wanted some python practice so I did this - certainly wasn't
    # necessary. There might still be a better way...
    if compression == True:
        compression = 'zlib'
    elif compression == False:
        compression = ''
    else:
        if not quiet:
            print("WARNING: 'compression' is unset. Defaulting to False")
        compression = ''
    
    if bigtiff == 0:
        bigtiff = False
        imagej = False
    elif bigtiff == 1:
        bigtiff = True
        imagej = False
    elif bigtiff == 2:
        bigtiff = False
        imagej = True
    else:
        if not quiet: 
            print(f"\nWARNING: bigtiff entry '{bigtiff}' is invalid!")
            print(f"\nDefaulting to bigtiff=0")
        bigtiff = False
        imagej = False

    bigtiff_compression_settings = {'bigtiff': bigtiff,\
                                    'imagej': imagej,\
                                    'compression': compression}
   
    io.imsave(file_path, imgs, plugin='tifffile', \
              photometric='minisblack', **bigtiff_compression_settings)

    if not quiet:
        print(f"\nSuccessfully saved {num_imgs} images!")
