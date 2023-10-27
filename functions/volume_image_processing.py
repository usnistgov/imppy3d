import numpy as np
from scipy import ndimage as ndim
from scipy import spatial as spt
#from scipy.spatial import transform as trfm
from skimage import transform as tran
from skimage import measure as meas
from skimage import segmentation as seg
from skimage.util import img_as_ubyte
from skimage.util import img_as_uint
from skimage.util import img_as_float32
from skimage.util import img_as_float64
import pyvista as pv
import cv_driver_functions as drv
import cython.bin.im3_processing as im3
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

        
def pad_image_boundary(img_arr_in, cval_in=0, n_pad_in=1, quiet_in=False):
    """
    Adds enlarges the image array in all directions by one extra
    row of pixels, and then pads these locations with values equal
    to cval_in. So, for a 2D image of size equal to (rows, cols), 
    the output image with extended boundaries in all directions will
    be of size (rows + 2, cols + 2). In 3D, which is a sequence of
    images, (num_imgs, rows, cols), the output will be a similarly
    extended image array, (num_imgs + 2, rows + 2, cols + 2). Note,
    the SciKit-Image library has a more powerful function if it is
    needed, called skimage.util.pad().

    ---- INPUT ARGUMENTS ---- 
    [[img_arr_in]]: A 2D or 3D Numpy array representing the image 
        sequence. If 3D, is important that this is a Numpy array and not
        a Python list of Numpy matrices. If 3D, the shape of img_arr_in
        is expected to be as (num_images, num_pixel_rows,
        num_pixel_cols). If 2D, the shape of img_arr_in is expected to
        be as (num_pixel_rows, num_pixel_cols). It is expected that the
        image(s) are single-channel (i.e., grayscale), and the data
        type of the values are np.uint8.

    cval_in: Constant padding value to be used in the extended regions
        of the image array. This should be an integer between 0 and 255,
        inclusive. 

    n_pad_in: An integer value to determine how much to pad by. For 
        example, if img_arr_in is of shape (5, 10, 20), and n_pad is
        equal to 3, then three extra rows/columns will be created on
        all sides. The resulting size would be (5+3*2, 10+3*2, 20+3*2).

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [[img_arr_out]]: The image array with extended boundaries in all
        dimensions. The input image array will be in the center of this
        output array, and the data type will be uint8. 

    ---- SIDE EFFECTS ---- 
    The input array may be affected since a deep copy is not made in 
    order to be more memory efficient. Strings are printed to standard
    output. Nothing is written to the hard drive.
    """

    # ---- Start Local Copies ----
    img_arr = img_arr_in # Makes a new view -- NOT a deep copy
    cval = cval_in
    n_pad1 = np.around(n_pad_in).astype(np.int32)
    n_pad2 = (2*n_pad1).astype(np.int32)
    quiet = quiet_in
    # ---- End Start Local Copies ----

    if not quiet_in:
        print(f"\nExtending the boundaries of the image data...\n"\
            f"    Values in the padded regions will be set to: {cval}")

    img_shape = img_arr_in.shape
    if len(img_shape) == 3:
        num_imgs = img_shape[0]
        num_rows = img_shape[1]
        num_cols = img_shape[2]

        img_arr_out = np.ones((num_imgs+n_pad2, \
            num_rows+n_pad2, num_cols+n_pad2), dtype=np.uint8)*cval

        img_arr_out[n_pad1:num_imgs+n_pad1, n_pad1:num_rows+n_pad1, \
            n_pad1:num_cols+n_pad1] = img_arr

    elif len(img_shape) == 2:
        num_rows = img_shape[0]
        num_cols = img_shape[1]

        img_arr_out = np.ones((num_rows+n_pad2, num_cols+n_pad2), \
            dtype=np.uint8)*cval

        img_arr_out[n_pad1:num_rows+n_pad1, n_pad1:num_cols+n_pad1] = img_arr

    else:
        if not quiet_in:
            print(f"\nERROR: Can only pad the boundary of an image array "\
                f"in 2D or 3D.\nCurrent image shape is: {img_shape}")

    if not quiet_in:
        print(f"\nSuccesfully padded the image boundaries!")

    return img_arr_out


def image_rescale(img_in, quiet_in=False):
    """
    Just use skimage.transform.rescale(...) instead. The skimage
    function uses interpolation functions and can apply anti-
    aliasing on the edges.
    """
    pass


def image_downscale(img_arr_in, factors_in, cval_in=0, 
    quiet_in=False):
    """
    Down-sample N-dimensional image by local averaging. The image is
    padded with cval if it is not perfectly divisible by the integer
    factors. This function calculates the local mean of elements in each
    block of size factors in the input image. This function is based on
    Scikit-Image's function, downscale_local_mean(...). However, that
    function returns an image sequence with float64 data types. So, this
    function will recast this output into a np.uint8 image sequence. In
    doing so, the histogram will also be normalized from 0 to 255.

    ---- INPUT ARGUMENTS ----
    [[img_arr_in]]: A 2D or 3D Numpy array representing the image 
        sequence. If 3D, is important that this is a Numpy array and not
        a Python list of Numpy matrices. If 3D, the shape of img_arr_in
        is expected to be as (num_images, num_pixel_rows,
        num_pixel_cols). If 2D, the shape of img_arr_in is expected to
        be as (num_pixel_rows, num_pixel_cols). It is  expected that the
        image(s) are single-channel (i.e., grayscale), and the data
        type of the values are np.uint8.

    (factors_in): A tuple containing two or three integers representing 
        how much to downsample the image array along each axis. If
        img_arr_in  is a single image (i.e., 2D), then provide two
        integers: (rows, cols).  If img_arr_in is a 3D image sequence,
        then provide three integers:  (num_imgs, rows, cols). A value of
        one  would not scale the image along that axis, and a value of
        two  would downscale the image by a factor of two along that
        axis.

    cval_in: Constant padding value if the image(s) are not perfectly 
        divisible by the integer factors.

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [[img_arr_out]]: The downscaled image data are returned. It will be
        the same dimensions as img_arr_in, and the data type will be
        uint8. 

    ---- SIDE EFFECTS ---- 
    The input array may be affected since a deep copy is not made in 
    order to be more memory efficient. Strings are printed to standard
    output. Nothing is written to the hard drive.
    """

    # ---- Start Local Copies ----
    img_arr = img_arr_in # Makes a new view -- NOT a deep copy
    factors = factors_in
    cval = cval_in
    quiet = quiet_in
    # ---- End Start Local Copies ----

    if not quiet:
        if len(factors) == 3:
            print("\nDownscaling image sequence by factors...\n"\
                f"    Along Z-axis through Image Sequence: {factors[0]}\n"\
                f"    Along Pixel Rows of Each Image: {factors[1]}\n"\
                f"    Along Pixel Columns of Each Image: {factors[2]}")
        elif len(factors) == 2:
            print("\nDownscaling image by factors...\n"\
                f"    Along Pixel Rows of Each Image: {factors[0]}\n"\
                f"    Along Pixel Columns of Each Image: {factors[1]}")

    img_arr_down = tran.downscale_local_mean(img_arr, factors, cval)

    img_arr_down_dtype = img_arr_down.dtype
    if (img_arr_down_dtype != "uint8"):
        if not quiet:
            print(f"\nWARNING: Image data type being converted from"\
                f" '{img_arr_down_dtype}' to 'uint8'")

        # Even though the image contains float64, SciKit-Image expects the
        # values to be normalized. So, need to do this first.
        max_value = np.amax(img_arr_down)
        img_arr_down = img_as_float64(img_arr_down) # Force it to np.float64

        # By normalizing the image by the maximum value, I am effectively
        # normalizing the histogram from 0.o to 1.0. When it gets converted
        # back to uint8, then it will be from 0 to 255.
        img_arr_down = img_arr_down/max_value # Normalized from 0.0 to 1.0

        # Clips any negative values, and scales all positive values from
        # 0 to 255
        img_arr_out = img_as_ubyte(img_arr_down) 

    else:
        img_arr_out = img_arr_down

    if not quiet:
        print("\nSuccessfully downscaled image data!")

    return img_arr_out


def calc_surf_normal3(img_arr_in, search_in, radius_in):
    """
    The objective of this function is to seek out white pixels along
    a surface patch of a feature in a segemented image sequence, and
    then, calculate a unit normal vector to the best-fit plane to this
    surface patch. The algorithm used to find white pixels begins by
    simply starting at the edge of the image sequece (i.e., from any of
    the six orthogonal faces of the volume defined by the indices). 
    Then, it iteratively progresses towards the center of the image
    sequence by increasing or decreasing one of the indices in 
    img_arr_in. The first white pixel it encounters will be saved. This
    method is done repeatedly in the neighborhood of this first white
    pixel. After the indices are found for the white pixels 
    corresponding to the local surface patch, a singular value 
    decomposition is performed to calculate the outward, unit normal
    vector to the best-fit plane of these points. The indices themselves
    are used as a coordinate system, so the coordinate system can be
    thought of as an intrinsic coordinate system based on the indices.

    ---- INPUT ARGUMENTS ----
    [[img_arr_in]]: A 3D Numpy array representing the image sequence.
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. If 3D, the shape of img_arr_in is expected to
        be as (num_images, num_pixel_rows, num_pixel_cols). It is 
        expected that the images are segmented into black and white
        pixels, and the data type of the values are np.uint8. So, the
        image sequence should contain values equal to 0 and 255 only.
        The algorithm will classify anything with intensity greater than
        or equal to 1 as a "white" pixel.

    (search_in): A tuple containing three values, each corresponding to 
        the indices of img_arr_in. This tuple will define the 
        neighborhood that will be searched. Two of the values should be
        integer values representing the indices that will be used to 
        begin the search in img_arr_in. The last value in search_in 
        should be either "ascend" or "descend", which will be used to
        determine the direction of the raycasts that are used to find
        the surface. Note, Python indices start at zero, so the 16th
        index should be given as 15. Examples are given below:
            ("ascend", 50, 75): This implies that the edge of white
                pixels lie at the row index 50 and column index 75.
                The "ascend" keyword implies that the algorithm will
                search in ascending values of the image index until a
                white pixel is found. So, img_arr_in[0,50,75] will be
                checked first, then img_arr_in[1,50,75], and so on.
            (25, 30, "descend"): This implies that the edge of white
                pixels lies at the image index 25 and row index 30.
                The "descend" keyword implies that the algorithm will
                search in descending values of the column index until a
                white pixel is found. So, img_arr_in[25,30,-1] will be
                checked first, then img_arr_in[25,30,-2], and so on.

    (radius_in): A tuple that contains two integers, which define the
        number of pixels to search along the surface based on the 
        initial white pixel that was found on search_in as the center.
        Recall that search_in is a tuple of three values, two of which
        are integers. The two integers in radius_in correspond to the
        two integers in search_in, and order matters: the first integer
        in search_in corresponds to the first integer in radius_in, and
        the second integer in search_in corresponds to the second 
        integer in radius_in. In short, radius_in controls the radial 
        size, or really the half-width in each dimension, of the region
        to search along in the index dimensions  defined by search_in.
        An example is given below:
            Let's assumed search_in = ("ascend", 50, 75). If radius_in 
            is (5, 10), then white pixels are going to be sought out 
            between row indices 45 and 55, as well as between column
            between indices 65 and 85. It can be thought of as defining
            the "slice length" of search_in: ("ascend", 45:55, 65:85).

    ---- RETURNED ---- 
    [dir_vec3_out]: A Numpy array containing three floats, which define
        the unit vector that is normal to the best-fit plane of the
        pixels that were found along the surface neighborhood. This
        direction vector can be thought of as being defined in pixel
        coordinates, since it is based on the image sequence indices.
        However, the direction vector will not be rounded off to the
        closest integer, so one could think of this vector being defined
        in an intrinsic coordinate system. The vector will be directed
        outward with respect to the feature, which is based on the 
        opposite direction of the raycast search algorithm.

    [pix_coords_out]: A 2D Numpy matrix that contains the indices of the
        pixels that were used from img_arr_in to calculate the surface
        normal vector. The matrix will be three columns wide, and each
        row corresponds to a pixel. The first column contains image 
        indices, the second column contains row indices, and the last
        column contains column indices. 

    [centroid_out]: A Numpy array containing three floats corresponding
        to the coordinates of the centroid for the points given in
        pix_coords_out.

    ---- SIDE EFFECTS ---- 
    The input array may be affected since a deep copy is not made in 
    order to be more memory efficient. Strings are printed to standard
    output. Nothing is written to the hard drive.
    """

    # ---- Start Local Copies ----
    img_arr = img_arr_in # Makes a new view -- NOT a deep copy
    init0 = search_in[0]
    init1 = search_in[1]
    init2 = search_in[2]
    rad1 = (np.rint(radius_in[0])).astype(np.int)
    rad2 = (np.rint(radius_in[1])).astype(np.int)
    # ---- End Start Local Copies ----

    # Determine which dimension the raycast will be performed along
    # and determine if indices will be ascending or descending
    is_ascend = True
    ray_dim = 0
    max_ray_dim = 0
    indx_pln = (0,0)
    if type(init0) == str:
        ray_dim = 0
        max_ray_dim = img_arr.shape[0]
        indx_pln = (    np.rint(init1).astype(np.int), \
                        np.rint(init2).astype(np.int)   )
        init0 = (init0.upper()).strip()
        if init0[0] == 'D':
            is_ascend = False

    elif type(init1) == str:
        ray_dim = 1
        max_ray_dim = img_arr.shape[1]
        indx_pln = (    np.rint(init0).astype(np.int), \
                        np.rint(init2).astype(np.int)   )
        init1 = (init1.upper()).strip()
        if init1[0] == 'D':
            is_ascend = False

    elif type(init2) == str:
        ray_dim = 2
        max_ray_dim = img_arr.shape[2]
        indx_pln = (    np.rint(init0).astype(np.int), \
                        np.rint(init1).astype(np.int)   )
        init2 = (init2.upper()).strip()
        if init2[0] == 'D':
            is_ascend = False

    else:
        print(f"\nERROR: Could not find 'ascend' or 'descend' in "\
            f"search_in!\n    Supplied input for search_in: {search_in}")
        return [None, None, None]

    # Find the white pixels
    num_search_pix = (rad1*2 + 1)*(rad2*2 + 1)
    pix_coords = (np.zeros((num_search_pix,3))).astype(np.int)

    # Search within the range of indices
    pix_cnt = 0
    skip_pix = []
    for m in range(indx_pln[0]-rad1, indx_pln[0]+rad1+1):
        for n in range(indx_pln[1]-rad2, indx_pln[1]+rad2+1):

            if ray_dim == 0:
                m_up_bound = img_arr.shape[1]
                n_up_bound = img_arr.shape[2]
            elif ray_dim == 1:
                m_up_bound = img_arr.shape[0]
                n_up_bound = img_arr.shape[2]
            else:
                m_up_bound = img_arr.shape[0]
                n_up_bound = img_arr.shape[1]

            # Ensure valid indices, else, skip
            if (m < 0) or (n < 0):
                skip_pix.append(pix_cnt)
                pix_cnt += 1
                continue

            elif (m >= m_up_bound) or (n >= n_up_bound):
                skip_pix.append(pix_cnt)
                pix_cnt += 1
                continue

            # Initialize search variables
            if is_ascend:
                ray_indx = 0
            else:
                ray_indx = max_ray_dim - 1

            # Begin raycast along the specified dimension and direction
            found_pix = False
            end_of_arr = False
            while not found_pix:

                if ray_dim == 0:
                    cur_pix = img_arr[ray_indx,m,n]
                elif ray_dim == 1:
                    cur_pix = img_arr[m,ray_indx,n]
                else:
                    cur_pix = img_arr[m,n,ray_indx]

                # If greater than 0 (i.e., black), keep the index coordinates
                if cur_pix > 0:
                    if ray_dim == 0:
                        pix_coords[pix_cnt,:] = np.array([ray_indx,m,n])
                    elif ray_dim == 1:
                        pix_coords[pix_cnt,:] = np.array([m,ray_indx,n])
                    else:
                        pix_coords[pix_cnt,:] = np.array([m,n,ray_indx])

                    found_pix = True # Exit while loop at next check

                if is_ascend:
                    ray_indx += 1 # Increment index for ascending
                    if ray_indx >= max_ray_dim:
                        end_of_arr = True
                else:
                    ray_indx -= 1 # Decrement index for descending
                    if ray_indx < 0:
                        end_of_arr = True

                # Increment counter for number of pixels searched in this raycast
                if end_of_arr and (not found_pix):
                    # Skipping this current raycast. Will not store a pixel.
                    found_pix = True # Exit at next check. Failed search.
                    skip_pix.append(pix_cnt)

            pix_cnt += 1 # Increment the counter for number of pixel searches 

    # Store the pixels that were found; skip over the failed searches
    if len(skip_pix) == num_search_pix:
        print("\nERROR: No white pixels found for provided search parameters!")
        return [None, None, None]

    elif skip_pix:
        pix_coords_out = (np.zeros((num_search_pix-len(skip_pix),3))).astype(np.int)
        skip_m = 0
        save_m = 0
        for m, cur_pix in enumerate(pix_coords):
            # Don't store the pixel at this index if this is true
            if m == skip_pix[skip_m]: 
                if skip_m < (len(skip_pix) - 1): # Only increase if not last skip
                    skip_m += 1 # Increase the skip index (i.e., number of skips)
            else: # Else, copy the pixel indices over to be saved
                pix_coords_out[save_m,:] = pix_coords[m,:]
                save_m += 1
    else:
        # Store all of them if no failed searches
        pix_coords_out = pix_coords.copy()

    num_pnts = len(pix_coords_out)
    if num_pnts < 3:
        print("\nERROR: Found less than 3 white pixels! Not enough to fit a plane.")
        return [None, None, None]

    print(f"\nFound {num_pnts} white pixels! "\
    "Calculating normal vector of the best-fit plane.")

    # Calculate the best-fit plane to the indices, which are taken to be 
    # coordinates. First, need to subtract out the centroid based on averaging
    # centroid_out is stored as a 1D array (flat)
    centroid_out = np.mean(pix_coords_out, axis=0)
    points1 = pix_coords_out - centroid_out

    # Use singular value decomposition to find normal vector of best fitting plane
    u_mat, s_vec, vh_mat = np.linalg.svd(points1)

    # Save normal vector and ensure it is normalized to a unit vector
    norm_vec3 = np.array([0.0, 0.0, 0.0]) # Stored as a 1D array (flat)
    norm_vec3[0] = vh_mat[-1, 0] # Grab the last column. This is based on some
    norm_vec3[1] = vh_mat[-1, 1] # linear algebra theory and how SVD works
    norm_vec3[2] = vh_mat[-1, 2]
    vec3_mag = np.sqrt( np.power(norm_vec3[0], 2) + 
                        np.power(norm_vec3[1], 2) +
                        np.power(norm_vec3[2], 2) )
    norm_vec3 = norm_vec3/vec3_mag

    # Check to see if the normal vector is outward pointed
    if ray_dim == 0:
        dir_out = np.array([-1.0, 0.0, 0.0])
    elif ray_dim == 1:
        dir_out = np.array([0.0, -1.0, 0.0])
    else:
        dir_out = np.array([0.0, 0.0, -1.0])

    if not is_ascend: # Flip direction if descending
        dir_out = -1.0*dir_out

    cos_ang = np.dot(norm_vec3, dir_out) # Both are unit vectors
    if cos_ang <= 0.0: # More than 90 deg apart, so flip the vector
        norm_vec3 = -1.0*norm_vec3

    dir_vec3_out = norm_vec3

    return [dir_vec3_out, pix_coords_out, centroid_out]


def calc_surf_normal2():
    """
    Not yet implemented. Similar to calc_surf_normal3(), but for a 
    single 2D image.
    """
    pass


def intrinsic_to_spatial(coords_in, make_copy=True):
    """
    Converts intrinsic coordinates of an image array into spatial 
    coordinates by simply flipping the indices. coords_in is a matrix
    of coordinates, three columns wide, with column one representing 
    the image index, column two as the row index, and column three as
    the column index. This can be interpretted as each row representing
    (Z, Y, X) coordinates. So, this function flips each row of the 
    matrix so that the output is in (X, Y, Z) format. Some conventions 
    also add an extra entry that is 0.5 larger than the largest 
    intrinsic coordinate and subtract 0.5 from the rest. This function 
    does not modify the values; it only rearranges them. If make_copy
    is false, the original input will be modified.
    """
    if make_copy:
        if coords_in.ndim == 1:
            img_arr = coords_in.copy() 
            return np.flip(img_arr)    # np.flip() works for 1D arrays 
        else:
            img_arr = coords_in.copy() # np.fliplr() must operate on 
            return np.fliplr(img_arr)  # at least a 2D matrix
    else:
        if coords_in.ndim == 1:
            return np.flip(coords_in) 
        else:
            return np.fliplr(coords_in)


def spatial_to_intrinsic(coords_in, make_copy=True):
    """
    The same thing as intrinsic_to_spatial(...). Going from intrinsic
    index coordinates to spatial coordinates is the same thing as 
    flipping the indices around to go in reverse.
    """
    if make_copy:
        return intrinsic_to_spatial(coords_in, make_copy=True)
    else:
        return intrinsic_to_spatial(coords_in, make_copy=False)


def make_axis_angle_rot3_matrix(axis_in, theta_in, homogeneous=False):
    """
    Creates the rotation matrix corresponding to a proper rotation
    by angle theta_in (expected as radians) about axis defined by
    unit vector axis_in. 
    """
    th = theta_in
    axis_mag = np.sqrt( np.power(axis_in[0], 2) + 
                        np.power(axis_in[1], 2) +
                        np.power(axis_in[2], 2) )
    ax = axis_in/axis_mag
    ux = ax[0] # No switching around with the indices of axis_in!
    uy = ax[1]
    uz = ax[2]
    #ux = ax[2]
    #uy = ax[1]
    #uz = ax[0]

    costh = np.cos(th)
    sinth = np.sin(th)

    R11 = costh + ux*ux*(1. - costh)
    R12 = ux*uy*(1. - costh) - uz*sinth
    R13 = ux*uz*(1. - costh) + uy*sinth

    R21 = uy*ux*(1. - costh) + uz*sinth
    R22 = costh + uy*uy*(1. - costh)
    R23 = uy*uz*(1. - costh) - ux*sinth

    R31 = uz*ux*(1. - costh) - uy*sinth
    R32 = uz*uy*(1. - costh) + ux*sinth
    R33 = costh + uz*uz*(1. - costh)

    rot_mat = np.zeros((3,3))

    rot_mat[0,0] = R11
    rot_mat[0,1] = R12
    rot_mat[0,2] = R13

    rot_mat[1,0] = R21
    rot_mat[1,1] = R22
    rot_mat[1,2] = R23

    rot_mat[2,0] = R31
    rot_mat[2,1] = R32
    rot_mat[2,2] = R33

    if homogeneous:
        rot_mat_homog = np.zeros((4,4))
        rot_mat_homog[-1,-1] = 1.
        rot_mat_homog[0:3,0:3] = rot_mat
        return rot_mat_homog

    else:
        return rot_mat


def img_rotate_3D(img_arr_in, axis_in, angle_in, binarize=False):
    """
    Rotates an image sequence in 3D and "re-slices" it so that it has
    the same shape and size as the original image sequence. The rotation
    is performed about a specified axis that goes through the center
    of the image sequence. If the images (or image features) are not
    already centered, then this should be done first -- look into 
    affine_transform() using homogeneous coordinates in the 
    scipy.ndimage library. The rotations are performed using the 
    scipy.ndimage.rotate() function based on linear interpolation. If
    the rotation is not about one of the principal axes (i.e., not about
    the X-, Y-, or Z-axes), then the arbitrary rotation is performed as
    three successive rotations using extrinsic Euler angles.

    ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel (i.e., grayscale), and the data
        should be of type uint8.

    [axis_in]: A Numpy array of size 1-by-3 (i.e., a row vector). This
        vector represents the direction of the axis that will be used to
        rotate the image sequence about. The axis will be constructed by
        forming a line starting at the center of the image sequence and
        then going along direction axis_in towards infinity. The  image
        sequence will be rotated counter-clockwise about this axis,
        which follows the right-hand rule. Image indices are used 
        throughout, not spatial coordinates. So, axis_in[0] will be
        interpretted as the image index (Z-direction), axis_in[1] as the
        row index (Y-direction), and axis_in[2] as the column index
        (X-direction). Said explicitly, [1, 0, 0] will rotate about the
        Z-axis, [0, 1, 0] the Y-axis, and [0, 0, 1] the X-axis.

    angle_in: A scalar value representing the angle by which to rotate 
        the image sequence about axis_in. The angle should be given in
        radians. Said again, the image sequence will be rotated counter-
        clockwise about axis_in (i.e., right-hand rule).

    binarize: A True/False boolean that determines how to return the
        image sequence. If True, then the data will be binarized before
        being returned using a global threshold value of 128, where 0
        is black and 255 is white. For grayscale images, this is likely
        not needed. However, if a binarized image is given as input, 
        then the user might expect a binarized image as output. Due to
        the linear interpolation, this will not be the case unless 
        binarize is set to True.

    ---- RETURNED ---- 
    [[[img_arr]]]: The rotated image sequence is returned. It will have 
        the same shape as img_arr_in. Any pixels that get rotated
        outside of the image volume will be cropped. Regions that become
        "empty"  due to the rotation will be filled with zeros. 

    ---- SIDE EFFECTS ---- 
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. A copy of the rotated image sequence should be returned.
    Nothing is written to the hard drive.
    """

    img_arr = img_arr_in

    # Currently axis-angle representation
    angle = angle_in # (radians)
    axis_mag = np.sqrt( np.power(axis_in[0], 2) + 
                        np.power(axis_in[1], 2) +
                        np.power(axis_in[2], 2) )
    ax = axis_in/axis_mag # Unit vector

    # Check if rotation axis is approximately parallel to the
    # X-, Y-, or Z-axes of the image sequence. If so, then do a single
    # rotation instead of a set of Euler angle rotations.
    col_dir = np.array([0.0, 0.0, 1.0]) # X-direction
    row_dir = np.array([0.0, 1.0, 0.0]) # Y-direction
    img_dir = np.array([1.0, 0.0, 0.0]) # Z-direction

    # Radians
    col_axis_angle = np.arccos(np.clip(np.dot(col_dir, ax), -1.0, 1.0))
    row_axis_angle = np.arccos(np.clip(np.dot(row_dir, ax), -1.0, 1.0))
    img_axis_angle = np.arccos(np.clip(np.dot(img_dir, ax), -1.0, 1.0))

    # Make arrays for comparing against using np.isclose()
    col_angle_arr = np.array([col_axis_angle, col_axis_angle])
    row_angle_arr = np.array([row_axis_angle, row_axis_angle])
    img_angle_arr = np.array([img_axis_angle, img_axis_angle])
    paral_angle_arr = np.array([0.0, np.pi]) # Radians

    # Check within a tolerance of 1 degree
    angle_TOL = 1.0*np.pi/180.0 # Radians

    col_paral_bools = np.isclose(col_angle_arr, paral_angle_arr,
                                 rtol=0.0, atol=angle_TOL)
    row_paral_bools = np.isclose(row_angle_arr, paral_angle_arr,
                                 rtol=0.0, atol=angle_TOL)
    img_paral_bools = np.isclose(img_angle_arr, paral_angle_arr,
                                 rtol=0.0, atol=angle_TOL)

    if col_paral_bools[0] or col_paral_bools[1]:
        # Parallel to the column/X-axis
        angle_deg = angle*180.0/np.pi # Convert to degrees
    
        if col_paral_bools[1]: # Rotate in the correct direction
            angle_deg = -angle_deg
       
        # Axes will be selected as (0,1) but want it as (1,0). Therefore, need
        # to add a negative sign to the input angle.
        img_arr = ndim.rotate(img_arr, -angle_deg, axes=(1,0), reshape=False,
            order=1)

        if binarize:
            img_indx = 0
            for imgI in img_arr:
                cur_thsh = 128
                thsh_params = ["global", cur_thsh]

                imgI = drv.apply_driver_thresh(imgI, thsh_params, quiet_in=True)

                # Overwrite the current image in memory to save space
                img_arr[img_indx] = imgI.copy()
                img_indx += 1 # Update counter
    
    elif row_paral_bools[0] or row_paral_bools[1]:
        # Parallel to the row/Y-axis
        angle_deg = angle*180.0/np.pi # Convert to degrees
    
        if row_paral_bools[1]: # Rotate in the correct direction
            angle_deg = -angle_deg
    
        img_arr = ndim.rotate(img_arr, angle_deg, axes=(0,2), reshape=False,
            order=1)

        if binarize:
            img_indx = 0
            for imgI in img_arr:
                cur_thsh = 128
                thsh_params = ["global", cur_thsh]

                imgI = drv.apply_driver_thresh(imgI, thsh_params, quiet_in=True)

                # Overwrite the current image in memory to save space
                img_arr[img_indx] = imgI.copy()
                img_indx += 1 # Update counter

    elif img_paral_bools[0] or img_paral_bools[1]:
        # Parallel to the image/Z-axis
        angle_deg = angle*180.0/np.pi # Convert to degrees
    
        if img_paral_bools[1]: # Rotate in the correct direction
            angle_deg = -angle_deg
    
        # Axes will be selected as (1,2) but want it as (2,1). Therefore, need
        # to add a negative sign to the input angle.
        img_arr = ndim.rotate(img_arr, -angle_deg, axes=(2,1), reshape=False,
            order=1)

        if binarize:
            img_indx = 0
            for imgI in img_arr:
                cur_thsh = 128
                thsh_params = ["global", cur_thsh]

                imgI = drv.apply_driver_thresh(imgI, thsh_params, quiet_in=True)

                # Overwrite the current image in memory to save space
                img_arr[img_indx] = imgI.copy()
                img_indx += 1 # Update counter
    
    else:
        # Convert from (img,row,col) to (X,Y,Z) by reversing the order
        rot_vec_form = np.flip(ax*angle)
        rot_obj = spt.transform.Rotation.from_rotvec(rot_vec_form)
        #rot_obj = trfm.Rotation.from_rotvec(rot_vec_form)

        # Convert to proper Euler angles (z-x-z extrinsic). Gimbal lock would
        # occur for rotations about the Z-axis. However, this will not occur
        # in practice since pure Z-axis rotations are caught above.
        eul_angles_deg = rot_obj.as_euler('zxz', degrees=True)

        phi_deg = eul_angles_deg[0]
        theta_deg = eul_angles_deg[1]
        psi_deg = eul_angles_deg[2]
        
        # Perform each Euler rotation (Z-X-Z). See comments above why negative
        # signs are necessary for the input angles.
        img_arr = ndim.rotate(img_arr, -phi_deg, axes=(2,1), reshape=False,
            order=1)
        img_arr = ndim.rotate(img_arr, -theta_deg, axes=(1,0), reshape=False,
            order=1)
        img_arr = ndim.rotate(img_arr, -psi_deg, axes=(2,1), reshape=False, 
            order=1)

        if binarize:
            img_indx = 0
            for imgI in img_arr:
                cur_thsh = 128
                thsh_params = ["global", cur_thsh]

                imgI = drv.apply_driver_thresh(imgI, thsh_params, quiet_in=True)

                # Overwrite the current image in memory to save space
                img_arr[img_indx] = imgI.copy()
                img_indx += 1 # Update counter

    return img_as_ubyte(img_arr)


def rodrigues_rot3(pnts_arr_in, vec3_axis_in, angle_rad_in, 
        rot_center_in=None, dtype_out=np.float64, vectorized=True):
    """
    Applies the Rodrigues rotation formula to a Numpy array of coordinates.
    """

    # If memory is tight, consider trying to get away with a half-precision 
    # float to save memory. That is, np.float16 instead of np.float64.
    # Size is (n_pnts, 3) where each row contains the [X, Y, Z] coordinates
    pnts_arr_out = pnts_arr_in.astype(dtype_out, copy=True) # Full "deep" copy
    num_pnts = pnts_arr_in.shape[0]
    
    # Ensure the shapes of input arrays are correct
    vec3_axis = np.zeros(3, dtype=dtype_out) # row vector (of order 1)
    vec3_axis[0] = vec3_axis_in[0]
    vec3_axis[1] = vec3_axis_in[1]
    vec3_axis[2] = vec3_axis_in[2]

    # Ensure vec3_axis is a unit vector
    vec3_axis_mag = np.sqrt( np.power(vec3_axis[0], 2) + 
                    np.power(vec3_axis[1], 2) +
                    np.power(vec3_axis[2], 2) )
    vec3_axis = (vec3_axis/vec3_axis_mag).astype(dtype_out)

    # Assumed to be scalar (of order 0)
    angle_rad = angle_rad_in # [radians]

    # If rot_center is equal to None, rotate about the centroid of the points
    rot_center = np.zeros(3, dtype=dtype_out)
    if rot_center_in is None:
        sum_0 = np.sum(pnts_arr_in[:,0], dtype=np.float64) # To prevent overflow,
        sum_1 = np.sum(pnts_arr_in[:,1], dtype=np.float64) # use float for a
        sum_2 = np.sum(pnts_arr_in[:,2], dtype=np.float64) # second
        rot_center[0] = (sum_0/num_pnts).astype(dtype_out)
        rot_center[1] = (sum_1/num_pnts).astype(dtype_out)
        rot_center[2] = (sum_2/num_pnts).astype(dtype_out)
    else:
        rot_center = np.zeros(3, dtype=dtype_out) # row vector (of order 1)
        rot_center[0] = (rot_center_in[0]).astype(dtype_out)
        rot_center[1] = (rot_center_in[1]).astype(dtype_out)
        rot_center[2] = (rot_center_in[2]).astype(dtype_out)

    # Apply translation to make rot_center the origin. Will change back later
    trans_vec3_forw = np.array([0.0, 0.0, 0.0]) - rot_center
    trans_vec3_forw = trans_vec3_forw.astype(dtype_out)

    trans_vec3_back = -1.0*trans_vec3_forw
    trans_vec3_back = trans_vec3_back.astype(dtype_out)

    # Taking advantage of NumPy broadcasting to apply translation to each point
    # To avoid temporary copies, using NumPy add()
    np.add(pnts_arr_out, trans_vec3_forw, pnts_arr_out)

    if vectorized:
        # Apply the Rodrigues rotation formula. Taking advantage of
        # vectorized broadcasting Python/NumPy will have to make a temporary
        # copy of the right-hand side here before it gets saved into
        # pnts_arr_out.

        # Shape pnts_arr_out, [n, 3]
        # Shape term1, [n, 3]
        term1 = pnts_arr_out*(np.cos(angle_rad, dtype=dtype_out))

        # Shape term2, [n, 3]
        # Shape vec3_axis_arr, [n,3]
        vec3_axis_arr = np.tile(vec3_axis, (num_pnts,1))
        term2 = np.cross(vec3_axis_arr, pnts_arr_out)*np.sin(angle_rad, dtype=dtype_out)

        # Shape temp1, [3, n]
        temp1 = np.transpose(vec3_axis_arr)*np.dot(vec3_axis, np.transpose(pnts_arr_out))

        # Shape term3, [n, 3]
        term3 = np.transpose(temp1)*(1.0 - (np.cos(angle_rad, dtype=dtype_out)))

        pnts_arr_out = term1 + term2 + term3

    else:
        # By creating a for-loop, then only a single-point needs to be 
        # temporarily copied, thereby saving memory. However, this will be
        # slower than the vectorized method.
        for m in np.arange(0, num_pnts):
            cur_pnt = np.zeros(3, dtype=dtype_out)
            cur_pnt = pnts_arr_out[m,:]

            # Apply the Rodrigues rotation formula to a single point.
            cur_pnt = cur_pnt*(np.cos(angle_rad, dtype=dtype_out)) + \
                (np.cross(vec3_axis, cur_pnt))* \
                (np.sin(angle_rad, dtype=dtype_out)) + \
                vec3_axis*(np.dot(vec3_axis, cur_pnt))* \
                (1.0 - (np.cos(angle_rad, dtype=dtype_out)))

            # Store it back into the points array. Should not need .astype()
            pnts_arr_out[m,:] = cur_pnt.astype(dtype_out)

    # Translate the output points back
    np.add(pnts_arr_out, trans_vec3_back, pnts_arr_out)

    # Return the rotated points array
    return pnts_arr_out.astype(dtype_out)


def del_edge_particles_3D(imgs_bin_in, img_bound_in, scale_img=2.0):
    """
    Deletes particles that are touching the circular, virtual core
    inside of the image. It is assumed that the exported 2D images from
    X-ray CT contain a circular region inside the image that actually
    corresponds to the reconstructed data. When analyzing powder
    particles, it is important to analyze only particles that are fully
    captured inside of this circular region. Any particles partially
    clipped because they are touching the border of the  virtual core
    should be deleted, which is what this function does.

    ---- INPUT ARGUMENTS ---- 
    [[[imgs_bin_in]]]: A binarized 3D image of the particles. 
        imgs_bin_in should be a 3D Numpy array, and its data type should
        be uint8. It is assumed that the background is denoted by black
        pixels(with values equal to zero), and that the particles are
        made up of white pixels (with values equal to 255). The shape
        of imgs_bin_in is taken to be (num_imgs, num_rows, num_cols).

    [[img_bound_in]]: A single, grayscale 2D image of the particles 
        which will be used as reference for the interior, circular
        region. img_bound_in should be a 2D Numpy array, and its data
        type should be uint8. It is assumed that all of the pixels
        outside of the circular region are perfectly black
        (with intensity values equal to zero). Moreover, the pixels
        inside the circular region should be "not quite" black. The
        border of the circular region will be calculated based on
        thresholding this image using a threshold intensity equal to
        one. In effect, the circular region will be perfectly white,
        thus, the border easily calculated. The shape of img_bound_in
        should be (num_rows, num_cols).

    scale_img: A float that will be used to multiply the intensity
        values of img_bound_in. Any value that ends up being above 
        255 will be truncated to 255 so that the images still fits
        within the data type uint8. This scaling helps to ensure that
        all values inside of circular region will be greater than
        pure black, thereby improving robustness of the thresholding
        algorithm used to identify the boundary of the circular region.

    ---- RETURNED ----
    [[imgs_out]]: The returned image stack is based on imgs_bin_in but
        with the particles touching the circular boundaries of each
        image now filled with black pixels. Thus, it is a 3D Numpy
        array with data type uint8, and of the same size as
        imgs_bin_in. Moreover, just like imgs_bin_in contains only
        values equal to zero and 255, imgs_out will also contain values
        equal to zero and 255. Note, the circular boundary calculated
        based on thresholding imgs_orig_in using a value of 1.

    num_erased: An integer representing the number of times the flood-
        fill operation was performed. In other words, this integer
        corresponds to the number of particles (i.e., features) that
        were erased from the boundaries.

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the
    hard drive. Nothing is printed to the standard output stream.
    """

    img_bound = img_bound_in.copy() # Original image in grayscale (UINT8)
    imgs_bin = imgs_bin_in   # Binarized images (UINT8)
    scale_factor = scale_img

    # Initialize the returned binarized images
    imgs_out = imgs_bin_in.copy() 

    if scale_factor < 0.0:
        scale_factor = -1.0*scale_factor

    # Calculate the boundary pixels of the circular region. Also,
    # apply scaling factor to the grayscale images. The exterior
    # black pixels (i.e., intensity values are zero) are unaffected
    # by this scaling.
    img1_scaled = (img_bound.copy()).astype(np.uint16)
    img1_scaled = img1_scaled*scale_factor

    trunc_mask = np.nonzero(img1_scaled > 255)
    img1_scaled[trunc_mask] = 255

    img1_scaled = img1_scaled.astype(np.uint8)

    # Threshold just a tiny bit above pure black to identify the
    # circular boundary region of the "virtual core"
    cur_thsh = 1
    thsh_params = ["global", cur_thsh]
    img_temp = drv.apply_driver_thresh(img1_scaled, thsh_params, 
        quiet_in=True)

    # Label the connected features
    label_arr = meas.label(img_temp, return_num=False, connectivity=2)

    # Get the boundary of the virtual core (now in white)
    # lab_bound is the same shape as img_temp but dtype == bool
    lab_bound = seg.find_boundaries(label_arr, connectivity=2, 
        mode='inner', background=0)

    inv_bound_indices = np.nonzero(lab_bound == False)

    # Loop through the binary image stack
    num_erased = 0
    for img_ii, cur_img_bin in enumerate(imgs_bin):

        # Apply the boundary as a mask on the binarized image.
        # Everything but the boundary should be black
        img_bin_masked = cur_img_bin.copy()
        img_bin_masked[inv_bound_indices] = 0 

        # List of indices on the boundary that correspond to white pixels
        seed_indices = np.argwhere(img_bin_masked >= 1)

        # Perform a 3D flood-fill on the corresponding boundary particles
        for seed in seed_indices:
            seed_3d = (img_ii, seed[0], seed[1])

            # It is possible this pixel is no longer a white pixel on
            # border because the flood-fill from a neighboring pixel
            # may have already filled in this pixel as black. So, only
            # perform the flood-fill if it is still white. Note, this
            # flood-fill operation is in 3D.
            if imgs_out[seed_3d] >= 1:
                imgs_out = seg.flood_fill(imgs_out, seed_3d, 0, 
                                connectivity=2)
                num_erased += 1

    return [imgs_out, num_erased]


def extract_sphere(img_arr_in, radius=None):
    """
    Creates a new image sequence of the same size and type as the input
    image sequence, img_arr_in. The values in the returned image 
    sequence, however, are only copied over if they lie within the 
    sphere defined by the input radius. The sphere of copied intensities
    is centered about the central indices of the image sequence.

    ---- INPUT ARGUMENTS ---- 
    [[[img_arr_in]]]: A 3D Numpy array representing the image sequence
        It is important that this is a Numpy array and not a Python list
        of Numpy matrices. The shape of img_arr_in is expected to be as
        (num_images, num_pixel_rows, num_pixel_cols). Note, when
        converting this to spatial coordinates, num_images is the
        Z-component, num_pixel_rows is the Y-component, and
        num_pixel_cols is the X-component. It is expected that the
        images are single-channel, and the data should be of type uint8.
        Specifically, intensities should range from 0 to 255. If it is
        segmented (i.e., binarized), then only values of 0 and 255 
        should be present (NOT 0 and 1).

    radius: An integer representing the radius of the sphere to be 
        extracted from the image sequence (in pixels) with respect to
        the 3D center point of the image sequence. The default is to
        extract the largest sphere possible, but an integer value less
        than this can also be specified.

    ---- RETURNED ----
    [[[img_sphere]]]: A 3D Numpy array of the same size as img_arr_in
        and of the same data type. Only pixels that were within the
        spherical radius from img_arr_in were copied into img_sphere.

    ---- SIDE EFFECTS ----
    Function input arguments should not be altered. However, a new view
    is created for img_arr_in, not a deep copy. So, this cannot be 
    guaranteed. Nothing is written to the hard drive.
    """

    # Do NOT transpose here. My C-extension expects the image array
    # in image coordinates, and will return it correctly in XYZ.
    img_arr = img_arr_in

    n_imgs = img_arr_in.shape[0]
    n_rows = img_arr_in.shape[1]
    n_cols = img_arr_in.shape[2]

    if radius is None:
        n_min = np.amin(img_arr_in.shape)
        radius = np.floor(n_min/2.0) - 1
        radius = radius.astype(np.uint32)

    print("\nExtracting spherical subregion...")
    img_sphere = im3.extract_sphere(img_arr, radius)
    print("  Done!")

    return img_sphere


def line_fit_3d(pnts_in):
    """
    Parametrization of a 3D line, as a function of t, is defined as:

        L_vec = a_vec + n_vec * t

    where a_vec is the position vector of the mean point, and n_vec is
    the unit vector describing the direction of the line. For a given
    set of 3D points, this function calculates a_vec and n_vec for the
    best fit line.

    ---- INPUT ARGUMENTS ---- 
    [[pnts_in]]: A 2D numpy array containing all of the 3D points. For 
        n points, then the shape should be [n,3].

    ---- RETURNED ----
    [a_vec, n_vec]: The mean point, a_vec, and the unit vector, n_vec,
        of the best-fitting line are returned. The shape of a_vec will
        be [1,3]. The shape of n_vec will be [1,3]. 

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the
    hard drive. Nothing is printed to the standard output stream.
    """
    
    pnts = pnts_in.copy()

    # Using the mean point to normalize the values.
    # Will need to translate the resultant bounding box later
    mean_pnt = np.mean(pnts, axis=0)
    pnts = pnts - mean_pnt

    # Use SVD to calculate the principal directions
    #  pnts: [n, 3]
    #  u: [n, n]
    #  s: [3]   <--- Singular values sorted in descending order
    #  vh: [3, 3]
    # The rows of vh are the eigenvectors of pnts^T * pnts
    # The columns of u are the eigenvectors of pnts * pnts^T
    u, s, vh = np.linalg.svd(pnts)

    # Note, the following code calculates pnts via the equation:
    #    pnts = u*s_mat*vh
    #
    # ---- Uncomment Below To Activate Code ----
    #
    #s_mat = np.zeros( (u.shape[0], vh.shape[0]) )
    #for m, cur_s in enumerate(s):
    #    s_mat[m,m] = cur_s
    #    
    #pnts_copy = np.matmul(u, np.matmul(s_mat, vh) )

    # eig_vec1 corresponds to the unit vector pointed in the
    # direction of the best-fitting line
    eig_vec1 = vh[0,:]
    eig_vec2 = vh[1,:]
    eig_vec3 = vh[2,:]

    mag1 = np.sqrt( np.power(eig_vec1[0], 2) + 
                    np.power(eig_vec1[1], 2) +
                    np.power(eig_vec1[2], 2) )
    eig_vec1 = eig_vec1/mag1

    # Recall that the parametrization of a 3D line, as a function 
    # of t, is defined as:
    #   L_vec = a_vec + n_vec * t
    # where a_vec is the mean point, and n_vec is the unit
    # normal vector describing the direction of the line.

    # Ensure that the returned values are the correct shapes
    mean_pnt_out = np.array([0.0, 0.0, 0.0])
    eig_vec1_out = np.array([0.0, 0.0, 0.0])

    for ii, val in enumerate(mean_pnt):
        mean_pnt_out[ii] = val

    for ii, val in enumerate(eig_vec1):
        eig_vec1_out[ii] = val

    # In this case, a_vec is mean_pnt_out, and n_vec is eig_vec1_out.
    return [mean_pnt_out, eig_vec1_out]


def calc_centroid(imgs_bin_in, find_largest=False, round_2_int=False):
    """
    Calculate the centroid coordinates of a binarized image stack.
    Either the entire image stack can be chosen towards the calculation,
    or the centroid of the largest feature can be chosen. Both a 2D
    image and a 3D iamge stack are supported as inputs.

    ---- INPUT ARGUMENTS ---- 
    [[imgs_bin_in]]: A 2D or 3D Numpy array that has been segmented. 
        More specifically, any pixels with an intensity above zero will
        be considered for the calculation of the centroid. This array
        should be an integer data type.

    find_largest: A boolean to determine if all of the pixels should be
        utilized or if the centroid of the largest feature should be
        sought after instead. Set to False to consider the centroid of
        the entire array, and set to True to seek the centroid of the
        largest feature based on 1-connectivity.

    round_2_int: By default, the centroid coordinates are represented
        as floats. Set this to True to round the coordinates to the 
        closest whole integers. When True, the returned coordinates
        can then also be thought of as the image, row, and column
        indices of the image stack; if 2D, then just the row and
        column indices.

    ---- RETURNED ----
    [centroid_coord]: A 1D Numpy array is returned containing the
        coordinates of the centroid. If round_2_int is True, these
        coordinates will be returned as int64 data types, else, they
        will be retunred as float64 data types. centroid_coord will
        be of shape [1, ndim] where ndim is the number of dimensions
        of imgs_bin_in (either 2 or 3). If imgs_bin_in contains only
        zeros, then the returned values in centroid_coord will all 
        equal -1 and a warning will be printed to standard output.
        Note, the returned centroid coordinates will be in intrinsic
        coordinates. For example, for a 3D image stack, the values of
        the centroid will be ordered as: [img_value, row_value, 
        col_value].

    ---- SIDE EFFECTS ----
    Function input arguments are not altered. Nothing is written to the
    hard drive. A potential warning may be printed to the standard 
    output stream.
    """

    imgs_bin = imgs_bin_in
    num_dim = imgs_bin.ndim

    if num_dim == 2:
        centroid_coord = np.array([-1, -1])
    else:
        centroid_coord = np.array([-1, -1, -1])

    centroid_coord = centroid_coord.astype(np.int64)

    num_white_pix = np.count_nonzero(imgs_bin)

    if num_white_pix == 0:
        print("\nWarning: Number of white pixels found to be zero. The centroid")
        print(f"cannot be calculated. Returning centroid coordinates: {centroid_coord}")
        return centroid_coord

    if find_largest == False:
        # Should be more memory efficient than using regionprops() on the
        # entire image stack
        white_indices = np.argwhere(imgs_bin > 0)
        centroid_coord = np.mean(white_indices, axis=0, dtype=np.float64)

    else:
        label_arr, num_feats = meas.label(imgs_bin, return_num=True, connectivity=1)
        feat_props = meas.regionprops(label_arr)

        # Find the feature with the largest number of pixels
        max_area = 0
        fi = 0
        for indx in range(0,num_feats):
            cur_feat = feat_props[indx]
            if cur_feat.area > max_area:
                max_area = cur_feat.area
                fi = indx

        centroid_coord = np.array(feat_props[fi].centroid)

    if round_2_int:
        centroid_coord = (np.round(centroid_coord)).astype(np.int64)

    # If 2D: [row_value, column_value]
    # If 3D: [image_value, row_value, column_value]
    return centroid_coord

